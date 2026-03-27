# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, nms, ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, RANK, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model
from ultralytics.utils.loss import DomainLoss


class DomainAdaptationValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None, target_dataloader=None) -> None:
        """Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator.
            _callbacks (list[Any], optional): List of callback functions.
            target_dataloader (torch.utils.data.DataLoader, optional): Target domain DataLoader for validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()
        self.target_dataloader = target_dataloader
        self.target_metrics = DetMetrics() if target_dataloader is not None else None
        self.domain_stats = None  # Domain classification statistics
    
    def _setup_model(self, trainer=None, model=None):
        """Setup model for validation.
        
        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.
            
        Returns:
            tuple: (model, augment) - The prepared model and augment flag.
        """
        augment = self.args.augment and (trainer is None)
        
        if trainer is not None:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod  # validate non-compiled original model to avoid issues
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            if hasattr(model, "end2end"):
                if self.args.end2end is not None:
                    model.end2end = self.args.end2end
                if model.end2end:
                    model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit = model.stride, model.pt, model.jit
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup
            
        return model, augment

    def _validate_dataloader(self, dataloader, model, augment, desc_suffix=""):
        """Run validation on a specific dataloader.
        
        Args:
            dataloader (DataLoader): DataLoader to validate.
            model (nn.Module): Model to use for validation.
            augment (bool): Whether to use augmentation.
            desc_suffix (str): Suffix to add to progress bar description.
            
        Returns:
            tuple: (dt, domain_pred) - Profiling timers and domain perdiction.
        """
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        
        # Reset metrics and jdict for this validation run
        if desc_suffix == " (target)":
            self.target_jdict = []
        else:
            self.jdict = []
        
        domain_pred = []
        bar = TQDM(dataloader, desc=self.get_desc() + desc_suffix, total=len(dataloader))
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    # 计算检测 loss
                    loss_items = model.loss(batch, preds)[1]
                    if desc_suffix == "":
                        # 源域：计算检测 loss，最后的domain_loss在__call__中计算后写入
                        self.loss[:3] += loss_items
                
                # domain_loss作为重要指标无论是否训练都收集
                actual_preds: dict = preds[1] # val模式下preds放在这里
                # 收集 domain_pred，用于后续计算 domain_loss
                if actual_preds.get("domain_pred", None) is not None:
                    domain_pred.append(actual_preds["domain_pred"])
                    
            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch, is_target=(desc_suffix != ""))
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                if desc_suffix == "":
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
            
        return dt, torch.cat(domain_pred) if len(domain_pred) > 0 else None

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        target_stats = {}  # Initialize target_stats
        
        # Setup model
        model, augment = self._setup_model(trainer, model)
        
        self.run_callbacks("on_val_start")
        
        has_target_domain = self.target_dataloader is not None and len(self.target_dataloader) > 0
        # Initialize metrics
        self.init_metrics(unwrap_model(model))
        
        # Validate on source domain
        dt_source, source_domain_preds = self._validate_dataloader(self.dataloader, model, augment, desc_suffix="")
        
        self.gather_stats(is_target=False)
        if RANK in {-1, 0}:
            stats = self.get_stats(is_target=False)
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt_source)))
            self.finalize_metrics(is_target=False)
            self.print_results(is_target=False)

        # Validate on target domain if available
        if has_target_domain:
            dt_target, target_domain_preds = self._validate_dataloader(self.target_dataloader, model, augment, desc_suffix=" (target)")
            # 计算 domain_loss（需要同时有源域和目标域）
            if source_domain_preds is not None and target_domain_preds is not None:
                # 使用 DomainLoss 计算 domain_loss（包含标签平滑）
                domain_loss = DomainLoss(epsilon=0.1)(
                    source_domain_preds,
                    target_domain_preds
                ) * self.args.domain_loss_weight
            else:
                domain_loss = torch.tensor(0.0, device=self.device) # Yolov26原始模型没有域分类头，相关指标为0           
            
            self.gather_stats(is_target=True)
            if RANK in {-1, 0}:
                self.target_speed = dict(zip(self.speed.keys(), (x.t / len(self.target_dataloader.dataset) * 1e3 for x in dt_target)))
                target_stats = self.get_stats(is_target=True)  # Compute stats for target domain
                # Add target_ prefix to target domain metrics
                target_stats = {f"target_{k}": v for k, v in target_stats.items()}
                self.finalize_metrics(is_target=True)
                # 计算域分类统计信息
                self._compute_domain_stats(source_domain_preds, target_domain_preds, domain_loss)
                self.print_results(is_target=True)
        else:
            dt_target = None
            domain_loss = torch.tensor(0.0, device=self.device) # 只有源域则 domain_loss 保持为 0
            
        
        if self.training:
            self.loss[3] = domain_loss.detach() # 写入domain_loss
        
        if RANK in {-1, 0}:
            self.run_callbacks("on_val_end")

        if self.training:
            model.float()
            # Reduce loss across all GPUs
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            
            if RANK > 0:
                return
            
            # Merge source and target stats, add target_ prefix to target metrics
            results = {**stats, **target_stats, **trainer.label_loss_items(loss / len(self.dataloader), prefix="val")}
            
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            # Merge source and target stats for non-training mode
            stats = {**stats, **target_stats}
            if RANK > 0:
                return stats
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if dt_target is not None:
                LOGGER.info(
                    "Target Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                        *tuple(self.target_speed.values())
                    )
                )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats


    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names
        if self.target_metrics is not None:
            self.target_metrics.names = model.names
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains 'bboxes', 'conf',
                'cls', and 'extra' tensors.
        """
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch of images and annotations for validation.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare predictions for evaluation against ground truth.

        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any], is_target: bool = False) -> None:
        """Update metrics with new predictions and ground truth.

        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            batch (dict[str, Any]): Batch data containing ground truth.
            is_target (bool): Whether this is target domain validation.
        """
        metrics = self.target_metrics if is_target else self.metrics
        jdict = getattr(self, "target_jdict", []) if is_target else self.jdict
        
        for si, pred in enumerate(preds):
            if not is_target:
                self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            # Evaluate
            if self.args.plots and not is_target:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # Save (only for source domain)
            if not is_target:
                if self.args.save_json or self.args.save_txt:
                    predn_scaled = self.scale_preds(predn, pbatch)
                if self.args.save_json:
                    self.pred_to_json(predn_scaled, pbatch)
                if self.args.save_txt:
                    self.save_one_txt(
                        predn_scaled,
                        self.args.save_conf,
                        pbatch["ori_shape"],
                        self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                    )

    def finalize_metrics(self, is_target: bool = False) -> None:
        """Set final values for metrics speed and confusion matrix.
        
        Args:
            is_target (bool): Whether this is target domain validation.
        """
        metrics = self.target_metrics if is_target else self.metrics
        
        if self.args.plots and not is_target:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        
        if is_target:
            metrics.speed = self.target_speed
        else:
            metrics.speed = self.speed
        metrics.confusion_matrix = self.confusion_matrix
        metrics.save_dir = self.save_dir

    def gather_stats(self, is_target: bool = False) -> None:
        """Gather stats from all GPUs.
        
        Args:
            is_target (bool): Whether this is target domain validation.
        """
        metrics = self.target_metrics if is_target else self.metrics
        jdict = getattr(self, "target_jdict", []) if is_target else self.jdict
        dataloader = self.target_dataloader if is_target else self.dataloader
        
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(metrics.stats, gathered_stats, dst=0)
            merged_stats = {key: [] for key in metrics.stats.keys()}
            for stats_dict in gathered_stats:
                for key in merged_stats:
                    merged_stats[key].extend(stats_dict[key])
            gathered_jdict = [None] * dist.get_world_size()
            dist.gather_object(jdict, gathered_jdict, dst=0)
            if is_target:
                self.target_jdict = []
                for jd in gathered_jdict:
                    self.target_jdict.extend(jd)
            else:
                self.jdict = []
                for jd in gathered_jdict:
                    self.jdict.extend(jd)
            metrics.stats = merged_stats
            if not is_target:
                self.seen = len(dataloader.dataset)  # total image count from dataset
        elif RANK > 0:
            dist.gather_object(metrics.stats, None, dst=0)
            dist.gather_object(jdict, None, dst=0)
            if is_target:
                self.target_jdict = []
            else:
                self.jdict = []
            metrics.clear_stats()

    def get_stats(self, is_target: bool = False) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Args:
            is_target (bool): Whether this is target domain validation.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        metrics = self.target_metrics if is_target else self.metrics
        metrics.process(save_dir=self.save_dir, plot=self.args.plots and not is_target, on_plot=self.on_plot)
        metrics.clear_stats()
        return metrics.results_dict

    def _compute_domain_stats(self, source_preds: torch.Tensor, target_preds: torch.Tensor, domain_loss: torch.Tensor) -> None:
        """Compute domain classification statistics and store in self.domain_stats.
        
        Args:
            source_preds: Source domain predictions (logits).
            target_preds: Target domain predictions (logits).
            domain_loss: Domain classification loss.
        """
        if source_preds is None or target_preds is None:
            self.domain_stats = {
                "loss": domain_loss.item(),
                "correct": 0,
                "error": 0,
                "total": 0,
                "accuracy": 0,
            }
            return
        
        # 将 logits 转换为预测标签 (>=0.5 预测为目标域/1, <0.5 预测为源域/0)
        source_pred_labels = (source_preds >= 0).long()  # sigmoid(0) = 0.5
        target_pred_labels = (target_preds >= 0).long()
        
        # 源域标签为 0，目标域标签为 1
        source_true_labels = torch.zeros_like(source_pred_labels)
        target_true_labels = torch.ones_like(target_pred_labels)
        
        # 计算正确数和错误数
        source_correct = (source_pred_labels == source_true_labels).sum().item()
        source_error = source_pred_labels.numel() - source_correct
        target_correct = (target_pred_labels == target_true_labels).sum().item()
        target_error = target_pred_labels.numel() - target_correct
        
        correct = source_correct + target_correct
        error = source_error + target_error
        total = correct + error
        accuracy = correct / total if total > 0 else 0.0
        
        self.domain_stats = {
            "loss": domain_loss.item(),
            "correct": correct,
            "error": error,
            "total": total,
            "accuracy": accuracy,
        }
    
    def print_results(self, is_target: bool = False) -> None:
        """Print training/validation set metrics per class.
        
        Args:
            is_target (bool): Whether this is target domain validation.
        """
        metrics = self.target_metrics if is_target else self.metrics
        seen = len(self.target_dataloader.dataset) if is_target else self.seen
        domain_label = "Target" if is_target else "Source"
        
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics.keys)  # print format
        LOGGER.info(pf % ("all", seen, metrics.nt_per_class.sum(), *metrics.mean_results()))
        if metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set {domain_label}, cannot compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(metrics.stats):
            for i, c in enumerate(metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        metrics.nt_per_image[c],
                        metrics.nt_per_class[c],
                        *metrics.class_result(i),
                    )
                )
        
        # Print domain classification results if available (only for target domain validation)
        if is_target and self.domain_stats is not None:
            # Print header for domain classification metrics
            LOGGER.info(f"{'Domain Metrics:':>22}{'loss':>11s}{'correct':>11s}{'error':>11s}{'total':>11s}{'accuracy':>11s}")
            LOGGER.info(
                f"{'Domain:':>22}{self.domain_stats['loss']:>11.3f}{self.domain_stats['correct']:>11d}{self.domain_stats['error']:>11d}{self.domain_stats['total']:>11d}{self.domain_stats['accuracy']:>11.3f}"
            )

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrix.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for
                10 IoU levels.
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): DataLoader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation image samples.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (int | None): Maximum number of detections to plot.
        """
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
                bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Examples:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict[str, Any]): Current statistics dictionary.

        Returns:
            (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
    ) -> dict[str, Any]:
        """Evaluate COCO/LVIS metrics using faster-coco-eval library.

        Performs evaluation using the faster-coco-eval library to compute mAP metrics for object detection. Updates the
        provided stats dictionary with computed metrics including mAP50, mAP50-95, and LVIS-specific metrics if
        applicable.

        Args:
            stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path): Path to JSON file containing predictions in COCO format.
            anno_json (str | Path): Path to JSON file containing ground truth annotations in COCO format.
            iou_types (str | list[str]): IoU type(s) for evaluation. Can be single string or list of strings. Common
                values include "bbox", "segm", "keypoints". Defaults to "bbox".
            suffix (str | list[str]): Suffix to append to metric names in stats dictionary. Should correspond to
                iou_types if multiple types provided. Defaults to "Box".

        Returns:
            (dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
        return stats
