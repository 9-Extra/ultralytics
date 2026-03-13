# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, nms, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images


class DomainAdaptationValidator(BaseValidator):
    """A class extending the BaseValidator class for validation based on a domain adaptation detection model.

    This class implements validation functionality specific to object detection tasks with domain adaptation,
    including metrics calculation, prediction processing, and visualization of results.
    It supports simultaneous validation on both source domain and target domain datasets.

    Attributes:
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list[int]): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (list[Any]): List for storing ground truth labels for hybrid saving.
        jdict (list[dict[str, Any]]): List for storing JSON detection results.
        stats (dict[str, list[torch.Tensor]]): Dictionary for storing statistics during validation.
        target_data (dict): Target domain dataset information dictionary.
        target_dataloader (DataLoader): Target domain validation data loader.
        target_metrics (DetMetrics): Target domain metrics calculator.

    Examples:
        >>> from ultralytics.models.yolo.domain_adapt import DomainAdaptationValidator
        >>> args = dict(model="yolodan.pt", data="source_coco8.yaml", target_data="target_cityscapes.yaml")
        >>> validator = DomainAdaptationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize domain adaptation validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator. Can contain 'target_data' for target domain.
            _callbacks (list[Any], optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()
        
        # Target domain attributes
        self.target_data_path = getattr(self.args, "target_data", None)
        self.target_data = None
        self.target_dataloader = None
        self.target_metrics = None

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

    def init_metrics(self, model: torch.nn.Module, is_target: bool = False) -> None:
        """Initialize evaluation metrics for YOLO domain adaptation validation.

        Args:
            model (torch.nn.Module): Model to validate.
            is_target (bool): Whether initializing for target domain. Defaults to False.
        """
        data = self.target_data if is_target else self.data
        metrics = self.target_metrics if is_target else self.metrics
        
        val = data.get(self.args.split, "")  # validation path
        is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        is_lvis = isinstance(val, str) and "lvis" in val and not is_coco  # is LVIS
        
        if not is_target:
            self.is_coco = is_coco
            self.is_lvis = is_lvis
            self.class_map = converter.coco80_to_coco91_class() if is_coco else list(range(1, len(model.names) + 1))
            self.args.save_json |= self.args.val and (is_coco or is_lvis) and not self.training  # run final val
        
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        metrics.names = model.names
        
        if is_target:
            self.target_confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)
        else:
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
            is_target (bool): Whether updating for target domain. Defaults to False.
        """
        metrics = self.target_metrics if is_target else self.metrics
        confusion_matrix = self.target_confusion_matrix if is_target else self.confusion_matrix
        
        for si, pred in enumerate(preds):
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
            if self.args.plots:
                confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # Save
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
            is_target (bool): Whether finalizing for target domain. Defaults to False.
        """
        metrics = self.target_metrics if is_target else self.metrics
        confusion_matrix = self.target_confusion_matrix if is_target else self.confusion_matrix
        
        if self.args.plots:
            suffix = "_target" if is_target else ""
            for normalize in True, False:
                confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot, suffix=suffix)
        metrics.speed = self.speed
        metrics.confusion_matrix = confusion_matrix
        metrics.save_dir = self.save_dir

    def gather_stats(self, is_target: bool = False) -> None:
        """Gather stats from all GPUs.

        Args:
            is_target (bool): Whether gathering for target domain. Defaults to False.
        """
        metrics = self.target_metrics if is_target else self.metrics
        dataloader = self.target_dataloader if is_target else self.dataloader
        
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(metrics.stats, gathered_stats, dst=0)
            merged_stats = {key: [] for key in metrics.stats.keys()}
            for stats_dict in gathered_stats:
                for key in merged_stats:
                    merged_stats[key].extend(stats_dict[key])
            gathered_jdict = [None] * dist.get_world_size()
            dist.gather_object(self.jdict, gathered_jdict, dst=0)
            self.jdict = []
            for jdict in gathered_jdict:
                self.jdict.extend(jdict)
            metrics.stats = merged_stats
            self.seen = len(dataloader.dataset)  # total image count from dataset
        elif RANK > 0:
            dist.gather_object(metrics.stats, None, dst=0)
            dist.gather_object(self.jdict, None, dst=0)
            self.jdict = []
            metrics.clear_stats()

    def get_stats(self, is_target: bool = False) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Args:
            is_target (bool): Whether getting stats for target domain. Defaults to False.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        metrics = self.target_metrics if is_target else self.metrics
        metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        metrics.clear_stats()
        return metrics.results_dict

    def print_results(self, is_target: bool = False) -> None:
        """Print training/validation set metrics per class.

        Args:
            is_target (bool): Whether printing for target domain. Defaults to False.
        """
        metrics = self.target_metrics if is_target else self.metrics
        domain_prefix = "[Target Domain] " if is_target else "[Source Domain] "
        
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics.keys)  # print format
        LOGGER.info(domain_prefix + pf % ("all", self.seen, metrics.nt_per_class.sum(), *metrics.mean_results()))
        if metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"{domain_prefix}no labels found in {self.args.task} set, cannot compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(metrics.stats):
            for i, c in enumerate(metrics.ap_class_index):
                LOGGER.info(
                    domain_prefix + pf
                    % (
                        self.names[c],
                        metrics.nt_per_image[c],
                        metrics.nt_per_class[c],
                        *metrics.class_result(i),
                    )
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

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None, data: dict | None = None) -> torch.utils.data.Dataset:
        """Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.
            data (dict, optional): Dataset configuration dictionary. Defaults to self.data.

        Returns:
            (Dataset): YOLO dataset.
        """
        data = data or self.data
        return build_yolo_dataset(self.args, img_path, batch, data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int, data: dict | None = None) -> torch.utils.data.DataLoader:
        """Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.
            data (dict, optional): Dataset configuration dictionary. Defaults to self.data.

        Returns:
            (torch.utils.data.DataLoader): DataLoader for validation.
        """
        data = data or self.data
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val", data=data)
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int, is_target: bool = False) -> None:
        """Plot validation image samples.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
            is_target (bool): Whether plotting for target domain. Defaults to False.
        """
        suffix = "_target" if is_target else ""
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels{suffix}.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None, is_target: bool = False
    ) -> None:
        """Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (int | None): Maximum number of detections to plot.
            is_target (bool): Whether plotting for target domain. Defaults to False.
        """
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])  # convert to xywh format
        suffix = "_target" if is_target else ""
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred{suffix}.jpg",
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
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]
                    # record mAP for small, medium, large objects as well
                    stats["metrics/mAP_small(B)"] = val.stats_as_dict["AP_small"]
                    stats["metrics/mAP_medium(B)"] = val.stats_as_dict["AP_medium"]
                    stats["metrics/mAP_large(B)"] = val.stats_as_dict["AP_large"]
                    # update fitness
                    stats["fitness"] = 0.9 * val.stats_as_dict["AP_all"] + 0.1 * val.stats_as_dict["AP_50"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats

    def __call__(self, trainer=None, model=None):
        """Execute validation process on both source and target domains.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics for both domains.
        """
        # Run source domain validation using parent class
        source_stats = super().__call__(trainer, model)
        
        # If target data is specified, run target domain validation
        if self.target_data_path:
            target_stats = self._validate_target(trainer, model)
            # Combine stats with prefix for target domain
            if target_stats:
                for key, value in target_stats.items():
                    source_stats[f"target_{key}"] = value
        
        return source_stats
    
    def _validate_target(self, trainer, model) -> dict[str, Any]:
        """Validate model on target domain dataset.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module): Model to validate.

        Returns:
            (dict[str, Any]): Dictionary containing target domain validation statistics.
        """
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import TQDM
        from ultralytics.utils.ops import Profile
        from ultralytics.utils.torch_utils import unwrap_model
        
        if self.target_data is None:
            # Load target domain dataset
            self.target_data = check_det_dataset(self.target_data_path)
            # Create target dataloader   
            target_path = self.target_data.get(self.args.split)
            if target_path is None:
                raise Exception(f"No validation split found in target dataset: {self.args.split}")
            self.target_dataloader = self.get_dataloader(target_path, self.args.batch, data=self.target_data)
        
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
            assert False, "todo"
            
        # Initialize target metrics
        self.target_metrics = DetMetrics()
        
        # Run validation on target domain
        augment = self.args.augment
        
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.target_dataloader, desc=self.get_desc(), total=len(self.target_dataloader))
        self.init_metrics(unwrap_model(model), is_target=True)
        self.jdict = []  # empty before each val
        
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch, is_target=True)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i, is_target=True)
                self.plot_predictions(batch, preds, batch_i, is_target=True)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats(is_target=True)
        if RANK in {-1, 0}:
            stats = self.get_stats(is_target=True)
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.target_dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics(is_target=True)
            self.print_results(is_target=True)
            self.run_callbacks("on_val_end")
        
        return stats
