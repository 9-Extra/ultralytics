# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
import time
from typing import Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.modules.head import DetectGRL
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tqdm import TQDM
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import autocast, torch_distributed_zero_first, unset_deterministic, unwrap_model

class DomainAdaptationTrainer(BaseTrainer):
    """用于域自适应检测模型的训练器，同时加载源域和目标域数据集。

    该训练器专门用于域自适应目标检测任务，支持同时加载源域（带标签）和目标域（不带标签）数据集。
    源域使用现有逻辑进行训练，目标域图像经过与源域相同的预处理后，放入 batch["domain_img"] 中，
    用于域分类器的训练。

    Attributes:
        model (DetectionModel): YOLO检测模型。
        data (dict): 源域数据集信息字典。
        target_data (dict): 目标域数据集信息字典。
        target_train_loader (DataLoader): 目标域训练数据加载器。
        target_iter (Iterator): 目标域数据加载器的迭代器。

    Methods:
        build_dataset: 构建YOLO数据集。
        get_dataset: 获取源域和目标域数据集。
        get_dataloader: 构造源域和目标域数据加载器。
        preprocess_batch: 预处理批次数据，包括目标域图像。
        set_model_attributes: 根据数据集信息设置模型属性。
        get_model: 返回YOLO域自适应检测模型。
        get_validator: 返回验证器。
        label_loss_items: 返回带标签的损失字典。
        progress_string: 返回格式化的训练进度字符串。
        plot_training_samples: 绘制训练样本。
        plot_training_labels: 创建带标签的训练图。
        auto_batch: 计算最优批次大小。

    Examples:
        >>> from ultralytics.models.yolo.domain_adapt import DomainAdaptationTrainer
        >>> args = dict(model="yolodan.yaml", data="source_coco8.yaml", target_data="target_cityscapes.yaml", epochs=3)
        >>> trainer = DomainAdaptationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """初始化 DomainAdaptationTrainer，支持目标域数据集。

        Args:
            cfg (dict, optional): 默认配置字典。
            overrides (dict, optional): 配置覆盖字典，可包含 target_data 指定目标域数据集路径。
            _callbacks (list, optional): 回调函数列表。
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None, data: dict | None = None):
        """构建YOLO数据集。

        Args:
            img_path (str): 图像文件夹路径。
            mode (str): 'train' 或 'val' 模式。
            batch (int, optional): 批次大小，用于 'rect' 模式。
            data (dict, optional): 数据集配置字典，默认为 self.data。

        Returns:
            (Dataset): YOLO数据集对象。
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        data = data or self.data
        return build_yolo_dataset(self.args, img_path, batch, data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataset(self):
        """获取源域和目标域数据集。

        Returns:
            (dict): 包含源域数据集信息的字典。
        """
        # 获取源域数据集（使用父类逻辑）
        from ultralytics.data.utils import check_det_dataset
        
        try:
            # 转换 ul:// 平台 URI 和 NDJSON 文件
            data_str = str(self.args.data)
            if data_str.endswith(".ndjson") or (data_str.startswith("ul://") and "/datasets/" in data_str):
                import asyncio
                from ultralytics.data.converter import convert_ndjson_to_yolo
                from ultralytics.utils.checks import check_file
                self.args.data = str(asyncio.run(convert_ndjson_to_yolo(check_file(self.args.data))))
            
            # 检测任务数据集检查
            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect", "segment", "pose", "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]
        except Exception as e:
            from ultralytics.utils import emojis
            raise RuntimeError(emojis(f"Dataset '{self.args.data}' error ❌ {e}")) from e
        
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        
        # 如果指定了目标域数据路径，加载目标域数据集
        if self.args.target_data:
            self.target_data = check_det_dataset(self.args.target_data)
            LOGGER.info(f"目标域数据集加载完成: {self.args.target_data}")
        else:
            self.target_data = None
            
        return data

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """构造源域和目标域数据加载器。

        Args:
            dataset_path (str): 源域数据集路径。
            batch_size (int): 批次大小。
            rank (int): 进程 rank（用于分布式训练）。
            mode (str): 'train' 或 'val' 模式。

        Returns:
            (DataLoader): 源域数据加载器（训练时还会创建目标域数据加载器）。
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        
        # 创建源域数据加载器
        loader = build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )
        
        # 只在训练模式下创建目标域数据加载器
        if mode == "train" and self.target_data:
            with torch_distributed_zero_first(rank):
                target_dataset = self.build_dataset(
                    self.target_data.get("train", self.target_data.get("path")),
                    mode="train",
                    batch=batch_size,
                    data=self.target_data
                )
                self.target_train_loader = build_dataloader(
                    target_dataset,
                    batch=batch_size,
                    workers=self.args.workers,
                    shuffle=True,
                    rank=rank,
                    drop_last=self.args.compile,
                )
                # 创建迭代器，用于循环获取目标域数据
                self.target_iter = iter(self.target_train_loader)
                LOGGER.info(f"目标域数据加载器创建完成，共 {len(target_dataset)} 张图像")
        
        return loader

    def preprocess_batch(self, batch: dict) -> dict:
        """预处理批次数据，包括源域和目标域图像。

        Args:
            batch (dict): 包含源域批次数据的字典。

        Returns:
            (dict): 预处理的批次数据，包含目标域图像（domain_img）。
        """
        # 预处理源域批次
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        
        # 多尺度处理
        sf = 1.0  # 默认缩放因子
        ns = None  # 新尺寸
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        
        # 加载并预处理目标域图像
        assert hasattr(self, "target_iter") and self.target_data
        try:
            # 获取目标域批次
            target_batch = next(self.target_iter)
        except StopIteration:
            # 如果目标域数据用完，重置迭代器
            self.target_iter = iter(self.target_train_loader)
            target_batch = next(self.target_iter)
        
        # 将目标域图像移到设备并预处理
        target_img = target_batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        target_img = target_img.float() / 255
        
        # 应用与源域相同的多尺度处理
        if self.args.multi_scale > 0.0 and sf != 1 and ns is not None:
            target_img = nn.functional.interpolate(target_img, size=ns, mode="bilinear", align_corners=False)
        
        # 将目标域图像放入批次中
        batch["domain_img"] = target_img
        
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)
        
        # 设置 DetectGRL 的 grl_weight
        grl_weight = getattr(self.args, "grl_weight", -0.1)
        head = self.model.model[-1]
        if isinstance(head, DetectGRL):
            head.grl_weight = grl_weight
            LOGGER.info(f"已设置 DetectGRL 模块的 grl_weight = {grl_weight}")
        else:
            LOGGER.info(f"DetectGRL 模块未找到")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO domain adaptation detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO domain adaptation detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DomainAdaptationValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "dom_loss"
        
        # Create target domain validation dataloader if target_data is available
        target_val_loader = None
        if hasattr(self, "target_data") and self.target_data is not None:
            try:
                target_val_path = self.target_data.get(self.args.split) or self.target_data.get("val")
                if target_val_path:
                    target_val_loader = self.get_dataloader(target_val_path, self.args.batch, mode="val")
                    LOGGER.info(f"目标域验证数据加载器创建完成，用于验证阶段")
            except Exception as e:
                LOGGER.warning(f"无法创建目标域验证数据加载器: {e}")
        
        return yolo.domain_adapt.DomainAdaptationValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args), 
            _callbacks=self.callbacks,
            target_dataloader=target_val_loader
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)
    
    def _do_train(self):
        """Perform the full training loop including setup, epoch iteration, validation, and final evaluation."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        self._oom_retries = 0  # OOM auto-reduce counter for first epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    
                    preds = self.model(batch["img"])
                    if self.args.compile:
                        # Decouple inference and loss calculations for improved compile performance
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model.loss(batch, preds)

                    original_yolo_loss = loss.sum() # original_yolo_loss
                    
                    source_domain_preds = preds["domain_pred"]
                    head = unwrap_model(self.model).model[-1]
                    backbone_neck = unwrap_model(self.model).model[:-1]  # 除 head 外的所有层
                    
                    # 根据 domain_batchnorm_update 参数决定是否冻结 BatchNorm
                    domain_batchnorm_update = self.args.domain_batchnorm_update
                    if not domain_batchnorm_update:
                        # 冻结所有 BatchNorm 的统计量更新，防止目标域数据影响 running statistics
                        for m in backbone_neck.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.eval()  # 切换到 eval 模式，禁用 running statistics 更新

                    # 跳过目标域分类器
                    head.domain_classify_only = True                    
                    target_domain_preds = self.model(batch["domain_img"])["domain_pred"]
                    
                    # 恢复训练模式
                    head.domain_classify_only = False
                    if not domain_batchnorm_update:
                        for m in backbone_neck.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                m.train()

                    sd_loss = torch.nn.functional.binary_cross_entropy_with_logits(source_domain_preds, torch.zeros_like(source_domain_preds), reduction="sum")
                    td_loss = torch.nn.functional.binary_cross_entropy_with_logits(target_domain_preds, torch.ones_like(target_domain_preds), reduction="sum")
                    domain_loss_weight = getattr(self.args, "domain_loss_weight", 0.2)
                    domain_loss = (sd_loss + td_loss) * domain_loss_weight
                    
                    # 计算域分类准确率
                    with torch.no_grad():
                        # Source domain: 预测 < 0.5 为正确 (label=0)
                        source_correct = (source_domain_preds < 0).sum().item()
                        # Target domain: 预测 >= 0.5 为正确 (label=1)
                        target_correct = (target_domain_preds >= 0).sum().item()
                        total_samples = source_domain_preds.numel() + target_domain_preds.numel()
                        domain_correct = source_correct + target_correct
                        domain_accuracy = domain_correct / total_samples if total_samples > 0 else 0.0
                    
                    # 合并loss
                    self.loss = original_yolo_loss + domain_loss
                    self.loss_items =  torch.cat((self.loss_items, domain_loss.detach().unsqueeze_(dim=0)))
                    # 保存域分类准确率用于后续 metrics
                    self.domain_acc = domain_accuracy
                    
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = (
                        self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                    )
                pass

                # Backward
                self.scaler.scale(self.loss).backward()    
                    
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break  # allow external stop (e.g. platform cancellation) between batches
            else:
                # for/else: this block runs only when the for loop completes without break (no OOM retry)
                self._oom_retries = 0  # reset OOM counter after successful first epoch

            if self._oom_retries and not self.stop:
                continue  # OOM recovery broke the for loop, restart with reduced batch size

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr, "domain_acc": getattr(self, "domain_acc", 0.0)})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")
