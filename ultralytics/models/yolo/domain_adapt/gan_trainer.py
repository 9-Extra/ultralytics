# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
GAN-style Domain Adaptation Trainer for YOLO.

This module implements a GAN-style training approach for domain adaptation:
- Generator: YOLOv26 backbone + detection head
- Discriminator: Independent domain classifier (Fully Convolutional Network)
- Training strategy: Alternate optimization (k steps for D, 1 step for G)
"""

from __future__ import annotations

import math
import random
import time
import warnings
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import GradScaler, distributed as dist

from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.domain_adapt.gan_validator import (
    GANDomainAdaptationValidator,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import (
    attempt_compile,
    autocast,
    torch_distributed_zero_first,
    unwrap_model,
)
from ultralytics.utils.tqdm import TQDM


class DomainDiscriminator(nn.Module):
    """独立域分类器（判别器）- 全卷积网络实现。

    接收YOLO骨干网络提取的多尺度特征，输出域分类预测。
    完全由卷积层组成，支持任意空间尺寸的输入。
    源域=0，目标域=1。

    Architecture:
        Input: List[Tensor] - 多尺度特征 [(B,C1,H1,W1), (B,C2,H2,W2), ...]
        → Per-scale conv encoding (1x1 conv reduce channels)
        → Spatial pooling (adaptive avg pool to fixed size)
        → Concatenate multi-scale features
        → 1x1 conv fusion layers
        → Global average pooling
        → Output: (B, 1) domain logits
    """

    def __init__(self, ch: tuple = (256, 512, 1024), hidden_dim: int = 256):
        """
        Args:
            ch: 输入特征通道数列表，对应多尺度特征
            hidden_dim: 隐藏层维度（中间特征通道数）
        """
        super().__init__()
        self.nl = len(ch)  # 特征层数
        self.hidden_dim = hidden_dim

        # 为每层构建特征编码器（全卷积）
        self.encoders = nn.ModuleList()
        # 计算每层的输出通道数，确保总和等于 hidden_dim
        channels_per_layer = [hidden_dim // self.nl] * self.nl
        # 将余数分配给最后一层
        remainder = hidden_dim - sum(channels_per_layer)
        if remainder > 0:
            channels_per_layer[-1] += remainder

        for i, c in enumerate(ch):
            out_ch = channels_per_layer[i]
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_ch, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.InstanceNorm2d(out_ch, affine=True),
                    nn.Conv2d(out_ch, out_ch, 1),
                )
            )

        # 特征融合层（全卷积）
        # 输入: hidden_dim (concatenated from all scales)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(hidden_dim // 2, affine=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 1, 1),
            nn.ReLU(inplace=True),
        )

        # 分类头（1x1 卷积）
        self.classifier = nn.Conv2d(hidden_dim // 2, 1, 1, 1, 0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """前向传播 - 全卷积，支持任意空间尺寸。

        Args:
            features: 多尺度特征列表，每个元素形状为 (B, C, H, W)
                     其中 H, W 可以是任意值

        Returns:
            域分类logits，形状为 (B, 1)
        """
        # 编码并统一空间尺寸
        encoded = []
        target_size = (features[2].shape[-2], features[2].shape[-1])
        for i, feat in enumerate(features):
            # 1x1 卷积降维
            enc = self.encoders[i](feat)  # (B, hidden_dim//nl, H, W)

            # 使用自适应池化统一到相同空间尺寸（保持通道数不变）
            if enc.shape[2:] != target_size:
                enc = F.adaptive_avg_pool2d(enc, target_size)

            encoded.append(enc)

        # 通道维度拼接 (B, hidden_dim, H, W)
        fused = torch.cat(encoded, dim=1)

        # 融合特征（全卷积）
        fused = self.fusion(fused)  # (B, hidden_dim//2, H, W)

        # 分类（1x1 卷积）
        logits_map = self.classifier(fused)  # (B, 1, H, W)

        # 全局平均池化得到最终分类结果
        logits = F.adaptive_avg_pool2d(logits_map, 1)  # (B, 1, 1, 1)
        logits = logits.view(-1, 1)  # (B, 1)

        return logits


class GANDomainAdaptationTrainer(BaseTrainer):
    """GAN风格域适应训练器。

    将YOLOv26骨干网络视为生成器，域分类器视为判别器，
    采用交替优化策略：
    - 判别器优化 k 次/批次
    - 生成器优化 1 次/批次

    Attributes:
        discriminator: 独立域分类器网络
        d_optimizer: 判别器优化器
        d_steps: 每批次判别器优化次数
        d_lr: 判别器学习率
        lambda_adv: 对抗损失权重
    """

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None
    ):
        # 提取GAN特有参数，避免传递给父类时出错
        gan_params = {}
        if overrides:
            gan_keys = [
                "d_steps",
                "d_lr",
                "lambda_adv",
                "lambda_gp",
                "discriminator_hidden",
                "discriminator_ch",
            ]
            for key in gan_keys:
                if key in overrides:
                    gan_params[key] = overrides.pop(key)

        super().__init__(cfg, overrides, _callbacks)

        # GAN特有参数
        self.d_steps = gan_params.get("d_steps", 5)
        self.d_lr = gan_params.get("d_lr", 0.0001)
        self.lambda_adv = gan_params.get("lambda_adv", 0.1)
        self.lambda_gp = gan_params.get("lambda_gp", 10.0)
        self.discriminator_hidden = gan_params.get("discriminator_hidden", 256)

        self.discriminator_amp = self.amp and True # 判别器训练是否使用半精度
        # 待初始化
        self.discriminator = None
        self.d_optimizer = None
        self.d_scaler = None  # 判别器独立的 GradScaler
        self.target_train_loader = None
        self.target_iter = None

        # 训练统计
        self.train_domain_stats = {}  # 每epoch前会重新初始化
        self.last_losses = None

    def _setup_discriminator(self):
        """初始化域分类器（判别器），自动检测输入通道数。"""
        # 从 DetectGAN 检测头的 cv2/cv3 层获取输入通道数
        head = self.model.model[-1]  # DetectGAN head
        if hasattr(head, "cv2") and len(head.cv2) > 0:
            actual_ch = []
            for cv in head.cv2:
                ch_in = cv[0].conv.in_channels
                actual_ch.append(ch_in)
            self.discriminator_ch = actual_ch
            LOGGER.info(f"自动检测判别器输入通道: {self.discriminator_ch}")

        self.discriminator = DomainDiscriminator(
            ch=self.discriminator_ch, hidden_dim=self.discriminator_hidden
        )

        if self.args.compile:
            self.discriminator = attempt_compile(self.discriminator, self.device)

        self.discriminator.to(self.device)

        # 判别器优化器 (WGAN-GP 推荐 Adam with beta1=0)
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.9)
        )

        # 判别器独立的 GradScaler
        self.d_scaler = GradScaler("cuda", enabled=self.discriminator_amp)

        LOGGER.info(
            f"域分类器初始化完成: input_ch={self.discriminator_ch}, hidden={self.discriminator_hidden}"
        )
        LOGGER.info(
            f"WGAN-GP训练参数: d_steps={self.d_steps}, d_lr={self.d_lr}, "
            f"lambda_adv={self.lambda_adv}, lambda_gp={self.lambda_gp}"
        )

    def get_dataset(self):
        """获取源域和目标域数据集。"""
        from ultralytics.data.utils import check_det_dataset

        data = check_det_dataset(self.args.data)
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1

        if self.args.target_data:
            self.target_data = check_det_dataset(self.args.target_data)
            LOGGER.info(f"目标域数据集加载完成: {self.args.target_data}")
        else:
            self.target_data = None
            raise ValueError("GAN domain adaptation requires target_data")

        return data

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """构造数据加载器。"""
        from ultralytics.data import build_dataloader, build_yolo_dataset

        with torch_distributed_zero_first(rank):
            dataset = build_yolo_dataset(
                self.args,
                dataset_path,
                batch_size,
                self.data,
                mode=mode,
                rect=mode == "val",
            )

        shuffle = mode == "train"
        loader = build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )
        return loader

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: int | None = None,
        data: dict | None = None,
    ):
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
        return build_yolo_dataset(
            self.args, img_path, batch, data, mode=mode, rect=mode == "val", stride=gs
        )

    def _prepare_target_dataloader(
        self, dataset_path, batch_size=16, rank=0, mode="train"
    ):
        """准备目标域数据加载器。"""

        with torch_distributed_zero_first(rank):
            target_dataset = self.build_dataset(
                dataset_path,
                mode=mode,
                batch=batch_size,
                data=self.target_data,
            )

        shuffle = mode == "train"
        if (
            getattr(target_dataset, "rect", False)
            and shuffle
            and not np.all(
                target_dataset.batch_shapes == target_dataset.batch_shapes[0]
            )
        ):
            LOGGER.warning(
                "'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
            )
            shuffle = False

        self.target_train_loader = build_dataloader(
            target_dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )
        self.target_iter = iter(self.target_train_loader)
        LOGGER.info(f"目标域数据加载器创建完成，共 {len(target_dataset)} 张图像")

    def preprocess_batch(self, batch):
        """预处理批次数据，包括源域和目标域图像。"""
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
                    math.ceil(x * sf / self.stride) * self.stride
                    for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(
                    imgs, size=ns, mode="bilinear", align_corners=False
                )
            batch["img"] = imgs

        # 加载目标域
        assert hasattr(self, "target_iter") and self.target_data
        try:
            target_batch = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.target_train_loader)
            target_batch = next(self.target_iter)

        # 将目标域图像移到设备并预处理
        target_img = target_batch["img"].to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        target_img = target_img.float() / 255

        # 应用与源域相同的多尺度处理
        if self.args.multi_scale > 0.0 and sf != 1 and ns is not None:
            target_img = nn.functional.interpolate(
                target_img, size=ns, mode="bilinear", align_corners=False
            )

        batch["domain_img"] = target_img
        return batch

    def set_model_attributes(self):
        """设置模型属性。"""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回YOLO检测模型。"""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回GAN域适应验证器，支持域分类器验证。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "adv_loss"

        # 创建目标域验证加载器
        target_val_loader = None
        if hasattr(self, "target_data") and self.target_data:
            try:
                target_val_path = self.target_data.get(
                    self.args.split
                ) or self.target_data.get("val")
                if target_val_path:
                    # 计算batch_size，与源域验证保持一致
                    batch_size = self.batch_size // max(self.world_size, 1)
                    # 非OBB任务使用2倍batch_size（与原始DomainAdaptationTrainer一致）
                    val_batch_size = (
                        batch_size if self.args.task == "obb" else batch_size * 2
                    )
                    target_val_loader = self.get_dataloader(
                        target_val_path, batch_size=val_batch_size, mode="val"
                    )
                    LOGGER.info(
                        f"目标域验证数据加载器创建完成 (batch_size={val_batch_size})"
                    )
            except Exception as e:
                LOGGER.warning(f"无法创建目标域验证数据加载器: {e}")

        # 返回GAN专用验证器，传入判别器
        return GANDomainAdaptationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            discriminator=self.discriminator,  # 关键：传入判别器
            target_dataloader=target_val_loader,
        )

    def validate(self):
        """执行验证，确保判别器正确传递给验证器。

        Returns:
            tuple: (metrics, fitness)
        """
        # 确保判别器在正确的设备上并处于评估模式
        if self.discriminator is not None:
            self.discriminator.eval()

        # 调用验证器，传入判别器（返回字典）
        self.metrics = self.validator(
            trainer=self,
            model=self.model,
            discriminator=self.discriminator,
        )

        # 如果验证被跳过，返回 None
        if self.metrics is None:
            return None, None

        # 提取 fitness（与父类 BaseTrainer.validate 一致）
        self.fitness = self.metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < self.fitness:
            self.best_fitness = self.fitness

        return self.metrics, self.fitness

    def _gradient_penalty(
        self,
        real_features: list[torch.Tensor],
        fake_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """计算多尺度特征输入下的 Gradient Penalty."""
        batch_size = real_features[0].size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)

        interpolates = []
        for real, fake in zip(real_features, fake_features):
            interp = alpha * real + (1 - alpha) * fake
            # 强制 float32 全精度，避免 AMP 低精度下二阶梯度数值不稳定
            interp = interp.float()
            interp.requires_grad_(True)
            interpolates.append(interp)

        with autocast(False):
            d_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,  # penalty用于损失计算后，其梯度为二阶梯度，因此需要为“求梯度”本身创建计算图
            retain_graph=True,
            only_inputs=True,
        )

        penalty = 0.0
        for grad in gradients:
            grad = grad.view(grad.size(0), -1)
            penalty += ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def _train_discriminator(self, epoch, batch):
        """训练判别器（k次迭代）- WGAN-GP."""

        if not epoch >= self.epochs // 20:
            # 等骨干网络稍微收敛再训练
            return

        # 提取特征（不计算梯度，保持生成器在 train 模式）
        with torch.no_grad():
            with autocast(self.amp):
                source_features: list[torch.Tensor] = self.model(batch["img"])[
                    "backbone_features"
                ]
                target_features: list[torch.Tensor] = self.model(batch["domain_img"])[
                    "backbone_features"
                ]
            if not self.discriminator_amp:    
                source_features = [f.float() for f in source_features]
                target_features = [f.float() for f in target_features]

        # 训练判别器 k 次
        # losses = []
        for _ in range(self.d_steps):
            self.d_optimizer.zero_grad()

            with autocast(self.discriminator_amp):
                d_source = self.discriminator(source_features)
                d_target = self.discriminator(target_features)

                # Wasserstein 距离: 最大化 D(target) - D(source)
                # score越高，表示判别器认为其偏离了源域
                wasserstein = d_target.mean() - d_source.mean()
                gp = self._gradient_penalty(source_features, target_features)
                loss_d = -wasserstein + self.lambda_gp * gp
            
            self.d_scaler.scale(loss_d).backward()
            self.d_scaler.step(self.d_optimizer)
            self.d_scaler.update()
            
            # with torch.no_grad():
            #     losses.append(torch.stack((loss_d, wasserstein, gp, d_source.max(), d_target.max())))
        
        pass
    
        # losses = torch.stack(losses)
        # print(losses)
        # if self.last_losses is not None:
        #     print(self.last_losses)
        # self.last_losses = losses.to(device="cpu", non_blocking=True)
        

    def _train_generator(self, epoch: int, batch):
        """训练生成器，同时统计训练集域分类准确率 - WGAN-GP."""
        source_preds = self.model(batch["img"])

        if self.args.compile:
            loss_det, loss_items = unwrap_model(self.model).loss(batch, source_preds)
        else:
            loss_det, loss_items = self.model.loss(batch, source_preds)

        if epoch >= self.epochs // 20:
            # 从1/20轮后再开始对抗训练
            target_preds = self.model(batch["domain_img"])

            d_target_scores: torch.Tensor = self.discriminator(
                target_preds["backbone_features"]
            )

            # WGAN 生成器损失：最小化 D(target)，即让目标域 score 接近源域
            loss_adv = d_target_scores.mean()

            with torch.no_grad():
                # 同时计算 source score 用于统计准确率和平均分数
                d_source_scores = self.discriminator(
                    source_preds["backbone_features"]
                )
                # 累加在 GPU tensor 上，避免每个 batch 都同步到 CPU
                self.train_domain_stats["source_score_sum"] += d_source_scores.sum()
                self.train_domain_stats["target_score_sum"] += d_target_scores.sum()
                self.train_domain_stats["source_count"] += d_source_scores.numel()
                self.train_domain_stats["target_count"] += d_target_scores.numel()
        else:
            loss_adv = torch.tensor(0, device=self.device, dtype=loss_det.dtype)

        total_loss = loss_det.sum() + self.lambda_adv * loss_adv

        loss_items = torch.cat([loss_items, loss_adv.detach().unsqueeze(0)])

        return total_loss, loss_items

    def _do_train(self):
        """执行GAN风格的训练循环。"""
        if self.world_size > 1:
            self._setup_ddp()

        self._setup_train()
        self._setup_discriminator()
        self._prepare_target_dataloader(
            self.target_data.get("train", self.target_data.get("path")),
            batch_size=self.batch_size // max(self.world_size, 1),
            rank=RANK,
            mode="train",
        )

        nb = len(self.train_loader)
        nw = (
            max(round(self.args.warmup_epochs * nb), 100)
            if self.args.warmup_epochs > 0
            else -1
        )
        last_opt_step = -1

        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        LOGGER.info(f"Starting GAN-style training for {self.epochs} epochs...")

        epoch = self.start_epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)

            pbar = enumerate(self.train_loader)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)

            self.tloss = None
            # 使用 GPU tensor 累加统计量，仅在 epoch 结束时做一次 .item() 同步
            device = self.device
            self.train_domain_stats = {
                "source_score_sum": torch.tensor(0.0, dtype=torch.float32, device=device),
                "target_score_sum": torch.tensor(0.0, dtype=torch.float32, device=device),
                "source_count": torch.tensor(0, dtype=torch.int64, device=device),
                "target_count": torch.tensor(0, dtype=torch.int64, device=device),
            }

            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch

                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(
                        1,
                        int(
                            np.interp(
                                ni, xi, [1, self.args.nbs / self.batch_size]
                            ).round()
                        ),
                    )
                    for x in self.optimizer.param_groups:
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                (
                                    0.0
                                    if x.get("param_group") != "bias"
                                    else self.args.warmup_bias_lr
                                ),
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni, xi, [self.args.warmup_momentum, self.args.momentum]
                            )

                batch = self.preprocess_batch(batch)

                self._train_discriminator(epoch, batch)

                with autocast(self.amp):
                    self.loss, self.loss_items = self._train_generator(epoch, batch)

                    if RANK != -1:
                        self.loss *= self.world_size

                    self.tloss = (
                        self.loss_items
                        if self.tloss is None
                        else (self.tloss * i + self.loss_items) / (i + 1)
                    )

                self.scaler.scale(self.loss).backward()

                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(
                                self.tloss
                                if loss_length > 1
                                else torch.unsqueeze(self.tloss, 0)
                            ),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )

                self.run_callbacks("on_train_batch_end")
            pass # batch end
        
            # epoch 结束时统一将 GPU 统计量同步到 CPU（仅一次）
            source_count = self.train_domain_stats["source_count"].item()
            train_source_score = (
                self.train_domain_stats["source_score_sum"].item() / source_count
                if source_count > 0 else 0.0
            )
            target_count = self.train_domain_stats["target_count"].item()
            train_target_score = (
                self.train_domain_stats["target_score_sum"].item() / target_count
                if target_count > 0 else 0.0
            )

            self.lr = {
                f"lr/pg{ir}": x["lr"]
                for ir, x in enumerate(self.optimizer.param_groups)
            }
            self.run_callbacks("on_train_epoch_end")

            if RANK in {-1, 0}:
                self.ema.update_attr(
                    self.model, include=["yaml", "nc", "args", "names", "stride"]
                )

            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue
            
            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            if RANK in {-1, 0}:
                self.save_metrics(
                    metrics={
                        **self.label_loss_items(self.tloss, prefix="train"),
                        **self.metrics,
                        **self.lr,
                        "train/source_scores": train_source_score,
                        "train/target_scores": train_target_score,
                        "val/source_scores": self.validator.domain_stats.get(
                            "source_score", 0.0
                        ),
                        "val/target_scores": self.validator.domain_stats.get(
                            "target_score", 0.0
                        ),
                    }
                )

                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch

                if self.args.save or final_epoch:
                    self.save_model()

            self.run_callbacks("on_fit_epoch_end")

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(
                    broadcast_list, 0
                )  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(
            f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours."
        )
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        from ultralytics.utils.torch_utils import unset_deterministic

        unset_deterministic()
        self.run_callbacks("teardown")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """返回带标签的损失字典。"""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """返回训练进度字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )
