# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
GAN-style Domain Adaptation Validator.

This validator extends the standard DetectionValidator to also evaluate
the domain discriminator's performance on both source and target domains.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.domain_adapt.val import DomainAdaptationValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import autocast, smart_inference_mode, unwrap_model
from ultralytics.utils.tqdm import TQDM


class GANDomainAdaptationValidator(DomainAdaptationValidator):
    """GAN风格域适应验证器，同时验证检测性能和域分类器性能。
    
    Attributes:
        discriminator: 域分类器（判别器）引用
        domain_stats: 域分类统计信息字典
    
    Examples:
        >>> validator = GANDomainAdaptationValidator(
        ...     dataloader=source_val_loader,
        ...     discriminator=discriminator,
        ...     target_dataloader=target_val_loader,
        ... )
        >>> metrics = validator(trainer=trainer, model=model)
    """
    
    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        discriminator: torch.nn.Module = None,
        target_dataloader=None,
    ) -> None:
        """Initialize GAN domain adaptation validator.
        
        Args:
            dataloader: Source domain validation dataloader.
            save_dir: Directory to save results.
            args: Validation arguments.
            _callbacks: List of callback functions.
            discriminator: Domain discriminator model.
            target_dataloader: Target domain validation dataloader.
        """
        super().__init__(dataloader, save_dir, args, _callbacks, target_dataloader)
        self.discriminator = discriminator
        self.domain_stats = {
            "val_domain_loss": 0.0,
            "val_source_acc": 0.0,
            "val_target_acc": 0.0,
        }
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, discriminator=None):
        """Execute validation process.
        
        Args:
            trainer: Trainer object containing the model.
            model: Model to validate.
            discriminator: Domain discriminator (optional, overrides self.discriminator).
        
        Returns:
            dict: Dictionary containing validation metrics including domain classification stats.
        """
        # Update discriminator reference if provided
        if discriminator is not None:
            self.discriminator = discriminator
        
        if self.discriminator is None:
            LOGGER.warning("Domain discriminator not provided, skipping domain validation")
            # Fall back to standard validation
            return super().__call__(trainer, model)
        
        self.training = trainer is not None
        
        # Setup model
        model, augment = self._setup_model(trainer, model)
        
        self.run_callbacks("on_val_start")
        
        has_target_domain = self.target_dataloader is not None and len(self.target_dataloader) > 0
        
        # Initialize metrics
        self.init_metrics(unwrap_model(model))
        
        # Validate on source domain (detection + domain classification)
        dt_source, source_domain_preds = self._validate_with_domain(model, is_target=False)
        
        self.gather_stats(is_target=False)
        if RANK in {-1, 0}:
            stats = self.get_stats(is_target=False)
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt_source)))
            self.finalize_metrics(is_target=False)
            self.print_results(is_target=False)
        
        # Validate on target domain if available (detection + domain classification)
        if has_target_domain:
            dt_target, target_domain_preds = self._validate_with_domain(model, is_target=True)
            
            # Compute domain classification metrics
            if source_domain_preds is not None and target_domain_preds is not None:
                self._compute_domain_stats(source_domain_preds, target_domain_preds)
            
            self.gather_stats(is_target=True)
            if RANK in {-1, 0}:
                self.target_speed = dict(zip(self.speed.keys(), (x.t / len(self.target_dataloader.dataset) * 1e3 for x in dt_target)))
                target_stats = self.get_stats(is_target=True)
                target_stats = {f"target_{k}": v for k, v in target_stats.items()}
                self.finalize_metrics(is_target=True)
                self.print_results(is_target=True)  # 内部已包含域分类结果打印
        else:
            dt_target = None
        
        if self.training:
            # Domain loss is not directly used for training in GAN mode, but we track it
            self.loss = torch.zeros(4, device=self.device)  # box, cls, dfl, domain
        
        if RANK in {-1, 0}:
            self.run_callbacks("on_val_end")
        
        if self.training:
            model.float()
            
            # Merge all stats
            results = {
                **stats,
                **target_stats,
                **trainer.label_loss_items(self.loss, prefix="val"),
                **self.domain_stats,
            }
            
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            stats = {**stats, **target_stats, **self.domain_stats}
            return stats
    
    def _validate_with_domain(self, model, is_target: bool = False):
        """Run validation on a dataloader with domain classification.
        
        Args:
            model: Model to validate.
            is_target: Whether this is target domain validation.
        
        Returns:
            tuple: (dt, domain_preds) - Profiling timers and domain predictions.
        """
        
        dataloader = self.target_dataloader if is_target else self.dataloader
        desc_suffix = " (target)" if is_target else ""
        
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        
        if is_target:
            self.target_jdict = []
        else:
            self.jdict = []
        
        domain_preds = []
        bar = TQDM(dataloader, desc=self.get_desc() + desc_suffix, total=len(dataloader))
        
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            
            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=False)
            
            # Loss (only for source domain during training)
            with dt[2]:
                if self.training and not is_target:
                    loss_items = model.loss(batch, preds)[1]
                    self.loss[:3] += loss_items
                
                # Extract features and compute domain predictions
                actual_preds: dict = preds[1] # val模式下preds放在这里
                features = actual_preds["backbone_features"]
                
                with autocast(self.args.amp):
                    domain_logits = unwrap_model(self.discriminator)(features)
                
                domain_preds.append(domain_logits.detach())
            
            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)
            
            # Update metrics
            self.update_metrics(preds, batch, is_target=is_target)
            
            if self.args.plots and batch_i < 3 and RANK in {-1, 0} and not is_target:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            
            self.run_callbacks("on_val_batch_end")
        
        return dt, torch.cat(domain_preds) if domain_preds else None
    
    def _compute_domain_stats(self, source_preds: torch.Tensor, target_preds: torch.Tensor):
        """Compute domain classification statistics.
        
        Args:
            source_preds: Domain predictions for source domain samples.
            target_preds: Domain predictions for target domain samples.
        """
        if source_preds is None or target_preds is None:
            self.domain_stats = {
                "loss": 0,
                "total": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
            }
            return
        
        # Compute domain classification loss
        source_labels = torch.zeros_like(source_preds)
        target_labels = torch.ones_like(target_preds)
        
        loss_source = F.binary_cross_entropy_with_logits(source_preds, source_labels, reduction="sum")
        loss_target = F.binary_cross_entropy_with_logits(target_preds, target_labels, reduction="sum")
        domain_loss = loss_source + loss_target
        
        # Compute accuracies
        # Source domain: should be classified as 0 (logit < 0)
        source_correct = (source_preds < 0).sum().item()
        source_total = source_preds.numel()
        
        # Target domain: should be classified as 1 (logit >= 0)
        target_correct = (target_preds >= 0).sum().item()
        target_total = target_preds.numel()
        
        total = source_total + target_total
        accuracy = (source_correct + target_correct) / total if total > 0 else 0.0
        
        tp = target_correct
        fp = source_total - source_correct
        fn = target_total - target_correct
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        self.domain_stats = {
            "loss": domain_loss.item(),
            "total": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
    
    
    def get_desc(self) -> str:
        """Return description string for progress bar."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")