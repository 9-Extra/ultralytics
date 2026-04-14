# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
GAN-style Domain Adaptation Validator.

This validator extends the standard DetectionValidator to also evaluate
the domain discriminator's performance on both source and target domains.
"""

import torch
import torch.distributed as dist

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
            "loss": 0.0,
            "source_score": 0.0,
            "target_score": 0.0,
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
        target_stats = {}  # Initialize target_stats
        
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
            
            # Merge all stats
            results = {
                **stats,
                **target_stats,
                **trainer.label_loss_items(loss / len(self.dataloader), prefix="val")
            }
            
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            stats = {**stats, **target_stats}
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
        """Compute domain statistics for WGAN-GP.
        
        Args:
            source_preds: Domain critic scores for source domain samples.
            target_preds: Domain critic scores for target domain samples.
        """
        if source_preds is None or target_preds is None:
            self.domain_stats = {
                "loss": 0.0,
                "source_score": 0.0,
                "target_score": 0.0,
            }
            return
        
        # WGAN domain loss: Wasserstein distance estimate
        domain_loss = target_preds.mean() - source_preds.mean()
        
        self.domain_stats = {
            "loss": domain_loss.item(),
            "source_score": source_preds.mean().item(),
            "target_score": target_preds.mean().item(),
        }
    
    
    def print_results(self, is_target: bool = False) -> None:
        """Print validation results with WGAN-GP domain metrics.
        
        Args:
            is_target: Whether this is target domain validation.
        """
        from ultralytics.utils import LOGGER
        
        metrics = self.target_metrics if is_target else self.metrics
        seen = len(self.target_dataloader.dataset) if is_target else self.seen
        domain_label = "Target" if is_target else "Source"
        
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(metrics.keys)
        LOGGER.info(pf % ("all", seen, metrics.nt_per_class.sum(), *metrics.mean_results()))
        if metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set {domain_label}, cannot compute metrics without labels")

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
        
        if is_target and self.domain_stats is not None:
            LOGGER.info(f"{'Domain Metrics:':>22}{'loss':>11s}{'source_score':>11s}{'target_score':>11s}")
            LOGGER.info(
                f"{'Domain:':>22}{self.domain_stats['loss']:>11.3f}{self.domain_stats['source_score']:>11.3f}{self.domain_stats['target_score']:>11.3f}"
            )

    def get_desc(self) -> str:
        """Return description string for progress bar."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")