from __future__ import annotations

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch

from .clearml_logger import ClearMLLogger
from .model_factory import build_model


def overlay(x, y, p) -> np.ndarray:
    rgb = x[:3].clone() if x.shape[0] >= 3 else x[:1].repeat(3, 1, 1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    out = rgb.clone()
    out[0] = torch.where(p[0] > 0.5, torch.ones_like(out[0]), out[0])   # prediction -> red
    out[1] = torch.where(y[0] > 0.5, torch.ones_like(out[1]), out[1])   # ground truth -> green
    return (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


class SegModule(L.LightningModule):
    """Single-device module (metrics are accumulated locally, not synced across ranks)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg                       # Hydra config stays the source of truth (no save_hyperparameters)
        self.model = build_model(cfg)
        self.dice = smp.losses.DiceLoss("binary", from_logits=True)
        self.best_iou, self.best_epoch = -1.0, -1

    def loss(self, logits, y):
        c = self.cfg.loss
        pw = None if c.pos_weight is None else torch.tensor(float(c.pos_weight), device=logits.device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        return c.bce_weight * bce + c.dice_weight * self.dice(logits, y)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._counts = torch.zeros(3, device=self.device)   # tp, fp, fn

    def validation_step(self, batch, batch_idx):
        x, y = batch
        lo = self(x)
        p = (lo.sigmoid() > 0.5).float()
        self._counts += torch.stack([(p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y).sum()])
        self.log("val/loss", self.loss(lo, y), on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._sample = (x[0].float().cpu(), y[0].cpu(), lo[0].sigmoid().float().cpu())

    def on_validation_epoch_end(self):
        tp, fp, fn = self._counts
        iou, f1 = tp / (tp + fp + fn + 1e-8), 2 * tp / (2 * tp + fp + fn + 1e-8)
        self.log("val/iou", iou)          # <- HPO objective (title=val, series=iou)
        self.log("val/f1", f1)
        if not self.trainer.sanity_checking:
            if iou.item() > self.best_iou:
                self.best_iou, self.best_epoch = iou.item(), self.current_epoch
            if isinstance(self.logger, ClearMLLogger):
                self.logger.report_image("val/samples", "pred(red) vs gt(green)",
                                         overlay(*self._sample), self.current_epoch)

    def configure_optimizers(self):
        o = self.cfg.optimizer
        opt = getattr(torch.optim, o.name)(self.parameters(), lr=o.lr, weight_decay=o.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.train.epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
