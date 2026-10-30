"""LightningModule: model + BCE/Dice loss + per-class IoU/F1 (multi-label: every class is its own binary mask)."""
from __future__ import annotations

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from ..data.tiles import class_names
from ..model_factory import build_model
from .logger import ClearMLLogger


def confusion(pred, y, dims) -> torch.Tensor:
    """tp, fp, fn summed over `dims`, stacked on a new last axis."""
    tp, fp, fn = pred * y, pred * (1 - y), (1 - pred) * y
    return torch.stack([tp.sum(dims), fp.sum(dims), fn.sum(dims)], -1)


def iou_f1(counts):
    tp, fp, fn = counts.unbind(-1)
    return tp / (tp + fp + fn + 1e-8), 2 * tp / (2 * tp + fp + fn + 1e-8)


def overlay(x, y, p) -> np.ndarray:
    """Image with the prediction in red and the ground truth in green."""
    rgb = x[:3].clone() if x.shape[0] >= 3 else x[:1].repeat(3, 1, 1)
    out = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    out[0] = torch.where(p[0] > 0.5, 1.0, out[0])
    out[1] = torch.where(y[0] > 0.5, 1.0, out[1])
    return (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


class SegModule(L.LightningModule):
    """Single device (metrics are not synced across ranks). `val/iou` and `val/f1` are the MEAN over classes (the
    checkpoint and HPO objective); `val/iou_<class>` / `val/f1_<class>` are logged when there are several classes."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg                          # the Hydra config stays the source of truth (no save_hyperparameters)
        self.model = build_model(cfg)
        self.names = class_names(cfg.data)
        self.dice = smp.losses.DiceLoss("binary" if len(self.names) == 1 else "multilabel", from_logits=True)
        self.best_iou, self.best_epoch = -1.0, -1

    def forward(self, x):
        return self.model(x)

    def loss(self, logits, y):
        c = self.cfg.loss
        pos_weight = None
        if c.pos_weight is not None:            # one number, or one per class (broadcast over H, W)
            pos_weight = torch.as_tensor(c.pos_weight, dtype=torch.float32, device=logits.device)
            if pos_weight.ndim == 1:
                pos_weight = pos_weight.view(-1, 1, 1)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        return c.bce_weight * bce + c.dice_weight * self.dice(logits, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=False, on_epoch=True)  # cosine schedule
        return loss

    def on_validation_epoch_start(self):
        self._counts = torch.zeros(len(self.names), 3, device=self.device)     # per class: tp, fp, fn

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = logits.sigmoid()
        self._counts += confusion((probs > 0.5).float(), y, (0, 2, 3))
        self.log("val/loss", self.loss(logits, y), on_step=False, on_epoch=True)
        if batch_idx == 0:                      # the picture sent to ClearML at the end of the epoch
            self._sample = (x[0].float().cpu(), y[0].cpu(), probs[0].float().cpu())

    def on_validation_epoch_end(self):
        iou, f1 = iou_f1(self._counts)          # one value per class
        metrics = {"val/iou": iou.mean(), "val/f1": f1.mean()}
        if len(self.names) > 1:
            metrics |= {f"val/iou_{n}": v for n, v in zip(self.names, iou)}
            metrics |= {f"val/f1_{n}": v for n, v in zip(self.names, f1)}
        self.log_dict(metrics)
        if self.trainer.sanity_checking:
            return
        if metrics["val/iou"].item() > self.best_iou:
            self.best_iou, self.best_epoch = metrics["val/iou"].item(), self.current_epoch
        if isinstance(self.logger, ClearMLLogger):
            x, y, p = self._sample
            for c, name in enumerate(self.names):
                self.logger.report_image(f"val/samples_{name}", "pred(red) vs gt(green)",
                                         overlay(x, y[c:c + 1], p[c:c + 1]), self.current_epoch)

    def configure_optimizers(self):
        o = self.cfg.optimizer
        optimizer = getattr(torch.optim, o.name)(self.parameters(), lr=o.lr, weight_decay=o.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.train.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
