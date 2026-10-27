from __future__ import annotations

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch

from .clearml_logger import ClearMLLogger
from .data import class_names
from .model_factory import build_model


def overlay(x, y, p) -> np.ndarray:
    rgb = x[:3].clone() if x.shape[0] >= 3 else x[:1].repeat(3, 1, 1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    out = rgb.clone()
    out[0] = torch.where(p[0] > 0.5, torch.ones_like(out[0]), out[0])   # prediction -> red
    out[1] = torch.where(y[0] > 0.5, torch.ones_like(out[1]), out[1])   # ground truth -> green
    return (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


class SegModule(L.LightningModule):
    """Single-device module (metrics are accumulated locally, not synced across ranks).

    Multi-label: the model outputs one logit channel per class (cfg.data.classes; one channel when none),
    each trained and scored as an independent binary mask. `val/iou` and `val/f1` are the MEAN over classes
    (it is the HPO objective); `val/iou_<class>` and `val/f1_<class>` are the per-class values."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg                       # Hydra config stays the source of truth (no save_hyperparameters)
        self.model = build_model(cfg)
        self.names = class_names(cfg.data)
        self.dice = smp.losses.DiceLoss("binary" if len(self.names) == 1 else "multilabel", from_logits=True)
        self.best_iou, self.best_epoch = -1.0, -1

    def loss(self, logits, y):
        c = self.cfg.loss
        # pos_weight: one number for every class, or a list with one number per class
        pw = None if c.pos_weight is None else torch.as_tensor(c.pos_weight, dtype=torch.float32, device=logits.device)
        if pw is not None and pw.ndim == 1:
            pw = pw.view(-1, 1, 1)
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
        self._counts = torch.zeros(len(self.names), 3, device=self.device)   # tp, fp, fn per class

    def validation_step(self, batch, batch_idx):
        x, y = batch
        lo = self(x)
        p = (lo.sigmoid() > 0.5).float()
        self._counts += torch.stack([(p * y).sum((0, 2, 3)), (p * (1 - y)).sum((0, 2, 3)),
                                     ((1 - p) * y).sum((0, 2, 3))], dim=1)
        self.log("val/loss", self.loss(lo, y), on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._sample = (x[0].float().cpu(), y[0].cpu(), lo[0].sigmoid().float().cpu())

    def on_validation_epoch_end(self):
        tp, fp, fn = self._counts.unbind(1)
        iou_c, f1_c = tp / (tp + fp + fn + 1e-8), 2 * tp / (2 * tp + fp + fn + 1e-8)   # one value per class
        iou, f1 = iou_c.mean(), f1_c.mean()
        self.log("val/iou", iou)          # <- HPO objective (title=val, series=iou), mean over classes
        self.log("val/f1", f1)
        if len(self.names) > 1:
            for n, i_, f_ in zip(self.names, iou_c, f1_c):
                self.log(f"val/iou_{n}", i_)
                self.log(f"val/f1_{n}", f_)
        if not self.trainer.sanity_checking:
            if iou.item() > self.best_iou:
                self.best_iou, self.best_epoch = iou.item(), self.current_epoch
            if isinstance(self.logger, ClearMLLogger):
                x, y, p = self._sample
                for c, n in enumerate(self.names):      # one overlay per class
                    self.logger.report_image(f"val/samples_{n}", "pred(red) vs gt(green)",
                                             overlay(x, y[c:c + 1], p[c:c + 1]), self.current_epoch)

    def configure_optimizers(self):
        o = self.cfg.optimizer
        opt = getattr(torch.optim, o.name)(self.parameters(), lr=o.lr, weight_decay=o.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.train.epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
