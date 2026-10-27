"""Model factory: Hydra `instantiate` over segmentation_models_pytorch (+ optional torchgeo encoder weights)."""
from __future__ import annotations

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    # cfg.model = {_target_: smp.create_model, arch, encoder_name, encoder_weights, in_channels, classes, ...}
    model = instantiate(cfg.model, _convert_="all")

    pre = cfg.get("pretrain")
    if pre is not None and pre.source == "torchgeo":
        import torchgeo.models as tgm

        weights = getattr(tgm, pre.weights_enum)[pre.weights_name]
        expected = weights.meta.get("in_chans")
        if expected is not None and expected != cfg.model.in_channels:
            raise ValueError(f"{pre.weights_name} expects {expected} channels, data.in_channels="
                             f"{cfg.model.in_channels}")
        missing, unexpected = model.encoder.load_state_dict(weights.get_state_dict(), strict=False)
        print(f"[factory] torchgeo {pre.weights_name}: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            raise RuntimeError(f"Encoder keys not matched by torchgeo weights: {missing[:5]}")
    return model
