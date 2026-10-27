"""Model factory: Hydra `instantiate` over segmentation_models_pytorch (+ optional local / torchgeo encoder weights)."""
from __future__ import annotations

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


def _load_encoder(encoder: nn.Module, state: dict, label: str) -> None:
    """Load `state` into the encoder and fail on missing keys.
    smp's encoders override load_state_dict and return None, so the keys are compared here."""
    wanted, given = set(encoder.state_dict()), set(state)
    missing, unexpected = sorted(wanted - given), sorted(given - wanted)
    encoder.load_state_dict(state, strict=False)
    print(f"[factory] {label}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        raise RuntimeError(f"Encoder keys not matched by {label}: {missing[:5]}")


def build_model(cfg: DictConfig) -> nn.Module:
    # cfg.model = {_target_: smp.create_model, arch, encoder_name, encoder_weights, in_channels, classes, ...}
    model = instantiate(cfg.model, _convert_="all")

    pre = cfg.get("pretrain")
    if pre is not None and pre.source == "local":
        # offline-safe: encoder weights from a file made by `python -m src.export_weights`
        _load_encoder(model.encoder, torch.load(pre.path, map_location="cpu", weights_only=True),
                      f"local weights {pre.path}")
    elif pre is not None and pre.source == "torchgeo":
        import torchgeo.models as tgm

        weights = getattr(tgm, pre.weights_enum)[pre.weights_name]
        expected = weights.meta.get("in_chans")
        if expected is not None and expected != cfg.model.in_channels:
            raise ValueError(f"{pre.weights_name} expects {expected} channels, data.in_channels="
                             f"{cfg.model.in_channels}")
        _load_encoder(model.encoder, weights.get_state_dict(), f"torchgeo {pre.weights_name}")
    return model
