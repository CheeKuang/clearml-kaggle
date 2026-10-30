"""Model factory: Hydra `instantiate` over segmentation_models_pytorch (+ optional local / torchgeo encoder weights)."""
from __future__ import annotations

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    """cfg.model = {_target_: smp.create_model, arch, encoder_name, encoder_weights, in_channels, classes, ...}.
    `pretrain.source` smp: create_model already loaded the weights; local / torchgeo: they replace the encoder's."""
    model = instantiate(cfg.model, _convert_="all") # construct the model using Hydra's instantiate and convert all to plain python so smp can understand
    pre = cfg.get("pretrain") # get the pretrain configuration if it exists example unet 
    source = pre.source if pre is not None else None
    if source == "local":       # offline-safe: a file made by `python -m src.export_weights`
        state = torch.load(pre.path, map_location="cpu", weights_only=True)
        _load_encoder(model.encoder, state, f"local weights {pre.path}")
    elif source == "torchgeo":
        _load_encoder(model.encoder, _torchgeo_state(pre, cfg.model.in_channels), f"torchgeo {pre.weights_name}")
    return model


def _torchgeo_state(pre: DictConfig, in_channels: int) -> dict:
    import torchgeo.models as tgm

    weights = getattr(tgm, pre.weights_enum)[pre.weights_name]
    expected = weights.meta.get("in_chans")
    if expected is not None and expected != in_channels:
        raise ValueError(f"{pre.weights_name} expects {expected} channels, data.in_channels={in_channels}")
    return weights.get_state_dict()


def _load_encoder(encoder: nn.Module, state: dict, label: str) -> None:
    """Load `state` and fail on missing keys. smp's encoders override load_state_dict and return None, so the keys
    are compared here."""
    missing = sorted(set(encoder.state_dict()) - set(state))
    unexpected = sorted(set(state) - set(encoder.state_dict()))
    encoder.load_state_dict(state, strict=False)
    print(f"[factory] {label}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        raise RuntimeError(f"Encoder keys not matched by {label}: {missing[:5]}")
