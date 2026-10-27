"""python -m src.export_weights [--encoder resnet34] [--weights imagenet] [--out weights/resnet34_imagenet.pth]

Run once while online: saves the pretrained encoder state dict to a plain file so training can build the
model without reaching the Hugging Face hub (conf/encoder/resnet34_local.yaml). Copy the file to the
offline environment (image, volume, or ClearML artifact)."""
from __future__ import annotations

import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="resnet34")
ap.add_argument("--weights", default="imagenet")
ap.add_argument("--out", default=None)
a = ap.parse_args()

out = Path(a.out or f"weights/{a.encoder}_{a.weights}.pth")
out.parent.mkdir(parents=True, exist_ok=True)
model = smp.create_model("Unet", encoder_name=a.encoder, encoder_weights=a.weights)
torch.save(model.encoder.state_dict(), out)
print(f"saved {out} ({out.stat().st_size / 1e6:.0f} MB)")
