"""Export a trained model to ONNX in the form the Chisel server loads, and check it against PyTorch.

  # from a finished training run (best.ckpt + metadata.json are artifacts of the train Task)
  python -m src.export_onnx --checkpoint best.ckpt --metadata metadata.json --out models/vegetation.onnx
  # untrained model, only to test sizes / speed / the serving path
  python -m src.export_onnx --random --data vegetation --architecture unet --encoder resnet34 --out models/vegetation.onnx

Contract of the exported file (matches libs/preprocess.py of the server):
  input   "images"  float32 (1, 3, S, S)   RGB scaled to 0..1 (no mean/std), letterboxed to S x S, pad value 114
  output  "probs"   float32 (1, C, S, S)   sigmoid probabilities, C = number of classes (1 per production model)
  S = --size, default 1280 (the server's input size)
A sidecar <out>.json carries the class name(s) and the threshold the server should use (best at TTA "none",
because the server does no test-time augmentation). The server needs the matching post-processing:
serving/postprocess_semantic.py.
After training, src.training.train also shrinks the file (shrunk/<class>.onnx, the one for production), writes the
train Task id, git commit and the sha256 of both files into the sidecar, and checks both through the server's
pre/post-processing (validate_onnx).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .model_factory import build_model


def load_cfg(a):
    if a.metadata:
        return OmegaConf.create(json.loads(Path(a.metadata).read_text())["config"])
    return compose_config([f"data={a.data}", f"architecture={a.architecture}", f"encoder={a.encoder}"])


def compose_config(overrides: list[str]):
    """Hydra config from the repo's conf/, inside or outside a running Hydra app."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        return compose("config", overrides=overrides)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "conf"), version_base="1.3"):
        return compose("config", overrides=overrides)


def threshold_from_rows(rows: list[dict], metric: str, class_name: str) -> float | None:
    """Best threshold among the sweep rows with tta == "none" (the server runs no TTA)."""
    rows = [r for r in rows if r["tta"] == "none"]
    if not rows:
        return None
    key = f"{metric}_{class_name}" if f"{metric}_{class_name}" in rows[0] else metric
    return float(max(rows, key=lambda r: r[key])["threshold"])


def best_threshold_without_tta(metadata_path: str | None, class_name: str) -> float | None:
    if not metadata_path:
        return None
    meta = json.loads(Path(metadata_path).read_text())
    return threshold_from_rows(meta["sweep"]["results"], meta["config"]["sweep"]["metric"], class_name)


def export_model(network: torch.nn.Module, out: Path, names: list[str], size: int = 1280, opset: int = 17,
                 threshold: float | None = None, architecture: str = "") -> dict:
    """Wrap `network` (logits) with a sigmoid, export to ONNX, check onnxruntime == PyTorch, write the sidecar json."""
    import onnxruntime as ort

    model = torch.nn.Sequential(network.cpu().eval(), torch.nn.Sigmoid()).eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, size, size)
    torch.onnx.export(model, dummy, str(out), input_names=["images"], output_names=["probs"],
                      opset_version=opset, dynamo=False)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(["probs"], {"images": dummy.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"[export] {out.name} ({out.stat().st_size / 1e6:.1f} MB) output {got.shape} max |torch - onnx| = {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError(f"ONNX parity failed: {diff}")
    side = {"classes": names, "input_size": size, "input": "RGB float32 0..1 letterboxed, pad 114",
            "output": "sigmoid probabilities (1, C, S, S)", "tta": "none", "threshold": threshold,
            "architecture": architecture, "opset": opset}
    out.with_suffix(".onnx.json").write_text(json.dumps(side, indent=2))
    return side


def shrink_and_record(onnx_path: Path, shrink, source: dict) -> dict[str, Path]:
    """Shrink `onnx_path` (if enabled) to shrunk/<same name>, then add `source` (task, commit) and the sha256 and size
    of every file to the sidecar, so a file in production can be traced to its run. Returns {artifact name: path}."""
    from .shrink import shrink_onnx_weights
    from .training.metadata import sha256_file

    files = {"onnx": onnx_path}
    if shrink.enabled:
        files["onnx_shrunk"] = onnx_path.parent / "shrunk" / onnx_path.name
        shrink_onnx_weights(str(onnx_path), str(files["onnx_shrunk"]), method=shrink.method, ir_version=shrink.ir_version)
    side_path = onnx_path.with_suffix(".onnx.json")
    side = json.loads(side_path.read_text())
    side["source"] = source
    side["files"] = {k: {"file": str(p.relative_to(onnx_path.parent)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                     for k, p in files.items()}
    side_path.write_text(json.dumps(side, indent=2))
    return files


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """The server's libs/preprocess.letterbox (auto=False): fit into size x size, pad 114 on both sides."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)), r, (dw, dh)


def validate_onnx(files: dict[str, Path], ds, threshold: float, size: int) -> dict[str, float]:
    """IoU of each single-class ONNX file on `ds`, through the server's path: uint8 tile -> letterbox -> RGB/255 ->
    onnxruntime CPU -> serving/postprocess_semantic.py -> 0/255 mask. Same tiles and threshold for every file."""
    import onnxruntime as ort

    from serving.postprocess_semantic import postprocess_semantic_masks

    sessions = {k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for k, p in files.items()}
    counts = {k: np.zeros(3) for k in files}                                    # tp, fp, fn
    for i in range(len(ds)):
        img = np.rint(ds.image(i).transpose(1, 2, 0) * 255).astype(np.uint8)  # back to the tile the server receives
        padded, ratio, pad = letterbox(img, size)
        x = np.ascontiguousarray(padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        y = ds.mask(i)[0] > 0.5
        for k, sess in sessions.items():
            p = postprocess_semantic_masks(sess.run(["probs"], {"images": x}), img.shape[:2], ratio, pad, size,
                                           threshold) > 0
            counts[k] += [(p & y).sum(), (p & ~y).sum(), (~p & y).sum()]
    return {f"{k}_iou": float(tp / (tp + fp + fn + 1e-8)) for k, (tp, fp, fn) in counts.items()}


def export_untrained(class_name: str, architecture: str, encoder: str, out: Path, size: int = 1280) -> dict:
    """Random-weight model of the configured architecture for `class_name` (cost is the same as a trained one)."""
    cfg = OmegaConf.create(OmegaConf.to_container(
        compose_config([f"data={class_name}", f"architecture={architecture}", f"encoder={encoder}"]), resolve=True))
    cfg.model.encoder_weights = None
    cfg.pretrain = None
    return export_model(build_model(cfg), out, [class_name], size, architecture=f"{cfg.model.arch}/{cfg.model.encoder_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--checkpoint")
    ap.add_argument("--metadata", help="metadata.json of the training run (config + sweep results)")
    ap.add_argument("--random", action="store_true", help="untrained weights (needs no checkpoint)")
    ap.add_argument("--data", default="vegetation", help="with --random: data config (class)")
    ap.add_argument("--architecture", default="unet")
    ap.add_argument("--encoder", default="resnet34_imagenet")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    if not a.random and not (a.checkpoint and a.metadata):
        raise SystemExit("give --checkpoint and --metadata, or --random")

    cfg = OmegaConf.create(OmegaConf.to_container(load_cfg(a), resolve=True))
    cfg.model.encoder_weights = None          # weights come from the checkpoint; nothing is downloaded
    cfg.pretrain = None
    model = build_model(cfg)
    if a.checkpoint:
        state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}, strict=True)
    names = list(cfg.data.classes.keys()) if cfg.data.get("classes") else ["building"]
    side = export_model(model, Path(a.out), names, a.size, a.opset,
                        best_threshold_without_tta(a.metadata, names[0]) if len(names) == 1 else None,
                        f"{cfg.model.arch}/{cfg.model.encoder_name}")
    print("[export] sidecar:", json.dumps(side))


if __name__ == "__main__":
    main()
