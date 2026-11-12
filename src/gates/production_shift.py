"""Does PRODUCTION look like the training data? Domain-shift check of a training split against a folder of production images.

  python -m src.gates.production_shift --prod-dir <folder of production images> --root data/oem_416 [--split train]
                                       [--tile 416] [--n 300] [--per-image 8] [--feature stats|encoder|both] [--out result.json]

The gate in preflight.py compares a dataset's train and val splits, so it can only say the dataset is consistent with
itself. This command uses the same test (`knn_interleaving`, see docs/GATES_EXPLAINED.md) with the production images as
the candidate side. Production images need no labels. It runs on your own machine: nothing is uploaded.

  * The production images may be any size, PNG / JPEG / TIFF. Transparent pixels (alpha 0, the area outside a drawn
    polygon) are never used: only windows fully inside the opaque area are cut, `--tile` px square (match the training crops).
  * The baseline is the training split against the dataset's own val split, so the production score has something to be
    read against: a 0.7 means little unless the baseline is 0.3.
  * Windows cut from one image look alike. `--per-image` caps them, and the report says how many independent images were
    used: under about 100 images the production score is optimistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .preflight import encoder_features, knn_interleaving, stats_features

Image.MAX_IMAGE_PIXELS = None
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class ImageSet:
    """The minimum the feature functions of preflight.py need from a dataset: len(), .image(i) as float CHW 0..1, .c.in_channels."""

    class c:
        in_channels = 3

    def __init__(self, images: list[np.ndarray]):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def image(self, i: int) -> np.ndarray:
        return self.images[i]


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


def opaque_windows(rgba: np.ndarray, tile: int, per_image: int, rng: np.random.Generator, min_mean: float = 20.0) -> list[np.ndarray]:
    """Up to `per_image` windows of tile x tile px, as float CHW 0..1. A window is used only if every pixel is opaque
    (alpha 255) and it is not black (mean above `min_mean` of 255)."""
    h, w = rgba.shape[:2]
    origins = [(y, x) for y in range(0, h - tile + 1, tile) for x in range(0, w - tile + 1, tile)]
    rng.shuffle(origins)
    out = []
    for y, x in origins:
        win = rgba[y:y + tile, x:x + tile]
        if (win[..., 3] == 255).all() and win[..., :3].mean() > min_mean:
            out.append(win[..., :3].astype(np.float32).transpose(2, 0, 1) / 255.0)
            if len(out) == per_image:
                break
    return out


def production_set(folder: Path, tile: int, n: int, per_image: int, seed: int) -> tuple[ImageSet, int, int]:
    """(windows, number of images that gave at least one window, number of images skipped)."""
    rng, wins, used, skipped = np.random.default_rng(seed), [], 0, 0
    files = list_images(folder)
    if not files:
        raise SystemExit(f"no PNG / JPEG / TIFF images in {folder}")
    for f in files:
        got = opaque_windows(np.array(Image.open(f).convert("RGBA")), tile, per_image, rng)
        wins += got
        used, skipped = used + bool(got), skipped + (not got)
    if not wins:
        raise SystemExit(f"no fully opaque {tile} px window in any of the {len(files)} images in {folder}")
    keep = rng.permutation(len(wins))[:n]
    return ImageSet([wins[i] for i in keep]), used, skipped


def split_set(root: Path, split: str, n: int, seed: int, divide_by: float) -> ImageSet:
    """n random tiles of <root>/<split>/images/, scaled like TileDataset.image (first 3 bands / divide_by)."""
    files = list_images(root / split / "images")
    if not files:
        raise SystemExit(f"no images under {root / split / 'images'}")
    pick = np.random.default_rng(seed).permutation(len(files))[:n]
    return ImageSet([np.array(Image.open(files[i]).convert("RGB")).astype(np.float32).transpose(2, 0, 1) / divide_by for i in pick])


def features(kind: str, ds, cfg):
    return stats_features(ds) if kind == "stats" else encoder_features(ds, cfg)


def run_check(prod_dir, root="data/oem_416", split="train", baseline_split="val", tile=416, n=300, per_image=8, k=10,
              feature="stats", data="oem_road", divide_by=255.0, seed=0, warn_above=0.5, out=None) -> dict:
    """Print the report and return it: {"production_windows", "production_images", "stats"/"encoder": {"<split> (baseline)": scores,
    "production": scores}, "max_production_shift": the highest production score over the feature kinds used}."""
    root = Path(root)
    prod, used, skipped = production_set(Path(prod_dir), tile, n, per_image, seed)
    ref = split_set(root, split, n, seed, divide_by)
    base = split_set(root, baseline_split, n, seed, divide_by) if baseline_split else None
    print(f"reference : {len(ref)} tiles of {root}/{split}\n"
          f"production: {len(prod)} windows of {tile} px from {used} images ({skipped} skipped: no fully opaque window)")
    if used < 100:
        print(f"WARNING   : only {used} independent images. Windows from one image look alike, so the production score is optimistic;"
              " use 100 or more images for a firm number.")

    cfg = None
    if feature != "stats":
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "conf"), version_base=None):
            cfg = compose("config", overrides=[f"data={data}", "clearml.enabled=false"])
    result = {"reference": f"{root}/{split}", "tile": tile, "k": k, "production_windows": len(prod), "production_images": used}
    for kind in (["stats", "encoder"] if feature == "both" else [feature]):
        f_ref, f_prod = features(kind, ref, cfg), features(kind, prod, cfg)
        rows = {"production": knn_interleaving(f_ref, f_prod, k)}
        if base is not None:
            rows = {f"{baseline_split} (baseline)": knn_interleaving(f_ref, features(kind, base, cfg), k), **rows}
        print(f"\nfeatures: {kind}\n  {'':22}{'shift':>7}{'reference side':>17}{'other side':>13}")
        for name, sc in rows.items():
            print(f"  {split + ' vs ' + name:<22}{sc['shift_score']:>7.2f}{sc['shift_score_ref']:>17.2f}{sc['shift_score_cand']:>13.2f}")
        result[kind] = {name: {key: sc[key] for key in ("shift_score", "shift_score_ref", "shift_score_cand")} for name, sc in rows.items()}
        score = rows["production"]["shift_score"]
        print("  -> " + (f"production looks different from the training data (score above {warn_above})" if score > warn_above
                         else f"production looks like the training data (score at or under {warn_above})"))
    result["max_production_shift"] = max(result[kind]["production"]["shift_score"] for kind in ("stats", "encoder") if kind in result)
    print("\n0 = mixed as much as chance, 1 = fully separated. Read the production score against the baseline, not on its own.")
    if out:
        Path(out).write_text(json.dumps(result, indent=2))
        print(f"written {out}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prod-dir", required=True, help="folder of production images (no masks needed)")
    ap.add_argument("--root", default="data/oem_416", help="prepared dataset: <root>/<split>/images/")
    ap.add_argument("--split", default="train", help="the reference split (what the model learns from)")
    ap.add_argument("--baseline-split", default="val", help="its own split used as the baseline (use '' to skip)")
    ap.add_argument("--tile", type=int, default=416, help="window size in pixels; the same as the training crops")
    ap.add_argument("--n", type=int, default=300, help="tiles per side")
    ap.add_argument("--per-image", type=int, default=8, help="at most this many windows from one production image")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--feature", choices=["stats", "encoder", "both"], default="stats")
    ap.add_argument("--data", default="oem_road", help="data config used to build the encoder (only for --feature encoder/both)")
    ap.add_argument("--divide-by", type=float, default=255.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warn-above", type=float, default=0.5, help="flag a production score above this")
    ap.add_argument("--out", default=None, help="write the result json here")
    a = ap.parse_args()
    run_check(a.prod_dir, a.root, a.split, a.baseline_split, a.tile, a.n, a.per_image, a.k, a.feature, a.data, a.divide_by, a.seed,
              a.warn_above, a.out)


if __name__ == "__main__":
    main()
