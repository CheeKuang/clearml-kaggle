"""Turn the OpenEarthMap folder into the train/val layout TileDataset reads, at the ground scale production sends.

  python -m src.data.prepare_oem --src data/OpenEarthMap_wo_xBD --out data/oem_416
  python -m src.data.upload --path data/oem_416 --name oem-416                      # then upload to ClearML

  <src>/<region>/images/<region>_<n>.tif   3-band uint8, 0.25-0.5 m per pixel, sizes 406..1024, georeferenced
  <src>/<region>/labels/<region>_<n>.tif   OEM ids: 0 unlabeled, 1 bareland, 2 rangeland, 3 developed space, 4 road,
                                                    5 tree, 6 water, 7 agriculture, 8 building
  -> <out>/{train,val}/images/<tile>_<y>_<x>.png   resampled to --gsd metres per pixel, cut into --crop px squares
     <out>/{train,val}/masks/<tile>_<y>_<x>.png    the ids unchanged (conf/data/oem_*.yaml maps ids to classes)
     <out>/split.json                              how it was made, logged by every train Task

Why resample: the Chisel web app exports every crop at ~0.55-0.6 m per pixel whatever zoom the user is on
(libs/map/print-map.ts), and the server cuts anything over 1280 px into 1280 px tiles at that native scale.
Images are shrunk with INTER_AREA (the browser's map render is already smooth), masks with nearest neighbour.

Why split by region: every OEM region sits in both the official train and val lists, so neighbouring tiles of one
city land on both sides and val IoU is optimistic. The default holds out whole regions (--split official keeps the
official lists). Tiles without a label (the unlabeled OEM test set, the xBD tiles) are skipped.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

N_IDS = 9                                   # OEM ids 0..8
METRES_PER_DEGREE = 111_320.0               # north-south: independent of latitude


def gsd_from_transform(a: float, e: float, geographic: bool) -> float:
    """Metres per pixel from an affine transform. Geographic CRS (degrees): use the north-south size, which does not
    depend on latitude (the east-west size shrinks with cos(lat))."""
    return abs(e) * METRES_PER_DEGREE if geographic else abs(a)


def crop_starts(n: int, size: int, sliver: float = 0.1) -> list[int]:
    """Start offsets of `size`-px crops that cover [0, n): as few as cover it, evenly spaced (neighbours overlap when
    n is not a multiple of size). A leftover under `sliver` of a crop is dropped, not given a crop of its own: that
    crop would repeat its neighbour almost pixel for pixel. [] when n < size."""
    if n < size:
        return []
    k = n // size if n % size <= sliver * size else -(-n // size)
    return [0] if k <= 1 else np.linspace(0, n - size, k).round().astype(int).tolist()


def resample(img: np.ndarray, factor: float, nearest: bool) -> np.ndarray:
    """(H, W[, C]) -> (round(H*factor), round(W*factor)). Masks must use nearest so ids stay ids."""
    h, w = img.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    if size == (w, h):
        return img
    method = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    return cv2.resize(img, size, interpolation=method)


def region_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def choose_val_regions(regions: list[str], frac: float, seed: int) -> set[str]:
    regions = sorted(regions)
    k = max(1, int(round(len(regions) * frac)))
    return set(np.random.default_rng(seed).choice(regions, k, replace=False).tolist())


def read_tile(img_path: Path, lab_path: Path):
    """(HWC uint8 image, HW uint8 labels, metres per pixel)."""
    import rasterio

    with rasterio.open(img_path) as s:
        img, t = s.read()[:3].transpose(1, 2, 0), s.transform
        gsd = gsd_from_transform(t.a, t.e, s.crs.is_geographic if s.crs else False)
    with rasterio.open(lab_path) as s:
        lab = s.read(1)
    if img.shape[:2] != lab.shape:
        raise ValueError(f"{img_path.name}: image {img.shape[:2]} and label {lab.shape} differ in size")
    return np.ascontiguousarray(img), lab, gsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/OpenEarthMap_wo_xBD")
    ap.add_argument("--out", default="data/oem_416")
    ap.add_argument("--gsd", type=float, default=0.6, help="target metres per pixel (production: 0.55-0.6)")
    ap.add_argument("--crop", type=int, default=416, help="crop size in pixels after resampling (416 keeps 91% of the tiles, 512 only 75%)")
    ap.add_argument("--split", choices=["region", "official"], default="region")
    ap.add_argument("--val-frac", type=float, default=0.15, help="share of regions held out (--split region)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="only the first N usable tiles (a quick dry run)")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: remove it or choose another --out")

    pairs = sorted((p, p.parent.parent / "labels" / p.name) for p in src.glob("*/images/*.tif"))
    usable = [(p, q) for p, q in pairs if q.exists()]
    print(f"{len(pairs)} images, {len(usable)} with a label ({len(pairs) - len(usable)} skipped: no label)")

    if a.split == "official":
        listed = {s: {x.strip() for x in (src / f"{s}.txt").read_text().splitlines() if x.strip()} for s in ("train", "val")}
        side = {p.name: s for p, _ in usable for s in listed if p.name in listed[s]}
        usable = [(p, q) for p, q in usable if p.name in side]
        val_regions = None
    else:
        val_regions = choose_val_regions(sorted({region_of(p.stem) for p, _ in usable}), a.val_frac, a.seed)
        side = {p.name: "val" if region_of(p.stem) in val_regions else "train" for p, _ in usable}
    if a.limit:
        usable = usable[:: max(1, len(usable) // a.limit)][: a.limit]

    for split in ("train", "val"):
        for sub in ("images", "masks"):
            (out / split / sub).mkdir(parents=True)
    counts = {s: np.zeros(N_IDS, np.int64) for s in ("train", "val")}
    crops, tiles, gsds = Counter(), Counter(), Counter()
    skipped_small, bad_ids = 0, 0

    for i, (ip, lp) in enumerate(usable):
        img, lab, gsd = read_tile(ip, lp)
        if not 0.05 < gsd < 2.0:
            raise SystemExit(f"{ip.name}: pixel size {gsd:.4f} m is not plausible: check the georeference")
        gsds[round(gsd, 2)] += 1
        factor = gsd / a.gsd                               # 0.3 m source at 0.6 m target -> 0.5
        img, lab = resample(img, factor, nearest=False), resample(lab, factor, nearest=True)
        bad_ids += int((lab >= N_IDS).any())
        ys, xs = crop_starts(lab.shape[0], a.crop), crop_starts(lab.shape[1], a.crop)
        if not ys or not xs:
            skipped_small += 1                             # smaller than one crop after resampling
            continue
        split = side[ip.name]
        tiles[split] += 1
        for y in ys:
            for x in xs:
                name = f"{ip.stem}_{y}_{x}.png"
                Image.fromarray(img[y:y + a.crop, x:x + a.crop]).save(out / split / "images" / name)
                m = lab[y:y + a.crop, x:x + a.crop]
                Image.fromarray(m).save(out / split / "masks" / name)
                counts[split] += np.bincount(m.ravel(), minlength=N_IDS)[:N_IDS]
                crops[split] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(usable)} tiles", flush=True)

    for split, c in counts.items():
        share = c / max(1, c.sum())
        print(f"{split}: {tiles[split]} tiles -> {crops[split]} crops; pixel share per id 0..8: "
              + " ".join(f"{x:.3f}" for x in share))
    if skipped_small or bad_ids:
        print(f"skipped {skipped_small} tiles smaller than one crop after resampling; {bad_ids} tiles with ids >= {N_IDS}")

    (out / "split.json").write_text(json.dumps({
        "method": "whole regions held out" if a.split == "region" else "official OEM train.txt / val.txt (by tile)",
        "src": str(src), "mode": "labels", "seed": a.seed, "target_gsd_m": a.gsd, "crop_px": a.crop,
        "resample": {"image": "INTER_AREA (down) / INTER_LINEAR (up)", "mask": "INTER_NEAREST"},
        "source_gsd_m": {str(k): v for k, v in sorted(gsds.items())},
        **({"val_frac": a.val_frac, "val_regions": sorted(val_regions)} if val_regions else {}),
        "oem_ids": {"0": "unlabeled", "1": "bareland", "2": "rangeland", "3": "developed space", "4": "road",
                    "5": "tree", "6": "water", "7": "agriculture", "8": "building"},
        "tiles": dict(tiles), "crops": dict(crops), "skipped_small": skipped_small,
        "pixel_share": {s: [round(float(x), 5) for x in c / max(1, c.sum())] for s, c in counts.items()}}, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
