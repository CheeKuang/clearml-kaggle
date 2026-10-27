"""Turn the flat LoveDA folder into the train/val layout src.data.TileDataset reads, with binary building masks.

  python -m src.prepare_data --src data/loveda --out data/loveda_split
  python -m src.dataset --path data/loveda_split          # then upload to ClearML

  <src>/images/*.png, <src>/masks/*.png   (LoveDA: 0 no-data, 1 background, 2 building, 3 road, ...)
  -> <out>/{train,val}/images/*.png        (copied)
     <out>/{train,val}/masks/*.png         (0 = not building, 255 = building; the loader uses mask > 0)

The split is random per tile (seeded). Neighbouring tiles of one scene can land on both sides, so val IoU
can be optimistic; split by scene/city instead if you have that information.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/loveda")
    ap.add_argument("--out", default="data/loveda_split")
    ap.add_argument("--building-class", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: remove it or choose another --out")
    names = sorted(p.name for p in (src / "images").glob("*.png"))
    missing = [n for n in names if not (src / "masks" / n).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images have no mask, e.g. {missing[:3]}")

    rng = np.random.default_rng(a.seed)
    val = set(rng.choice(names, int(round(len(names) * a.val_frac)), replace=False).tolist())

    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "masks").mkdir(parents=True)
    ratios = {"train": [], "val": []}
    for n in names:
        split = "val" if n in val else "train"
        shutil.copy2(src / "images" / n, out / split / "images" / n)
        building = np.array(Image.open(src / "masks" / n)) == a.building_class
        Image.fromarray(building.astype(np.uint8) * 255).save(out / split / "masks" / n)
        ratios[split].append(building.mean())

    for split, r in ratios.items():
        print(f"{split}: {len(r)} tiles, mean building ratio {np.mean(r):.3f}, "
              f"tiles without buildings {int(np.sum(np.asarray(r) == 0))}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
