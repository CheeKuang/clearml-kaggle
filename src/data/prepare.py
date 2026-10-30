"""Split the flat LoveDA folder into the train/val layout TileDataset reads.

  python -m src.data.prepare --src data/loveda --out data/loveda_multi                 # class-id masks (default)
  python -m src.data.prepare --src data/loveda --out data/loveda_split --mode binary   # one building mask (legacy)
  python -m src.data.upload --path data/loveda_multi --name loveda-multi              # then upload to ClearML

  <src>/images/*.png, <src>/masks/*.png   LoveDA ids: 0 no-data, 1 background, 2 building, 3 road, 4 water,
                                                      5 barren, 6 forest, 7 agriculture
  -> <out>/{train,val}/images/*.png        copied
     <out>/{train,val}/masks/*.png         labels: the ids unchanged (conf/data/*.yaml maps ids to classes)
                                           binary: 0 = not building, 255 = building (conf/data/buildings.yaml)

The split is random per tile (seeded): neighbouring tiles of one scene can land on both sides, so val IoU can be
optimistic. Split by scene/city if you have that information.
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
    ap.add_argument("--out", default="data/loveda_multi")
    ap.add_argument("--mode", choices=["labels", "binary"], default="labels")
    ap.add_argument("--building-class", type=int, default=2, help="mode binary only")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: remove it or choose another --out")
    names = sorted(p.name for p in (src / "images").glob("*.png"))
    if missing := [n for n in names if not (src / "masks" / n).exists()]:
        raise SystemExit(f"{len(missing)} images have no mask, e.g. {missing[:3]}")
    val = set(np.random.default_rng(a.seed).choice(names, int(round(len(names) * a.val_frac)), replace=False).tolist())

    for split in ("train", "val"):
        for sub in ("images", "masks"):
            (out / split / sub).mkdir(parents=True)
    ratios = {"train": [], "val": []}           # per tile: pixel share per id 0..7 (labels) or of building (binary)
    for n in names:
        split = "val" if n in val else "train"
        shutil.copy2(src / "images" / n, out / split / "images" / n)
        raw = np.array(Image.open(src / "masks" / n))
        if a.mode == "labels":
            shutil.copy2(src / "masks" / n, out / split / "masks" / n)
            ratios[split].append(np.bincount(raw.ravel(), minlength=8)[:8] / raw.size)
        else:
            building = raw == a.building_class
            Image.fromarray(building.astype(np.uint8) * 255).save(out / split / "masks" / n)
            ratios[split].append([building.mean()])

    for split, r in ratios.items():
        r = np.asarray(r)
        if a.mode == "labels":
            print(f"{split}: {len(r)} tiles; mean pixel share per id 0..7: " + " ".join(f"{x:.3f}" for x in r.mean(0)))
        else:
            print(f"{split}: {len(r)} tiles, mean building ratio {r.mean():.3f}, without buildings {int((r == 0).sum())}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
