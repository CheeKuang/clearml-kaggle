"""Upload a local folder to ClearML as a versioned Dataset.

  python -m src.dataset --path data/loveda_split --name loveda-buildings
  python -m src.dataset --path data/loveda_split --name loveda-buildings --parent <previous dataset id>

The folder must already be in the layout src.data.TileDataset reads:
  <path>/train/images  <path>/train/masks  <path>/val/images  <path>/val/masks
The printed id goes into  data.clearml_dataset_id  (train / pipeline / agents download it with
Dataset.get(...).get_local_copy()). Follows the ClearML Dataset pattern: create -> add_files -> upload -> finalize.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clearml import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="local folder with train/ and val/")
    ap.add_argument("--project", default="building-segmentation")
    ap.add_argument("--name", default="loveda-buildings")
    ap.add_argument("--parent", default=None, help="id of the previous version (stores only the differences)")
    a = ap.parse_args()

    root = Path(a.path)
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            if not (root / split / sub).is_dir():
                raise SystemExit(f"{root / split / sub} not found: split the data into train/val first")

    # 1) create a new (draft) dataset version
    dataset = Dataset.create(dataset_project=a.project, dataset_name=a.name,
                             parent_datasets=[a.parent] if a.parent else None)

    # 2) register the files
    dataset.add_files(path=str(root))

    # 3) upload to the ClearML file server (or the configured output_uri)
    dataset.upload()

    # 4) finalize: the version becomes read-only and can be fetched by id
    dataset.finalize()
    print(f"dataset id: {dataset.id}")


if __name__ == "__main__":
    main()
