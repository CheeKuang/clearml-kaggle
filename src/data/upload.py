"""Upload a train/val folder to ClearML as a versioned Dataset and print its id (-> data.clearml_dataset_id).

  python -m src.data.upload --path data/loveda_multi --name loveda-multi
  python -m src.data.upload --path data/loveda_multi --name loveda-multi --parent <previous id>   # store the diff only

The folder needs <path>/{train,val}/{images,masks}. ClearML Dataset pattern: create -> add_files -> upload -> finalize.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clearml import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="local folder with train/ and val/")
    ap.add_argument("--project", default="building-segmentation")
    ap.add_argument("--name", default="loveda-multi")
    ap.add_argument("--parent", default=None, help="id of the previous version")
    a = ap.parse_args()

    root = Path(a.path)
    for sub in ("train/images", "train/masks", "val/images", "val/masks"):
        if not (root / sub).is_dir():
            raise SystemExit(f"{root / sub} not found: split the data first (python -m src.data.prepare)")

    dataset = Dataset.create(dataset_project=a.project, dataset_name=a.name,
                             parent_datasets=[a.parent] if a.parent else None)
    dataset.add_files(path=str(root))
    dataset.upload()
    dataset.finalize()                          # read-only from now on; fetched by id
    print(f"dataset id: {dataset.id}")


if __name__ == "__main__":
    main()
