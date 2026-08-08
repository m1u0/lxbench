import json
import os
from pathlib import Path


def load_dataset(path, **kwargs):
    expected_cache = os.environ["IFEVAL_EXPECTED_CACHE"]
    if path != "google/IFEval":
        raise AssertionError(f"unexpected dataset: {path}")
    if kwargs != {
        "cache_dir": expected_cache,
        "download_mode": "reuse_dataset_if_exists",
    }:
        raise AssertionError(f"unexpected load_dataset arguments: {kwargs}")

    fixture = Path(os.environ["IFEVAL_DATASET_FIXTURE"])
    return {"train": json.loads(fixture.read_text(encoding="utf-8"))}
