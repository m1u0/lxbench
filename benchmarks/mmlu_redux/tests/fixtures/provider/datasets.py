import json
import os
from pathlib import Path


ROWS = {
    "abstract_algebra": [
        {
            "question": "Which option is the identity?",
            "choices": ["zero", "one", "two", "three"],
            "answer": 1,
            "error_type": "ok",
        },
        {
            "question": "This row must be filtered out.",
            "choices": ["A", "B", "C", "D"],
            "answer": 0,
            "error_type": "wrong_groundtruth",
        },
    ],
    "anatomy": [
        {
            "question": "This row must also be filtered out.",
            "choices": ["A", "B", "C", "D"],
            "answer": 0,
            "error_type": "expert",
        },
        {
            "question": "Which organ pumps blood?",
            "choices": ["Lung", "Heart", "Kidney", "Liver"],
            "answer": 1,
            "error_type": "ok",
        },
    ],
}


def record(call):
    log_path = Path(os.environ["MMLU_FIXTURE_LOG"])
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(call) + "\n")


def get_dataset_config_names(path, **kwargs):
    record({"function": "get_dataset_config_names", "path": path, "kwargs": kwargs})
    return list(reversed(ROWS))


def load_dataset(path, name, **kwargs):
    record(
        {
            "function": "load_dataset",
            "path": path,
            "name": name,
            "kwargs": kwargs,
        }
    )
    rows = json.loads(json.dumps(ROWS[name]))
    if os.environ.get("MMLU_FIXTURE_CHANGED"):
        rows[0]["question"] += " Changed"
    return rows
