#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid


DATASET = "edinburgh-dawg/mmlu-redux-2.0"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MMLU-Redux 2.0")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def make_request(row):
    choices = row["choices"]
    prompt = "\n".join(
        [
            row["question"],
            "",
            *(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)),
            "",
            "Respond with only the answer letter (A, B, C, or D).",
        ]
    )
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_row(row, subject, row_index):
    if not isinstance(row, dict):
        raise ValueError(f"{subject} row {row_index} is not an object")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError(f"{subject} row {row_index} has an invalid question")
    choices = row.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 4
        or any(not isinstance(choice, str) for choice in choices)
    ):
        raise ValueError(f"{subject} row {row_index} must have four string choices")
    answer = row.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, int) or answer not in range(4):
        raise ValueError(f"{subject} row {row_index} has an invalid answer")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def load_jsonl(path, record_name):
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"{record_name} contains a blank record")
    records = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{record_name} line {line_number} is not valid JSON"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"{record_name} line {line_number} is not an object")
        records.append(record)
    return records


def validate_prepared(directory):
    requests = load_jsonl(directory / "requests.jsonl", "requests file")
    ids = (directory / "ids.txt").read_text(encoding="utf-8").splitlines()
    cases = load_jsonl(directory / "cases.jsonl", "cases file")

    if any(not SAFE_ID.fullmatch(case_id) for case_id in ids):
        raise ValueError("prepared IDs must be nonblank safe IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("prepared IDs must be unique")
    case_ids = [case.get("id") for case in cases]
    if len(requests) != len(ids) or case_ids != ids:
        raise ValueError("requests, IDs, and cases are not aligned")


def publish(staging, output):
    backup = None
    if output.exists():
        if not output.is_dir():
            raise ValueError("prepared output must be a directory")
        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except OSError:
        if backup is not None:
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def prepare(cache_dir, output):
    from datasets import get_dataset_config_names, load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    ids = []
    cases = []

    subjects = sorted(get_dataset_config_names(DATASET))
    for subject in subjects:
        rows = load_dataset(
            DATASET,
            subject,
            split="test",
            cache_dir=str(cache_dir),
            download_mode="reuse_dataset_if_exists",
        )
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{subject} row {row_index} is not an object")
            if row.get("error_type") != "ok":
                continue
            validate_row(row, subject, row_index)
            case_id = f"{subject}:{row_index}"
            if not SAFE_ID.fullmatch(case_id):
                raise ValueError(f"generated unsafe ID: {case_id}")
            requests.append(make_request(row))
            ids.append(case_id)
            cases.append(
                {"id": case_id, "subject": subject, "answer": row["answer"]}
            )

    if len(ids) != len(set(ids)):
        raise ValueError("generated IDs are not unique")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        write_jsonl(staging / "requests.jsonl", requests)
        (staging / "ids.txt").write_text(
            "".join(f"{case_id}\n" for case_id in ids), encoding="utf-8"
        )
        write_jsonl(staging / "cases.jsonl", cases)
        validate_prepared(staging)
        publish(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    args = parse_args()
    try:
        prepare(args.cache_dir, args.output)
    except (ImportError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
