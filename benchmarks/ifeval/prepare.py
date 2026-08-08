#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid


DATASET = "google/IFEval"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare IFEval requests and cases.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def request_for(prompt):
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def case_for(row):
    if not isinstance(row, dict):
        raise ValueError("dataset rows must be objects")
    key = row.get("key")
    prompt = row.get("prompt")
    instruction_ids = row.get("instruction_id_list")
    instruction_kwargs = row.get("kwargs")
    if isinstance(key, bool) or not isinstance(key, int):
        raise ValueError("provider key must be an integer")
    if not isinstance(prompt, str):
        raise ValueError(f"provider prompt for key {key} must be a string")
    if (
        not isinstance(instruction_ids, list)
        or not instruction_ids
        or any(not isinstance(item, str) or not item for item in instruction_ids)
    ):
        raise ValueError(f"instruction IDs for key {key} must be nonempty strings")
    if (
        not isinstance(instruction_kwargs, list)
        or len(instruction_kwargs) != len(instruction_ids)
        or any(not isinstance(item, dict) for item in instruction_kwargs)
    ):
        raise ValueError(f"instruction arguments for key {key} must align with IDs")

    prepared_id = str(key)
    if not SAFE_ID.fullmatch(prepared_id):
        raise ValueError(f"provider key {key} does not produce a safe ID")
    return {
        "id": prepared_id,
        "key": key,
        "prompt": prompt,
        "instruction_id_list": instruction_ids,
        "kwargs": instruction_kwargs,
    }


def reject_json_constant(value):
    raise ValueError(f"non-JSON constant {value}")


def read_jsonl(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"{path.name} must contain nonblank records")
    return [json.loads(line, parse_constant=reject_json_constant) for line in lines]


def validate_prepared(directory):
    requests = read_jsonl(directory / "requests.jsonl")
    cases = read_jsonl(directory / "cases.jsonl")
    ids = (directory / "ids.txt").read_text(encoding="utf-8").splitlines()
    if not ids or any(not SAFE_ID.fullmatch(item) for item in ids):
        raise ValueError("prepared IDs must be nonblank safe IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("prepared IDs contain a duplicate")
    if len(requests) != len(ids) or len(cases) != len(ids):
        raise ValueError("request, ID, and grading case counts do not match")
    if any(not isinstance(request, dict) for request in requests):
        raise ValueError("requests must be JSON objects")
    case_ids = [case.get("id") if isinstance(case, dict) else None for case in cases]
    if case_ids != ids:
        raise ValueError("grading cases are not aligned with prepared IDs")


def publish_prepared(version, output):
    previous_version = None
    if output.exists() or output.is_symlink():
        if not output.is_symlink():
            raise ValueError("existing prepared output was not created by this command")
        previous_target = Path(os.readlink(output))
        if not previous_target.is_absolute():
            previous_target = output.parent / previous_target
        previous_version = previous_target.resolve()

    pending_link = output.with_name(f".{output.name}.link-{uuid.uuid4().hex}")
    relative_target = os.path.relpath(version, output.parent)
    os.symlink(relative_target, pending_link, target_is_directory=True)
    try:
        os.replace(pending_link, output)
    finally:
        if pending_link.is_symlink():
            pending_link.unlink()

    versions = version.parent.resolve()
    if (
        previous_version is not None
        and previous_version.parent == versions
        and previous_version.name.startswith("version-")
    ):
        shutil.rmtree(previous_version, ignore_errors=True)


def prepare(cache_dir, output):
    from datasets import load_dataset

    rows = load_dataset(
        DATASET,
        cache_dir=str(cache_dir),
        download_mode="reuse_dataset_if_exists",
    )["train"]
    cases = [case_for(dict(row)) for row in rows]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("provider keys produce duplicate prepared IDs")
    requests = [request_for(case["prompt"]) for case in cases]

    output.parent.mkdir(parents=True, exist_ok=True)
    versions = output.parent / f".{output.name}.versions"
    versions.mkdir(exist_ok=True)
    version = Path(tempfile.mkdtemp(prefix="version-", dir=versions))
    published = False
    try:
        write_jsonl(version / "requests.jsonl", requests)
        (version / "ids.txt").write_text(
            "".join(f"{item}\n" for item in ids), encoding="utf-8"
        )
        write_jsonl(version / "cases.jsonl", cases)
        validate_prepared(version)
        publish_prepared(version, output)
        published = True
    finally:
        if not published and version.exists():
            shutil.rmtree(version)


def main():
    args = parse_args()
    try:
        prepare(args.cache_dir, args.output)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
