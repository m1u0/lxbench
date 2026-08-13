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

DATASET_ID = "zai-org/LongBench-v2"
TOKENIZER_ID = "Qwen/Qwen3.6-35B-A3B"
GENERATION_TOKENS = 128
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
TOKENIZER_FILES = [
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
BENCHMARK_DIRECTORY = Path(__file__).resolve().parent
PROMPT_TEMPLATE = (BENCHMARK_DIRECTORY / "upstream" / "0shot.txt").read_text(
    encoding="utf-8"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare LongBench v2 zero-shot direct requests."
    )
    parser.add_argument(
        "--context-size",
        type=int,
        required=True,
        help="Effective context size configured for the target server.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/longbench-v2"),
        help="Working cache for the dataset and tokenizer assets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prepared/longbench-v2"),
        help="Prepared benchmark directory.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def render_prompt(row):
    replacements = {
        "$DOC$": row["context"].strip(),
        "$Q$": row["question"].strip(),
        "$C_A$": row["choice_A"].strip(),
        "$C_B$": row["choice_B"].strip(),
        "$C_C$": row["choice_C"].strip(),
        "$C_D$": row["choice_D"].strip(),
    }
    prompt = PROMPT_TEMPLATE
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def chat_tokens(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def middle_truncate(tokenizer, prompt, input_budget):
    if len(chat_tokens(tokenizer, prompt)) <= input_budget:
        return prompt, False

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    def candidate(token_count):
        beginning_count = (token_count + 1) // 2
        ending_count = token_count // 2
        token_ids = prompt_tokens[:beginning_count]
        if ending_count:
            token_ids += prompt_tokens[-ending_count:]
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    if len(prompt_tokens) < 2 or len(chat_tokens(tokenizer, candidate(2))) > input_budget:
        raise ValueError("context size cannot fit chat overhead and both prompt ends")

    low = 2
    high = min(len(prompt_tokens) - 1, input_budget)
    best = candidate(2)
    while low <= high:
        middle = (low + high) // 2
        truncated = candidate(middle)
        if len(chat_tokens(tokenizer, truncated)) <= input_budget:
            best = truncated
            low = middle + 1
        else:
            high = middle - 1

    return best, True


def require_string(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset row has invalid {field}")
    return value


def prepare_records(dataset, tokenizer, context_size):
    if context_size <= GENERATION_TOKENS:
        raise ValueError("context size must exceed the 128-token generation reserve")
    input_budget = context_size - GENERATION_TOKENS
    if len(chat_tokens(tokenizer, "")) > input_budget:
        raise ValueError("context size cannot fit the empty chat template")

    ids = []
    seen_ids = set()
    requests = []
    cases = []
    for row in dataset:
        if not isinstance(row, dict):
            raise ValueError("dataset rows must be objects")
        prepared_id = require_string(row, "_id")
        if not SAFE_ID.fullmatch(prepared_id):
            raise ValueError(f"unsafe LongBench ID: {prepared_id}")
        if prepared_id in seen_ids:
            raise ValueError(f"duplicate LongBench ID: {prepared_id}")
        for field in (
            "context",
            "question",
            "choice_A",
            "choice_B",
            "choice_C",
            "choice_D",
        ):
            require_string(row, field)
        answer = require_string(row, "answer")
        category = require_string(row, "domain")
        difficulty = require_string(row, "difficulty")
        length = require_string(row, "length")
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"invalid answer for {prepared_id}")
        if difficulty not in {"easy", "hard"}:
            raise ValueError(f"invalid difficulty for {prepared_id}")
        if length not in {"short", "medium", "long"}:
            raise ValueError(f"invalid length for {prepared_id}")

        prompt, truncated = middle_truncate(
            tokenizer, render_prompt(row), input_budget
        )
        ids.append(prepared_id)
        seen_ids.add(prepared_id)
        requests.append(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": GENERATION_TOKENS,
                "temperature": 0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        cases.append(
            {
                "id": prepared_id,
                "answer": answer,
                "category": category,
                "difficulty": difficulty,
                "length": length,
                "truncated": truncated,
            }
        )
    return requests, ids, cases


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def replace_symlink(target, output, purpose):
    pending_link = output.with_name(f".{output.name}.{purpose}-{uuid.uuid4().hex}")
    os.symlink(target, pending_link, target_is_directory=True)
    try:
        os.replace(pending_link, output)
    finally:
        if pending_link.is_symlink():
            pending_link.unlink()


def output_points_to(output, version):
    return output.is_symlink() and output.resolve() == version.resolve()


def validate_prepared(directory):
    request_lines = (directory / "requests.jsonl").read_text(
        encoding="utf-8"
    ).split("\n")
    ids = (directory / "ids.txt").read_text(encoding="utf-8").splitlines()
    case_lines = (directory / "cases.jsonl").read_text(encoding="utf-8").split("\n")
    if request_lines[-1:] == [""]:
        request_lines.pop()
    if case_lines[-1:] == [""]:
        case_lines.pop()
    if (
        not request_lines
        or len(request_lines) != len(ids)
        or len(ids) != len(case_lines)
        or any(not line.strip() for line in request_lines + ids + case_lines)
    ):
        raise ValueError("prepared request, ID, and case counts do not align")
    requests = [json.loads(line) for line in request_lines]
    cases = [json.loads(line) for line in case_lines]
    if any(not isinstance(request, dict) for request in requests):
        raise ValueError("prepared requests must be JSON objects")
    if len(ids) != len(set(ids)) or any(not SAFE_ID.fullmatch(item) for item in ids):
        raise ValueError("prepared IDs must be unique safe IDs")
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("prepared cases must be JSON objects")
    if [case.get("id") for case in cases] != ids:
        raise ValueError("prepared case IDs do not align")


def already_prepared(output):
    if not output.is_symlink():
        return False
    try:
        validate_prepared(output)
    except (OSError, ValueError):
        return False
    return True


def publish(output_directory, requests, ids, cases):
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}-version-", dir=output_directory.parent
        )
    )
    previous_link_target = None
    previous_generation = None
    try:
        write_jsonl(staged / "requests.jsonl", requests)
        (staged / "ids.txt").write_text(
            "".join(f"{item}\n" for item in ids), encoding="utf-8"
        )
        write_jsonl(staged / "cases.jsonl", cases)
        validate_prepared(staged)

        if os.path.lexists(output_directory) and not output_directory.is_symlink():
            raise ValueError(
                "prepared output path must be absent or a prior version symlink"
            )
        if output_directory.is_symlink():
            previous_link_target = os.readlink(output_directory)
            previous_generation = Path(previous_link_target)
            if not previous_generation.is_absolute():
                previous_generation = output_directory.parent / previous_generation
            previous_generation = previous_generation.resolve()
        try:
            replace_symlink(staged.name, output_directory, "link")
        except BaseException:
            if output_points_to(output_directory, staged):
                if previous_link_target is None:
                    output_directory.unlink()
                else:
                    replace_symlink(
                        previous_link_target, output_directory, "rollback"
                    )
            raise
        if (
            previous_generation is not None
            and previous_generation.parent == output_directory.parent.resolve()
            and previous_generation.name.startswith(
                f".{output_directory.name}-version-"
            )
        ):
            shutil.rmtree(previous_generation, ignore_errors=True)
    finally:
        if not output_points_to(output_directory, staged):
            shutil.rmtree(staged)


def prepare(cache, output, context_size):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer_directory = snapshot_download(
        TOKENIZER_ID,
        local_dir=str(cache / "tokenizer"),
        allow_patterns=TOKENIZER_FILES,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory,
        local_files_only=True,
    )
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        cache_dir=str(cache / "dataset"),
        download_mode="reuse_dataset_if_exists",
    )
    requests, ids, cases = prepare_records(dataset, tokenizer, context_size)
    publish(output, requests, ids, cases)
    print(f"prepared {len(ids)} LongBench v2 requests in {output}")


def main():
    args = parse_args()
    try:
        if not args.force and already_prepared(args.output):
            print(f"prepared output already exists in {args.output}; skipping")
            return 0
        prepare(args.cache, args.output, args.context_size)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
