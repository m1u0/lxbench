#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


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
).rstrip("\n")


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
    requests = []
    cases = []
    for row in dataset:
        if not isinstance(row, dict):
            raise ValueError("dataset rows must be objects")
        prepared_id = require_string(row, "_id")
        if not SAFE_ID.fullmatch(prepared_id):
            raise ValueError(f"unsafe LongBench ID: {prepared_id}")
        if prepared_id in ids:
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


def validate_staged(directory):
    request_lines = (directory / "requests.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    ids = (directory / "ids.txt").read_text(encoding="utf-8").splitlines()
    case_lines = (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()
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
    for request, case in zip(requests, cases):
        if (
            request.get("max_tokens") != GENERATION_TOKENS
            or request.get("temperature") != 0
            or request.get("stream") is not False
            or request.get("chat_template_kwargs") != {"enable_thinking": False}
            or "model" in request
            or "gguf" in request
        ):
            raise ValueError("prepared request profile is invalid")
        messages = request.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or not isinstance(messages[0], dict)
            or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str)
        ):
            raise ValueError("prepared request messages are invalid")
        if (
            case.get("answer") not in {"A", "B", "C", "D"}
            or not isinstance(case.get("category"), str)
            or case.get("difficulty") not in {"easy", "hard"}
            or case.get("length") not in {"short", "medium", "long"}
            or not isinstance(case.get("truncated"), bool)
        ):
            raise ValueError("prepared grading case is invalid")


def publish(output_directory, requests, ids, cases):
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_directory.name}-", dir=output_directory.parent
    ) as temporary:
        staged = Path(temporary)
        write_jsonl(staged / "requests.jsonl", requests)
        (staged / "ids.txt").write_text("".join(f"{item}\n" for item in ids), encoding="utf-8")
        write_jsonl(staged / "cases.jsonl", cases)
        validate_staged(staged)

        output_directory.mkdir(parents=True, exist_ok=True)
        for filename in ("requests.jsonl", "ids.txt", "cases.jsonl"):
            os.replace(staged / filename, output_directory / filename)


def run(args):
    tokenizer_directory = snapshot_download(
        TOKENIZER_ID,
        local_dir=str(args.cache / "tokenizer"),
        allow_patterns=TOKENIZER_FILES,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory,
        local_files_only=True,
    )
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        cache_dir=str(args.cache / "dataset"),
        download_mode="reuse_dataset_if_exists",
    )
    requests, ids, cases = prepare_records(dataset, tokenizer, args.context_size)
    publish(args.output, requests, ids, cases)
    print(f"prepared {len(ids)} LongBench v2 requests in {args.output}")
    return 0


def main():
    args = parse_args()
    try:
        return run(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
