#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from upstream.provider import extract_answer, score_predictions


BENCHMARK = "longbench-v2"
PROFILE = "qwen36-deterministic-no-thinking"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def reject_json_constant(value):
    raise ValueError(f"non-JSON constant {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grade a LongBench v2 raw run.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path, record_name):
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines[-1:] == [""]:
        lines.pop()
    if any(not line.strip() for line in lines):
        raise ValueError(f"{record_name} file contains a blank record")
    records = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line, parse_constant=reject_json_constant)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{record_name} line {line_number} is not valid JSON"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"{record_name} line {line_number} is not an object")
        records.append(record)
    return records


def load_cases(path):
    cases = read_jsonl(path, "case")
    if not cases:
        raise ValueError("cases file is empty")
    ids = []
    for case in cases:
        prepared_id = case.get("id")
        if not isinstance(prepared_id, str) or not SAFE_ID.fullmatch(prepared_id):
            raise ValueError("case IDs must be nonblank safe IDs")
        if prepared_id in ids:
            raise ValueError(f"duplicate case ID: {prepared_id}")
        if case.get("answer") not in {"A", "B", "C", "D"}:
            raise ValueError(f"invalid answer for {prepared_id}")
        if case.get("difficulty") not in {"easy", "hard"}:
            raise ValueError(f"invalid difficulty for {prepared_id}")
        if case.get("length") not in {"short", "medium", "long"}:
            raise ValueError(f"invalid length for {prepared_id}")
        ids.append(prepared_id)
    return cases, ids


def load_responses(path, expected_ids):
    envelopes = read_jsonl(path, "response")
    responses = {}
    expected = set(expected_ids)
    for envelope in envelopes:
        response_id = envelope.get("id")
        if response_id not in expected:
            raise ValueError(f"unknown response ID: {response_id}")
        if response_id in responses:
            raise ValueError(f"duplicate response ID: {response_id}")
        responses[response_id] = envelope.get("response")
    return responses


def extract_content(response):
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def grade(cases, responses):
    predictions = []
    correct = 0
    invalid = 0
    missing = 0
    for case in cases:
        prepared_id = case["id"]
        if prepared_id not in responses:
            prediction = None
            missing += 1
        else:
            content = extract_content(responses[prepared_id])
            prediction = extract_answer(content) if content is not None else None
            if prediction is None:
                invalid += 1
        judge = prediction == case["answer"]
        correct += int(judge)
        predictions.append(
            {
                "pred": prediction,
                "judge": judge,
                "difficulty": case["difficulty"],
                "length": case["length"],
            }
        )

    expected = len(cases)
    return {
        "benchmark": BENCHMARK,
        "profile": PROFILE,
        "correct": correct,
        "incorrect": expected - correct,
        "invalid": invalid,
        "missing": missing,
        "total": expected,
        "expected": expected,
        "received": len(responses),
        "metrics": score_predictions(predictions),
    }


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(summary, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def print_summary(summary):
    print(f"benchmark: {summary['benchmark']}")
    print(f"profile: {summary['profile']}")
    print(f"overall_accuracy: {summary['metrics']['overall_accuracy']}")
    print(f"difficulty: {json.dumps(summary['metrics']['difficulty'])}")
    print(f"context_length: {json.dumps(summary['metrics']['context_length'])}")
    for name in (
        "correct",
        "incorrect",
        "invalid",
        "missing",
        "total",
        "expected",
        "received",
    ):
        print(f"{name}: {summary[name]}")


def run(args):
    cases, ids = load_cases(args.cases)
    responses = load_responses(args.responses, ids)
    summary = grade(cases, responses)
    write_summary(args.output, summary)
    print_summary(summary)
    return 0


def main():
    args = parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
