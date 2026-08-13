#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import sys


BENCHMARK = "mmlu-redux-2.0"
PROFILE = "qwen36-deterministic-no-thinking"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
ANSWER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-D](?![A-Za-z0-9_])", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Grade MMLU-Redux 2.0")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def reject_json_constant(value):
    raise ValueError(f"non-JSON constant {value}")


def load_jsonl(path, record_name):
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines[-1:] == [""]:
        lines.pop()
    if any(not line.strip() for line in lines):
        raise ValueError(f"{record_name} contains a blank record")
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
    cases = load_jsonl(path, "cases file")
    case_ids = []
    for line_number, case in enumerate(cases, start=1):
        case_id = case.get("id")
        subject = case.get("subject")
        answer = case.get("answer")
        if not isinstance(case_id, str) or not SAFE_ID.fullmatch(case_id):
            raise ValueError(f"case line {line_number} has an invalid ID")
        if not isinstance(subject, str) or not subject:
            raise ValueError(f"case line {line_number} has an invalid subject")
        if isinstance(answer, bool) or not isinstance(answer, int) or answer not in range(4):
            raise ValueError(f"case line {line_number} has an invalid answer")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    return cases


def load_responses(path, expected_ids):
    records = load_jsonl(path, "responses file")
    responses = {}
    for line_number, record in enumerate(records, start=1):
        response_id = record.get("id")
        if not isinstance(response_id, str):
            raise ValueError(f"response line {line_number} has an invalid ID")
        if response_id not in expected_ids:
            raise ValueError(f"response line {line_number} has an unknown ID")
        if response_id in responses:
            raise ValueError(f"response line {line_number} has a duplicate ID")
        responses[response_id] = record.get("response")
    return responses


def extract_answer(response):
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    tokens = ANSWER_TOKEN.findall(content.strip())
    if len(tokens) != 1:
        return None
    return "ABCD".index(tokens[0].upper())


def grade(cases, responses):
    correct = 0
    invalid = 0
    missing = 0
    subjects = {}

    for case in cases:
        subject = case["subject"]
        subject_score = subjects.setdefault(subject, {"correct": 0, "total": 0})
        subject_score["total"] += 1
        if case["id"] not in responses:
            missing += 1
            continue
        prediction = extract_answer(responses[case["id"]])
        if prediction is None:
            invalid += 1
            continue
        if prediction == case["answer"]:
            correct += 1
            subject_score["correct"] += 1

    total = len(cases)
    per_subject = {}
    for subject, score in subjects.items():
        subject_total = score["total"]
        subject_correct = score["correct"]
        per_subject[subject] = {
            "accuracy": subject_correct / subject_total,
            "correct": subject_correct,
            "incorrect": subject_total - subject_correct,
            "total": subject_total,
        }

    return {
        "benchmark": BENCHMARK,
        "profile": PROFILE,
        "expected": total,
        "received": len(responses),
        "correct": correct,
        "incorrect": total - correct,
        "invalid": invalid,
        "missing": missing,
        "total": total,
        "metrics": {
            "accuracy": correct / total if total else 0.0,
            "per_subject": per_subject,
        },
    }


def main():
    args = parse_args()
    try:
        cases = load_cases(args.cases)
        responses = load_responses(
            args.responses, {case["id"] for case in cases}
        )
        summary = grade(cases, responses)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        encoded_summary = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        args.output.write_text(encoded_summary, encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
