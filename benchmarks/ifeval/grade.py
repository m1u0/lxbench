#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile


PROFILE = "qwen36-deterministic-no-thinking"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).resolve().parent / "upstream"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))

from sample_manifest import load_sample  # noqa: E402
from instruction_following_eval import evaluation_lib  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Grade an IFEval raw run.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reject_json_constant(value):
    raise ValueError(f"non-JSON constant {value}")


def read_jsonl(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines[-1:] == [""]:
        lines.pop()
    if any(not line.strip() for line in lines):
        raise ValueError(f"{path.name} contains a blank record")
    records = []
    for line_number, line in enumerate(lines, start=1):
        try:
            records.append(json.loads(line, parse_constant=reject_json_constant))
        except ValueError as error:
            raise ValueError(
                f"{path.name} line {line_number} is not valid JSON"
            ) from error
    return records


def load_cases(path):
    records = read_jsonl(path)
    if not records:
        raise ValueError("cases file must contain at least one case")

    cases = []
    seen_ids = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("grading cases must be JSON objects")
        prepared_id = record.get("id")
        if not isinstance(prepared_id, str) or not SAFE_ID.fullmatch(prepared_id):
            raise ValueError("grading case IDs must be nonblank safe IDs")
        if prepared_id in seen_ids:
            raise ValueError(f"duplicate grading case ID: {prepared_id}")
        seen_ids.add(prepared_id)

        key = record.get("key")
        prompt = record.get("prompt")
        instruction_ids = record.get("instruction_id_list")
        instruction_kwargs = record.get("kwargs")
        if isinstance(key, bool) or not isinstance(key, int):
            raise ValueError(f"provider key for {prepared_id} must be an integer")
        if not isinstance(prompt, str):
            raise ValueError(f"provider prompt for {prepared_id} must be a string")
        if (
            not isinstance(instruction_ids, list)
            or not instruction_ids
            or any(not isinstance(item, str) or not item for item in instruction_ids)
        ):
            raise ValueError(f"instruction IDs for {prepared_id} are invalid")
        if (
            not isinstance(instruction_kwargs, list)
            or len(instruction_kwargs) != len(instruction_ids)
            or any(not isinstance(item, dict) for item in instruction_kwargs)
        ):
            raise ValueError(f"instruction arguments for {prepared_id} are invalid")
        cases.append(
            (
                prepared_id,
                evaluation_lib.InputExample(
                    key=key,
                    prompt=prompt,
                    instruction_id_list=instruction_ids,
                    kwargs=instruction_kwargs,
                ),
            )
        )
    return cases


def extract_response_text(envelope):
    try:
        content = envelope["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def load_responses(path, expected_ids):
    responses = {}
    for envelope in read_jsonl(path):
        if not isinstance(envelope, dict):
            raise ValueError("response envelopes must be JSON objects")
        response_id = envelope.get("id")
        if not isinstance(response_id, str):
            raise ValueError("response envelope IDs must be strings")
        if response_id not in expected_ids:
            raise ValueError(f"unknown response ID: {response_id}")
        if response_id in responses:
            raise ValueError(f"duplicate response ID: {response_id}")
        responses[response_id] = extract_response_text(envelope)
    return responses


def evaluate(cases, responses, evaluator):
    outputs = []
    for prepared_id, provider_case in cases:
        response_text = responses.get(prepared_id)
        if response_text is None:
            response_text = ""
        outputs.append(evaluator(provider_case, {provider_case.prompt: response_text}))
    return outputs


def accuracies(outputs):
    prompt_correct = sum(output.follow_all_instructions for output in outputs)
    instruction_correct = sum(
        sum(output.follow_instruction_list) for output in outputs
    )
    instruction_total = sum(len(output.follow_instruction_list) for output in outputs)
    return prompt_correct / len(outputs), instruction_correct / instruction_total


def grade(cases_path, responses_path):
    cases = load_cases(cases_path)
    selected_ids, sample_metadata = load_sample(
        responses_path, [prepared_id for prepared_id, _ in cases]
    )
    selected_id_set = set(selected_ids)
    cases = [case for case in cases if case[0] in selected_id_set]
    responses = load_responses(responses_path, selected_id_set)
    strict = evaluate(cases, responses, evaluation_lib.test_instruction_following_strict)
    loose = evaluate(cases, responses, evaluation_lib.test_instruction_following_loose)
    strict_prompt, strict_instruction = accuracies(strict)
    loose_prompt, loose_instruction = accuracies(loose)
    expected = len(cases)
    received = len(responses)
    invalid = sum(response is None for response in responses.values())
    summary = {
        "benchmark": "ifeval",
        "profile": PROFILE,
        "expected": expected,
        "received": received,
        "missing": expected - received,
        "invalid": invalid,
        "metrics": {
            "strict_prompt_level_accuracy": strict_prompt,
            "strict_instruction_level_accuracy": strict_instruction,
            "loose_prompt_level_accuracy": loose_prompt,
            "loose_instruction_level_accuracy": loose_instruction,
        },
    }
    if sample_metadata is not None:
        summary.update(sample_metadata)
    return summary


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(summary, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def print_summary(summary):
    for field in ("benchmark", "profile", "expected", "received", "missing", "invalid"):
        print(f"{field}: {summary[field]}")
    if summary.get("sampled"):
        print(
            f"sample: {summary['expected']} of {summary['population_total']} cases "
            f"(seed {summary['sample_seed']})"
        )
    for name, accuracy in summary["metrics"].items():
        print(f"{name}: {accuracy:.6f}")


def main():
    args = parse_args()
    try:
        summary = grade(args.cases, args.responses)
        write_summary(args.output, summary)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
