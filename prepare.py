#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare every lxbench dataset.")
    parser.add_argument(
        "--longbench-context-size",
        type=int,
        required=True,
        help="Effective context size configured for the target server.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def preparer_commands(longbench_context_size, force):
    result = [
        [
            sys.executable,
            "benchmarks/ifeval/prepare.py",
            "--cache-dir",
            "data/ifeval",
            "--output",
            "prepared/ifeval",
        ],
        [
            sys.executable,
            "benchmarks/longbench_v2/prepare.py",
            "--context-size",
            str(longbench_context_size),
            "--cache",
            "data/longbench-v2",
            "--output",
            "prepared/longbench-v2",
        ],
        [
            sys.executable,
            "benchmarks/mmlu_redux/prepare.py",
            "--cache-dir",
            "data/mmlu-redux-2.0",
            "--output",
            "prepared/mmlu-redux-2.0",
        ],
    ]
    if force:
        for command in result:
            command.append("--force")
    return result


def main(argv=None):
    args = parse_args(argv)
    for command in preparer_commands(args.longbench_context_size, args.force):
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
