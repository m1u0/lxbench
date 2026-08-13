import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_DEPENDENCIES = Path(__file__).resolve().parent / "fake_dependencies"


class IFEvalCommandTests(unittest.TestCase):
    def run_prepare(
        self,
        cache,
        output,
        fixture="input_data.json",
        extra_environment=None,
        force=False,
    ):
        environment = os.environ.copy()
        environment["IFEVAL_DATASET_FIXTURE"] = str(FIXTURES / fixture)
        environment["IFEVAL_EXPECTED_CACHE"] = str(cache)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(FAKE_DEPENDENCIES), environment.get("PYTHONPATH", "")]
        )
        environment.update(extra_environment or {})
        command = [
            sys.executable,
            str(BENCHMARK_ROOT / "prepare.py"),
            "--cache-dir",
            str(cache),
            "--output",
            str(output),
        ]
        if force:
            command.append("--force")
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
        )

    def run_grade(self, cases, responses, output):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(FAKE_DEPENDENCIES), environment.get("PYTHONPATH", "")]
        )
        return subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_ROOT / "grade.py"),
                "--cases",
                str(cases),
                "--responses",
                str(responses),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            env=environment,
        )

    def write_sample_run(self, root):
        responses = root / "responses.jsonl"
        first_response = (FIXTURES / "responses.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        responses.write_text(first_response + "\n", encoding="utf-8")
        manifest_path = Path(f"{responses}.manifest.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "population_total": 4,
                    "sample_size": 2,
                    "seed": 7,
                    "selected_ids": ["1", "4"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return responses, manifest_path

    def test_prepare_publishes_provider_rows_as_aligned_requests_and_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            output = root / "prepared"

            result = self.run_prepare(cache, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_symlink())
            first_version = os.readlink(output)
            self.assertEqual(
                json.loads((output / "requests.jsonl").read_text(encoding="utf-8")),
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply with the exact provider prompt unchanged.",
                        }
                    ],
                    "max_tokens": 8192,
                    "temperature": 0,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            replacement = self.run_prepare(cache, output)
            self.assertEqual(replacement.returncode, 0, replacement.stderr)
            self.assertTrue(output.is_symlink())
            self.assertEqual(os.readlink(output), first_version)
            self.assertIn("skipping", replacement.stdout)
            self.assertEqual((output / "ids.txt").read_text(encoding="utf-8"), "7\n")
            self.assertEqual(
                json.loads((output / "cases.jsonl").read_text(encoding="utf-8")),
                {
                    "id": "7",
                    "key": 7,
                    "prompt": "Reply with the exact provider prompt unchanged.",
                    "instruction_id_list": ["startend:quotation"],
                    "kwargs": [{}],
                },
            )

    def test_failed_prepare_does_not_replace_a_valid_prepared_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            output = root / "prepared"
            first = self.run_prepare(cache, output)
            self.assertEqual(first.returncode, 0, first.stderr)
            published = {
                name: (output / name).read_bytes()
                for name in ("requests.jsonl", "ids.txt", "cases.jsonl")
            }

            failed = self.run_prepare(cache, output, "duplicate_ids.json", force=True)

            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("duplicate", failed.stderr.lower())
            self.assertEqual(
                {
                    name: (output / name).read_bytes()
                    for name in ("requests.jsonl", "ids.txt", "cases.jsonl")
                },
                published,
            )

    def test_interrupted_atomic_swap_restores_the_previous_prepared_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            output = root / "prepared"
            first = self.run_prepare(cache, output)
            self.assertEqual(first.returncode, 0, first.stderr)
            previous_target = os.readlink(output)
            published = {
                name: (output / name).read_bytes()
                for name in ("requests.jsonl", "ids.txt", "cases.jsonl")
            }

            interrupted = self.run_prepare(
                cache,
                output,
                "changed_input_data.json",
                {"IFEVAL_FAIL_AFTER_SWAP": str(output.absolute())},
                force=True,
            )

            self.assertNotEqual(interrupted.returncode, 0)
            self.assertTrue(output.is_symlink())
            self.assertEqual(os.readlink(output), previous_target)
            self.assertEqual(
                {
                    name: (output / name).read_bytes()
                    for name in ("requests.jsonl", "ids.txt", "cases.jsonl")
                },
                published,
            )

    def test_grade_reports_provider_strict_and_loose_metrics_with_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "scores.json"

            result = self.run_grade(
                FIXTURES / "cases.jsonl", FIXTURES / "responses.jsonl", output
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {
                    "benchmark": "ifeval",
                    "profile": "qwen36-deterministic-no-thinking",
                    "expected": 4,
                    "received": 3,
                    "missing": 1,
                    "invalid": 1,
                    "metrics": {
                        "strict_prompt_level_accuracy": 0.25,
                        "strict_instruction_level_accuracy": 0.25,
                        "loose_prompt_level_accuracy": 0.5,
                        "loose_instruction_level_accuracy": 0.5,
                    },
                },
            )
            for label in (
                "benchmark: ifeval",
                "profile: qwen36-deterministic-no-thinking",
                "expected: 4",
                "received: 3",
                "missing: 1",
                "invalid: 1",
                "strict_prompt_level_accuracy: 0.250000",
                "strict_instruction_level_accuracy: 0.250000",
                "loose_prompt_level_accuracy: 0.500000",
                "loose_instruction_level_accuracy: 0.500000",
            ):
                self.assertIn(label, result.stdout)

    def test_grade_uses_sample_manifest_as_the_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses, _ = self.write_sample_run(root)
            output = root / "scores.json"

            result = self.run_grade(FIXTURES / "cases.jsonl", responses, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["expected"], 2)
            self.assertEqual(summary["received"], 1)
            self.assertEqual(summary["missing"], 1)
            self.assertEqual(summary["invalid"], 0)
            self.assertEqual(summary["metrics"]["strict_prompt_level_accuracy"], 0.5)
            self.assertEqual(summary["sampled"], True)
            self.assertEqual(summary["population_total"], 4)
            self.assertEqual(summary["sample_seed"], 7)
            self.assertIn("sample: 2 of 4 cases (seed 7)", result.stdout)

    def test_grade_rejects_non_integer_sample_manifest_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses, manifest_path = self.write_sample_run(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["population_total"] = 4.0
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            rejected = self.run_grade(
                FIXTURES / "cases.jsonl", responses, root / "rejected.json"
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_grade_rejects_responses_outside_the_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses, _ = self.write_sample_run(root)
            second_response = (FIXTURES / "responses.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()[1]
            responses.write_text(second_response + "\n", encoding="utf-8")
            outside_sample = self.run_grade(
                FIXTURES / "cases.jsonl", responses, root / "outside.json"
            )
            self.assertNotEqual(outside_sample.returncode, 0)

    def test_grade_rejects_duplicate_and_unknown_response_ids(self):
        invalid_responses = {
            "duplicate": (
                '{"id":"1","response":{}}\n'
                '{"id":"1","response":{}}\n'
            ),
            "unknown": '{"id":"unknown","response":{}}\n',
        }
        for name, response_text in invalid_responses.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                responses = root / "responses.jsonl"
                responses.write_text(response_text, encoding="utf-8")
                output = root / "scores.json"

                result = self.run_grade(FIXTURES / "cases.jsonl", responses, output)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr.lower())
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
