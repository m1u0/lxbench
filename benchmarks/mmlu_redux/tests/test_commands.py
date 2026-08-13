import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


BENCHMARK = Path(__file__).resolve().parents[1]
PREPARE = BENCHMARK / "prepare.py"
GRADE = BENCHMARK / "grade.py"
FIXTURE_PROVIDER = Path(__file__).parent / "fixtures" / "provider"
RUNNER = Path(__file__).resolve().parents[3] / "run.py"


class DirectAnswerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        json.loads(self.rfile.read(content_length))
        response = json.dumps(
            {"choices": [{"message": {"content": "B"}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass


class MMLUReduxCommandTests(unittest.TestCase):
    def write_jsonl(self, path, records):
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def write_sample_grade_inputs(self, root):
        cases_path = root / "cases.jsonl"
        responses_path = root / "raw.jsonl"
        self.write_jsonl(
            cases_path,
            [
                {"id": "alpha", "subject": "math", "answer": 0},
                {"id": "beta", "subject": "math", "answer": 1},
                {"id": "gamma", "subject": "science", "answer": 2},
            ],
        )
        self.write_jsonl(
            responses_path,
            [
                {
                    "id": "alpha",
                    "response": {"choices": [{"message": {"content": "A"}}]},
                }
            ],
        )
        manifest_path = Path(f"{responses_path}.manifest.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "population_total": 3,
                    "sample_size": 2,
                    "seed": 7,
                    "selected_ids": ["alpha", "beta"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return cases_path, responses_path, manifest_path

    def run_prepare(
        self, cache_dir, output_dir, log_path, extra_environment=None, force=False
    ):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(FIXTURE_PROVIDER)
        environment["MMLU_FIXTURE_LOG"] = str(log_path)
        environment.update(extra_environment or {})
        command = [
            sys.executable,
            str(PREPARE),
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(output_dir),
        ]
        if force:
            command.append("--force")
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_prepare_reuses_official_cache_and_publishes_aligned_profile(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_dir = root / "cache"
            output_dir = root / "prepared"
            log_path = root / "provider-calls.jsonl"

            result = self.run_prepare(cache_dir, output_dir, log_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            ids = (output_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
            requests = [
                json.loads(line)
                for line in (output_dir / "requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            cases = [
                json.loads(line)
                for line in (output_dir / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(ids, ["abstract_algebra:0", "anatomy:1"])
            self.assertEqual(ids, [case["id"] for case in cases])
            self.assertEqual([case["subject"] for case in cases], ["abstract_algebra", "anatomy"])
            self.assertEqual([case["answer"] for case in cases], [1, 1])
            self.assertEqual(len(requests), len(ids))

            expected_request_fields = {
                "messages",
                "max_tokens",
                "temperature",
                "stream",
                "chat_template_kwargs",
            }
            for request in requests:
                self.assertEqual(set(request), expected_request_fields)
                self.assertEqual(request["max_tokens"], 16)
                self.assertEqual(request["temperature"], 0)
                self.assertIs(request["stream"], False)
                self.assertEqual(
                    request["chat_template_kwargs"], {"enable_thinking": False}
                )
                self.assertEqual(len(request["messages"]), 1)
                self.assertEqual(request["messages"][0]["role"], "user")
                prompt = request["messages"][0]["content"]
                for letter in "ABCD":
                    self.assertIn(f"{letter}.", prompt)
                self.assertIn("answer letter", prompt.lower())

            calls = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(calls[0]["path"], "edinburgh-dawg/mmlu-redux-2.0")
            self.assertNotIn("revision", calls[0]["kwargs"])
            load_calls = [call for call in calls if call["function"] == "load_dataset"]
            self.assertEqual(
                [call["name"] for call in load_calls],
                ["abstract_algebra", "anatomy"],
            )
            for call in load_calls:
                self.assertEqual(call["path"], "edinburgh-dawg/mmlu-redux-2.0")
                self.assertEqual(call["kwargs"]["cache_dir"], str(cache_dir))
                self.assertEqual(
                    call["kwargs"]["download_mode"], "reuse_dataset_if_exists"
                )
                self.assertEqual(call["kwargs"]["split"], "test")
                self.assertNotIn("revision", call["kwargs"])

            calls_before = log_path.read_text(encoding="utf-8")
            skipped = self.run_prepare(cache_dir, output_dir, log_path)
            self.assertEqual(skipped.returncode, 0, skipped.stderr)
            self.assertIn("skipping", skipped.stdout)
            self.assertEqual(log_path.read_text(encoding="utf-8"), calls_before)

    def test_failed_atomic_swap_preserves_the_live_prepared_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_dir = root / "cache"
            output_dir = root / "prepared"
            log_path = root / "provider-calls.jsonl"

            first_result = self.run_prepare(cache_dir, output_dir, log_path)
            self.assertEqual(first_result.returncode, 0, first_result.stderr)
            previous_files = {
                path.name: path.read_bytes() for path in output_dir.iterdir()
            }
            replace_log = root / "replace.log"

            failed_result = self.run_prepare(
                cache_dir,
                output_dir,
                log_path,
                {
                    "MMLU_FIXTURE_CHANGED": "1",
                    "MMLU_FIXTURE_FAIL_ATOMIC_SWAP": "1",
                    "MMLU_FIXTURE_REPLACE_LOG": str(replace_log),
                },
                force=True,
            )

            self.assertNotEqual(failed_result.returncode, 0)
            self.assertEqual(
                previous_files,
                {path.name: path.read_bytes() for path in output_dir.iterdir()},
            )
            replace_sources = [
                line.split("\t", 1)[0]
                for line in replace_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertNotIn(str(output_dir), replace_sources)

    def test_interrupted_atomic_swap_restores_the_live_prepared_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_dir = root / "cache"
            output_dir = root / "prepared"
            log_path = root / "provider-calls.jsonl"

            first_result = self.run_prepare(cache_dir, output_dir, log_path)
            self.assertEqual(first_result.returncode, 0, first_result.stderr)
            previous_target = os.readlink(output_dir)
            previous_files = {
                path.name: path.read_bytes() for path in output_dir.iterdir()
            }

            interrupted_result = self.run_prepare(
                cache_dir,
                output_dir,
                log_path,
                {
                    "MMLU_FIXTURE_CHANGED": "1",
                    "MMLU_FIXTURE_FAIL_AFTER_ATOMIC_SWAP": "1",
                },
                force=True,
            )

            self.assertNotEqual(interrupted_result.returncode, 0)
            self.assertTrue(output_dir.is_symlink())
            self.assertEqual(os.readlink(output_dir), previous_target)
            self.assertEqual(
                previous_files,
                {path.name: path.read_bytes() for path in output_dir.iterdir()},
            )

    def test_interrupted_generation_cleanup_keeps_the_new_prepared_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_dir = root / "cache"
            output_dir = root / "prepared"
            log_path = root / "provider-calls.jsonl"

            first_result = self.run_prepare(cache_dir, output_dir, log_path)
            self.assertEqual(first_result.returncode, 0, first_result.stderr)
            previous_target = os.readlink(output_dir)

            interrupted_result = self.run_prepare(
                cache_dir,
                output_dir,
                log_path,
                {
                    "MMLU_FIXTURE_CHANGED": "1",
                    "MMLU_FIXTURE_FAIL_GENERATION_CLEANUP": "1",
                },
                force=True,
            )

            self.assertNotEqual(interrupted_result.returncode, 0)
            self.assertTrue(output_dir.is_symlink())
            self.assertTrue(output_dir.exists())
            self.assertNotEqual(os.readlink(output_dir), previous_target)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"requests.jsonl", "ids.txt", "cases.jsonl"},
            )
            ids = (output_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
            requests = [
                json.loads(line)
                for line in (output_dir / "requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            cases = [
                json.loads(line)
                for line in (output_dir / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(requests), len(ids))
            self.assertEqual([case["id"] for case in cases], ids)
            self.assertIn("Changed", requests[0]["messages"][0]["content"])

    def test_grade_extracts_one_standalone_letter_and_reports_known_scores(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.jsonl"
            responses_path = root / "raw.jsonl"
            score_path = root / "scores" / "summary.json"
            self.write_jsonl(
                cases_path,
                [
                    {"id": "math:0", "subject": "math", "answer": 0},
                    {"id": "math:1", "subject": "math", "answer": 1},
                    {"id": "science:0", "subject": "science", "answer": 2},
                    {"id": "science:1", "subject": "science", "answer": 3},
                    {"id": "science:2", "subject": "science", "answer": 0},
                    {"id": "logic:0", "subject": "logic", "answer": 2},
                    {"id": "logic:1", "subject": "logic", "answer": 3},
                    {"id": "logic:2", "subject": "logic", "answer": 0},
                ],
            )
            self.write_jsonl(
                responses_path,
                [
                    {
                        "id": "math:0",
                        "response": {"choices": [{"message": {"content": " A "}}]},
                    },
                    {
                        "id": "math:1",
                        "response": {
                            "choices": [{"message": {"content": "Answer: b."}}]
                        },
                    },
                    {
                        "id": "science:0",
                        "response": {
                            "choices": [{"message": {"content": "C or D"}}]
                        },
                    },
                    {"id": "science:2", "response": {"choices": []}},
                    {
                        "id": "logic:0",
                        "response": {"choices": [{"message": {"content": "cab"}}]},
                    },
                    {
                        "id": "logic:1",
                        "response": {"choices": [{"message": {"content": "a"}}]},
                    },
                    {
                        "id": "logic:2",
                        "response": {"choices": [{"message": {"content": "A A"}}]},
                    },
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(GRADE),
                    "--cases",
                    str(cases_path),
                    "--responses",
                    str(responses_path),
                    "--output",
                    str(score_path),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(score_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(result.stdout), summary)
            self.assertEqual(summary["benchmark"], "mmlu-redux-2.0")
            self.assertEqual(summary["profile"], "qwen36-deterministic-no-thinking")
            self.assertEqual(summary["expected"], 8)
            self.assertEqual(summary["received"], 7)
            self.assertEqual(summary["total"], 8)
            self.assertEqual(summary["correct"], 2)
            self.assertEqual(summary["incorrect"], 6)
            self.assertEqual(summary["invalid"], 4)
            self.assertEqual(summary["missing"], 1)
            self.assertEqual(summary["metrics"]["accuracy"], 0.25)
            self.assertEqual(
                summary["metrics"]["per_subject"],
                {
                    "logic": {
                        "accuracy": 0.0,
                        "correct": 0,
                        "incorrect": 3,
                        "total": 3,
                    },
                    "math": {
                        "accuracy": 1.0,
                        "correct": 2,
                        "incorrect": 0,
                        "total": 2,
                    },
                    "science": {
                        "accuracy": 0.0,
                        "correct": 0,
                        "incorrect": 3,
                        "total": 3,
                    },
                },
            )

    def test_grade_uses_sample_manifest_as_the_denominator(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path, responses_path, _ = self.write_sample_grade_inputs(root)
            score_path = root / "summary.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(GRADE),
                    "--cases",
                    str(cases_path),
                    "--responses",
                    str(responses_path),
                    "--output",
                    str(score_path),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(score_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["expected"], 2)
            self.assertEqual(summary["received"], 1)
            self.assertEqual(summary["missing"], 1)
            self.assertEqual(summary["incorrect"], 1)
            self.assertEqual(summary["metrics"]["accuracy"], 0.5)
            self.assertEqual(summary["sampled"], True)
            self.assertEqual(summary["population_total"], 3)
            self.assertEqual(summary["sample_seed"], 7)

    def test_grade_rejects_non_integer_sample_manifest_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path, responses_path, manifest_path = self.write_sample_grade_inputs(
                root
            )
            score_path = root / "summary.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["version"] = True
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(GRADE),
                    "--cases",
                    str(cases_path),
                    "--responses",
                    str(responses_path),
                    "--output",
                    str(score_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_grade_rejects_responses_outside_the_sample(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path, responses_path, _ = self.write_sample_grade_inputs(root)
            score_path = root / "summary.json"
            self.write_jsonl(
                responses_path,
                [
                    {
                        "id": "gamma",
                        "response": {"choices": [{"message": {"content": "C"}}]},
                    }
                ],
            )
            outside_sample = subprocess.run(
                [
                    sys.executable,
                    str(GRADE),
                    "--cases",
                    str(cases_path),
                    "--responses",
                    str(responses_path),
                    "--output",
                    str(score_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(outside_sample.returncode, 0)

    def test_grade_rejects_duplicate_and_unknown_response_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.jsonl"
            self.write_jsonl(
                cases_path,
                [{"id": "math:0", "subject": "math", "answer": 0}],
            )
            response = {
                "id": "math:0",
                "response": {"choices": [{"message": {"content": "A"}}]},
            }
            scenarios = {
                "duplicate": [response, response],
                "unknown": [{**response, "id": "math:1"}],
            }
            for name, responses in scenarios.items():
                with self.subTest(name=name):
                    responses_path = root / f"{name}.jsonl"
                    output_path = root / f"{name}.json"
                    self.write_jsonl(responses_path, responses)
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(GRADE),
                            "--cases",
                            str(cases_path),
                            "--responses",
                            str(responses_path),
                            "--output",
                            str(output_path),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output_path.exists())

    def test_grade_rejects_non_json_constants(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.jsonl"
            self.write_jsonl(
                cases_path,
                [{"id": "math:0", "subject": "math", "answer": 0}],
            )

            for constant in ("NaN", "Infinity"):
                with self.subTest(constant=constant):
                    responses_path = root / f"{constant}.jsonl"
                    output_path = root / f"{constant}.json"
                    responses_path.write_text(
                        '{"id":"math:0","response":{"choices":'
                        '[{"message":{"content":"A"}}],"usage":'
                        f"{constant}}}}}\n",
                        encoding="utf-8",
                    )

                    result = subprocess.run(
                        [
                            sys.executable,
                            str(GRADE),
                            "--cases",
                            str(cases_path),
                            "--responses",
                            str(responses_path),
                            "--output",
                            str(output_path),
                        ],
                        capture_output=True,
                        text=True,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output_path.exists())

    def test_prepare_to_generic_runner_to_grade_smoke_flow(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prepared = root / "prepared"
            prepare_result = self.run_prepare(
                root / "cache", prepared, root / "provider-calls.jsonl"
            )
            self.assertEqual(prepare_result.returncode, 0, prepare_result.stderr)

            server = ThreadingHTTPServer(("127.0.0.1", 0), DirectAnswerHandler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            try:
                raw_run = root / "raw.jsonl"
                run_result = subprocess.run(
                    [
                        sys.executable,
                        str(RUNNER),
                        "--endpoint",
                        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                        "--requests",
                        str(prepared / "requests.jsonl"),
                        "--output",
                        str(raw_run),
                    ],
                    capture_output=True,
                    text=True,
                )
            finally:
                server.shutdown()
                server.server_close()
                server_thread.join()
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

            score_path = root / "summary.json"
            grade_result = subprocess.run(
                [
                    sys.executable,
                    str(GRADE),
                    "--cases",
                    str(prepared / "cases.jsonl"),
                    "--responses",
                    str(raw_run),
                    "--output",
                    str(score_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(grade_result.returncode, 0, grade_result.stderr)
            summary = json.loads(score_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"]["accuracy"], 1.0)
            self.assertEqual(summary["correct"], 2)
            self.assertEqual(summary["received"], 2)


if __name__ == "__main__":
    unittest.main()
