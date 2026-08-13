import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests.test_run import FakeInferenceServer


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "longbench_v2"
PREPARE = BENCHMARK / "prepare.py"
GRADE = BENCHMARK / "grade.py"


class LongBenchV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.work = Path(self.temporary_directory.name)
        self.fake_modules = self.work / "fake_modules"
        self.fake_modules.mkdir()
        self.fixture_path = self.work / "dataset.json"
        self.call_log = self.work / "calls.jsonl"
        self._write_fake_hugging_face_modules()

    def _write_fake_hugging_face_modules(self):
        (self.fake_modules / "datasets.py").write_text(
            textwrap.dedent(
                """
                import json
                import os

                def load_dataset(repo_id, **kwargs):
                    with open(os.environ["LXBENCH_FAKE_CALL_LOG"], "a", encoding="utf-8") as log:
                        call = {"call": "load_dataset", "repo_id": repo_id, "kwargs": kwargs}
                        log.write(json.dumps(call) + "\\n")
                    with open(os.environ["LXBENCH_FAKE_DATASET"], encoding="utf-8") as fixture:
                        return json.load(fixture)
                """
            ),
            encoding="utf-8",
        )
        (self.fake_modules / "huggingface_hub.py").write_text(
            textwrap.dedent(
                """
                import json
                import os
                from pathlib import Path

                def snapshot_download(repo_id, **kwargs):
                    with open(os.environ["LXBENCH_FAKE_CALL_LOG"], "a", encoding="utf-8") as log:
                        call = {"call": "snapshot_download", "repo_id": repo_id, "kwargs": kwargs}
                        log.write(json.dumps(call) + "\\n")
                    local_dir = Path(kwargs["local_dir"])
                    local_dir.mkdir(parents=True, exist_ok=True)
                    return str(local_dir)
                """
            ),
            encoding="utf-8",
        )
        (self.fake_modules / "transformers.py").write_text(
            textwrap.dedent(
                """
                import json
                import os

                class CharacterTokenizer:
                    def encode(self, text, add_special_tokens=False):
                        return [ord(character) for character in text]

                    def decode(self, token_ids, skip_special_tokens=True):
                        return ''.join(chr(token_id) for token_id in token_ids)

                    def apply_chat_template(
                        self, messages, tokenize, add_generation_prompt, **kwargs
                    ):
                        rendered = '<user>' + messages[0]['content'] + '</user><assistant>'
                        if tokenize:
                            return self.encode(rendered)
                        return rendered

                class AutoTokenizer:
                    @classmethod
                    def from_pretrained(cls, path, **kwargs):
                        with open(
                            os.environ["LXBENCH_FAKE_CALL_LOG"], "a", encoding="utf-8"
                        ) as log:
                            call = {
                                "call": "from_pretrained",
                                "path": str(path),
                                "kwargs": kwargs,
                            }
                            log.write(json.dumps(call) + "\\n")
                        return CharacterTokenizer()
                """
            ),
            encoding="utf-8",
        )
        (self.fake_modules / "sitecustomize.py").write_text(
            textwrap.dedent(
                """
                import os

                failure_target = os.environ.get("LXBENCH_FAIL_AFTER_SWAP")
                if failure_target:
                    original_replace = os.replace
                    failed = False

                    def replace(source, destination):
                        global failed
                        original_replace(source, destination)
                        if not failed and os.path.abspath(destination) == failure_target:
                            failed = True
                            raise KeyboardInterrupt("fixture interrupt after atomic swap")

                    os.replace = replace
                """
            ),
            encoding="utf-8",
        )

    def run_prepare(self, *arguments, extra_environment=None):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.fake_modules)
        environment["LXBENCH_FAKE_DATASET"] = str(self.fixture_path)
        environment["LXBENCH_FAKE_CALL_LOG"] = str(self.call_log)
        environment.update(extra_environment or {})
        return subprocess.run(
            [sys.executable, str(PREPARE), *map(str, arguments)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )

    def read_jsonl(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def write_jsonl(self, path, records):
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def chat_response(self, content):
        return {"choices": [{"message": {"content": content}}]}

    def run_grade(self, *arguments):
        return subprocess.run(
            [sys.executable, str(GRADE), *map(str, arguments)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_prepare_emits_official_prompt_profile_and_tokenizer_only_download(self):
        row = {
            "_id": "fixture-short",
            "domain": "Single-Document QA",
            "sub_domain": "Technical manual",
            "difficulty": "easy",
            "length": "short",
            "question": "Which option is supported?",
            "choice_A": "Alpha",
            "choice_B": "Beta",
            "choice_C": "Gamma",
            "choice_D": "Delta",
            "answer": "B",
            "context": "The manual explicitly supports Beta.",
        }
        self.fixture_path.write_text(json.dumps([row]), encoding="utf-8")
        output = self.work / "prepared"

        result = self.run_prepare(
            "--context-size",
            4096,
            "--cache",
            self.work / "cache",
            "--output",
            output,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        expected_prompt = textwrap.dedent(
            """\
            Please read the following text and answer the question below.

            <text>
            The manual explicitly supports Beta.
            </text>

            What is the correct answer to this question: Which option is supported?
            Choices:
            (A) Alpha
            (B) Beta
            (C) Gamma
            (D) Delta

            Format your response as follows: "The correct answer is (insert answer here)".
            """
        )
        self.assertEqual(
            self.read_jsonl(output / "requests.jsonl"),
            [
                {
                    "messages": [{"role": "user", "content": expected_prompt}],
                    "max_tokens": 128,
                    "temperature": 0,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ],
        )
        self.assertEqual((output / "ids.txt").read_text(encoding="utf-8"), "fixture-short\n")
        self.assertEqual(
            self.read_jsonl(output / "cases.jsonl"),
            [
                {
                    "id": "fixture-short",
                    "answer": "B",
                    "category": "Single-Document QA",
                    "difficulty": "easy",
                    "length": "short",
                    "truncated": False,
                }
            ],
        )

        calls = self.read_jsonl(self.call_log)
        dataset_call = next(call for call in calls if call["call"] == "load_dataset")
        self.assertEqual(dataset_call["repo_id"], "zai-org/LongBench-v2")
        self.assertNotIn("revision", dataset_call["kwargs"])
        self.assertEqual(
            dataset_call["kwargs"]["download_mode"], "reuse_dataset_if_exists"
        )
        tokenizer_call = next(call for call in calls if call["call"] == "snapshot_download")
        self.assertEqual(tokenizer_call["repo_id"], "Qwen/Qwen3.6-35B-A3B")
        self.assertNotIn("revision", tokenizer_call["kwargs"])
        self.assertEqual(
            Path(tokenizer_call["kwargs"]["local_dir"]), self.work / "cache/tokenizer"
        )
        allowed = tokenizer_call["kwargs"]["allow_patterns"]
        self.assertEqual(
            allowed,
            [
                "chat_template.jinja",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
        )
        load_call = next(call for call in calls if call["call"] == "from_pretrained")
        self.assertTrue(load_call["kwargs"]["local_files_only"])

        calls_before = self.call_log.read_text(encoding="utf-8")
        skipped = self.run_prepare(
            "--context-size",
            4096,
            "--cache",
            self.work / "cache",
            "--output",
            output,
        )
        self.assertEqual(skipped.returncode, 0, skipped.stderr)
        self.assertIn("skipping", skipped.stdout)
        self.assertEqual(self.call_log.read_text(encoding="utf-8"), calls_before)

    def test_prepare_requires_effective_context_size(self):
        self.fixture_path.write_text("[]", encoding="utf-8")

        result = self.run_prepare(
            "--cache",
            self.work / "cache",
            "--output",
            self.work / "prepared",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--context-size", result.stderr)
        self.assertFalse(self.call_log.exists())

    def test_prepare_middle_truncates_to_chat_input_budget_and_preserves_both_ends(self):
        row = {
            "_id": "fixture-long",
            "domain": "Code Repository Understanding",
            "sub_domain": "Repository QA",
            "difficulty": "hard",
            "length": "long",
            "question": "Where is the answer?",
            "choice_A": "At the start",
            "choice_B": "In the removed middle",
            "choice_C": "At the end",
            "choice_D": "Nowhere",
            "answer": "C",
            "context": "BEGINNING-EVIDENCE " + ("middle-noise " * 80) + " ENDING-EVIDENCE",
        }
        self.fixture_path.write_text(json.dumps([row]), encoding="utf-8")
        output = self.work / "prepared"
        context_size = 360

        result = self.run_prepare(
            "--context-size",
            context_size,
            "--cache",
            self.work / "cache",
            "--output",
            output,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        request = self.read_jsonl(output / "requests.jsonl")[0]
        prompt = request["messages"][0]["content"]
        rendered_chat = "<user>" + prompt + "</user><assistant>"
        self.assertLessEqual(len(rendered_chat), context_size - 128)
        self.assertTrue(prompt.startswith("Please read the following text"))
        self.assertTrue(
            prompt.endswith(
                'Format your response as follows: "The correct answer is (insert answer here)".\n'
            )
        )
        self.assertNotIn("middle-noise middle-noise middle-noise", prompt)
        case = self.read_jsonl(output / "cases.jsonl")[0]
        self.assertTrue(case["truncated"])

    def test_prepare_validates_all_records_before_replacing_published_files(self):
        self.fixture_path.write_text("[]", encoding="utf-8")
        output = self.work / "prepared"
        output.mkdir()
        previous = {
            "requests.jsonl": '{"previous": true}\n',
            "ids.txt": "previous\n",
            "cases.jsonl": '{"id": "previous"}\n',
        }
        for filename, content in previous.items():
            (output / filename).write_text(content, encoding="utf-8")

        result = self.run_prepare(
            "--context-size",
            4096,
            "--cache",
            self.work / "cache",
            "--output",
            output,
        )

        self.assertNotEqual(result.returncode, 0)
        for filename, content in previous.items():
            self.assertEqual((output / filename).read_text(encoding="utf-8"), content)

    def test_prepare_atomically_switches_complete_validated_generations(self):
        row = {
            "_id": "first",
            "domain": "Single-Document QA",
            "sub_domain": "Atomic publish",
            "difficulty": "easy",
            "length": "short",
            "question": "Which choice?",
            "choice_A": "Alpha",
            "choice_B": "Beta",
            "choice_C": "Gamma",
            "choice_D": "Delta",
            "answer": "A",
            "context": "Alpha is correct.",
        }
        self.fixture_path.write_text(json.dumps([row]), encoding="utf-8")
        output = self.work / "prepared"
        first_result = self.run_prepare(
            "--context-size",
            4096,
            "--cache",
            self.work / "cache",
            "--output",
            output,
        )
        self.assertEqual(first_result.returncode, 0, first_result.stderr)
        first_generation = output.resolve()

        row["_id"] = "second"
        self.fixture_path.write_text(json.dumps([row]), encoding="utf-8")
        second_result = self.run_prepare(
            "--context-size",
            4096,
            "--cache",
            self.work / "cache",
            "--output",
            output,
            "--force",
        )

        self.assertEqual(second_result.returncode, 0, second_result.stderr)
        self.assertTrue(output.is_symlink())
        self.assertNotEqual(output.resolve(), first_generation)
        self.assertFalse(first_generation.exists())
        self.assertEqual((output / "ids.txt").read_text(encoding="utf-8"), "second\n")

    def test_prepare_interrupted_after_swap_restores_previous_generation(self):
        row = {
            "_id": "first",
            "domain": "Single-Document QA",
            "sub_domain": "Atomic publish",
            "difficulty": "easy",
            "length": "short",
            "question": "Which choice?",
            "choice_A": "Alpha",
            "choice_B": "Beta",
            "choice_C": "Gamma",
            "choice_D": "Delta",
            "answer": "A",
            "context": "Alpha is correct.",
        }
        self.fixture_path.write_text(json.dumps([row]), encoding="utf-8")
        output = self.work / "prepared"
        first_result = self.run_prepare(
            "--context-size",
            4096,
            "--cache",
            self.work / "cache",
            "--output",
            output,
        )
        self.assertEqual(first_result.returncode, 0, first_result.stderr)
        previous_target = os.readlink(output)
        previous = {
            name: (output / name).read_bytes()
            for name in ("requests.jsonl", "ids.txt", "cases.jsonl")
        }

        row["_id"] = "second"
        self.fixture_path.write_text(json.dumps([row]), encoding="utf-8")
        interrupted = self.run_prepare(
            "--context-size",
            4096,
            "--cache",
            self.work / "cache",
            "--output",
            output,
            "--force",
            extra_environment={"LXBENCH_FAIL_AFTER_SWAP": str(output.absolute())},
        )

        self.assertNotEqual(interrupted.returncode, 0)
        self.assertTrue(output.is_symlink())
        self.assertEqual(os.readlink(output), previous_target)
        self.assertEqual(
            {
                name: (output / name).read_bytes()
                for name in ("requests.jsonl", "ids.txt", "cases.jsonl")
            },
            previous,
        )

    def test_grade_uses_provider_extraction_and_standard_breakdowns(self):
        case_fields = [
            ("one", "A", "A", "easy", "short", False),
            ("two", "B", "A", "easy", "medium", False),
            ("three", "D", "B", "hard", "long", True),
            ("four", "C", "B", "hard", "short", False),
            ("five", "A", "C", "easy", "long", True),
            ("six", "B", "C", "hard", "medium", False),
        ]
        cases = [
            {
                "id": prepared_id,
                "answer": answer,
                "category": category,
                "difficulty": difficulty,
                "length": length,
                "truncated": truncated,
            }
            for prepared_id, answer, category, difficulty, length, truncated in case_fields
        ]
        responses = [
            {"id": "six", "response": self.chat_response("The correct answer is (B)")},
            {"id": "four", "response": self.chat_response(42)},
            {"id": "one", "response": self.chat_response("**The correct answer is (A)**")},
            {"id": "three", "response": self.chat_response("I cannot tell.")},
            {"id": "two", "response": self.chat_response("The correct answer is C")},
        ]
        cases_path = self.work / "cases.jsonl"
        responses_path = self.work / "responses.jsonl"
        self.write_jsonl(cases_path, cases)
        self.write_jsonl(responses_path, responses)
        output = self.work / "score.json"

        result = self.run_grade(
            "--cases",
            cases_path,
            "--responses",
            responses_path,
            "--output",
            output,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            summary,
            {
                "benchmark": "longbench-v2",
                "profile": "qwen36-deterministic-no-thinking",
                "correct": 2,
                "incorrect": 4,
                "invalid": 2,
                "missing": 1,
                "total": 6,
                "expected": 6,
                "received": 5,
                "metrics": {
                    "overall_accuracy": 33.3,
                    "difficulty": {"easy": 33.3, "hard": 33.3},
                    "context_length": {"short": 50.0, "medium": 50.0, "long": 0.0},
                },
            },
        )
        for label in (
            "overall_accuracy",
            "difficulty",
            "context_length",
            "correct",
            "incorrect",
            "invalid",
            "missing",
            "total",
            "expected",
            "received",
            "benchmark",
            "profile",
        ):
            self.assertIn(label, result.stdout)

    def test_grade_rejects_duplicate_and_unknown_response_ids(self):
        cases_path = self.work / "cases.jsonl"
        self.write_jsonl(
            cases_path,
            [
                {
                    "id": "known",
                    "answer": "A",
                    "category": "QA",
                    "difficulty": "easy",
                    "length": "short",
                    "truncated": False,
                }
            ],
        )
        response = {
            "id": "known",
            "response": {"choices": [{"message": {"content": "The correct answer is (A)"}}]},
        }
        fixtures = {
            "duplicate": [response, response],
            "unknown": [{**response, "id": "unknown"}],
        }
        for name, responses in fixtures.items():
            with self.subTest(name=name):
                responses_path = self.work / f"{name}.jsonl"
                output = self.work / f"{name}.json"
                self.write_jsonl(responses_path, responses)
                result = self.run_grade(
                    "--cases",
                    cases_path,
                    "--responses",
                    responses_path,
                    "--output",
                    output,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)
                self.assertFalse(output.exists())

    def test_grade_rejects_non_json_constants(self):
        cases_path = self.work / "cases.jsonl"
        self.write_jsonl(
            cases_path,
            [
                {
                    "id": "known",
                    "answer": "A",
                    "category": "QA",
                    "difficulty": "easy",
                    "length": "short",
                    "truncated": False,
                }
            ],
        )
        response_template = (
            '{"id":"known","response":{"choices":[{"message":'
            '{"content":CONSTANT}}]}}\n'
        )

        for constant in ("NaN", "Infinity"):
            with self.subTest(constant=constant):
                responses_path = self.work / f"responses-{constant}.jsonl"
                responses_path.write_text(
                    response_template.replace("CONSTANT", constant), encoding="utf-8"
                )
                output = self.work / f"score-{constant}.json"

                result = self.run_grade(
                    "--cases",
                    cases_path,
                    "--responses",
                    responses_path,
                    "--output",
                    output,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("non-JSON constant", result.stderr)
                self.assertFalse(output.exists())

    def test_prepare_runner_grade_smoke_flow_uses_unchanged_generic_runner(self):
        row = {
            "_id": "smoke",
            "domain": "Single-Document QA",
            "sub_domain": "Smoke",
            "difficulty": "easy",
            "length": "short",
            "question": "Which choice is correct?",
            "choice_A": "This one",
            "choice_B": "Not this one",
            "choice_C": "Not this one",
            "choice_D": "Not this one",
            "answer": "A",
            "context": "The first choice is correct.",
        }
        self.fixture_path.write_text(json.dumps([row]), encoding="utf-8")
        prepared = self.work / "prepared"
        prepare_result = self.run_prepare(
            "--context-size",
            4096,
            "--cache",
            self.work / "cache",
            "--output",
            prepared,
        )
        self.assertEqual(prepare_result.returncode, 0, prepare_result.stderr)
        raw = self.work / "raw.jsonl"
        server_response = json.dumps(
            {"choices": [{"message": {"content": "The correct answer is (A)"}}]}
        ).encode()

        with FakeInferenceServer([(200, server_response)]) as server:
            runner_result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "--endpoint",
                    server.endpoint,
                    "--requests",
                    str(prepared / "requests.jsonl"),
                    "--output",
                    str(raw),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(runner_result.returncode, 0, runner_result.stderr)
        self.assertEqual(
            server.requests[0]["body"],
            self.read_jsonl(prepared / "requests.jsonl")[0],
        )
        score = self.work / "score.json"
        grade_result = self.run_grade(
            "--cases",
            prepared / "cases.jsonl",
            "--responses",
            raw,
            "--output",
            score,
        )
        self.assertEqual(grade_result.returncode, 0, grade_result.stderr)
        summary = json.loads(score.read_text(encoding="utf-8"))
        self.assertEqual(summary["metrics"]["overall_accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()
