import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_RUNNER = ROOT / "run.py"
BASH_RUNNER = ROOT / "run.sh"


class FakeInferenceServer:
    def __init__(self, responses=None, response_for=None):
        if (responses is None) == (response_for is None):
            raise ValueError("provide responses or response_for")

        self.responses = None if responses is None else list(responses)
        self.response_for = response_for
        self.requests = []
        self.active_requests = 0
        self.max_active_requests = 0
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                request_body = json.loads(self.rfile.read(length))
                with owner.lock:
                    owner.requests.append(
                        {
                            "body": request_body,
                            "content_type": self.headers["Content-Type"],
                        }
                    )
                    owner.active_requests += 1
                    owner.max_active_requests = max(
                        owner.max_active_requests, owner.active_requests
                    )
                    response = (
                        None
                        if owner.response_for is not None
                        else owner.responses.pop(0)
                    )

                try:
                    status, body = (
                        owner.response_for(request_body)
                        if owner.response_for is not None
                        else response
                    )
                    if status is None:
                        self.connection.shutdown(2)
                        self.connection.close()
                        return
                    if callable(body):
                        body = body()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with owner.lock:
                        owner.active_requests -= 1

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)

    @property
    def endpoint(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1/chat/completions"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class RunnerContract:
    runner_command = ()
    validates_json = False

    def run_runner(
        self, endpoint, requests_path, output, environment=None, extra_args=()
    ):
        return subprocess.run(
            [
                *self.runner_command,
                "--endpoint",
                endpoint,
                "--requests",
                str(requests_path),
                "--output",
                str(output),
                *extra_args,
            ],
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_posts_opaque_request_and_appends_exact_envelope(self):
        request = {
            "messages": [{"role": "user", "content": "Say hello"}],
            "temperature": 0,
            "stream": False,
        }
        response = {"choices": [{"message": {"content": "Hello"}}]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            requests_path = prepared / "requests.jsonl"
            requests_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            (prepared / "ids.txt").write_text("case-1\n", encoding="utf-8")
            output = root / "nested" / "raw.jsonl"

            with FakeInferenceServer([(200, json.dumps(response).encode())]) as server:
                result = self.run_runner(server.endpoint, requests_path, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                server.requests,
                [{"body": request, "content_type": "application/json"}],
            )
            self.assertEqual(
                [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()],
                [{"id": "case-1", "response": response}],
            )

    def test_unicode_line_separator_inside_json_string_is_not_a_record_boundary(self):
        request = {"content": "first paragraph\u2028second paragraph"}
        response = {"answer": "received"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            requests_path.write_text(
                json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (root / "ids.txt").write_text("case-1\n", encoding="utf-8")
            output = root / "raw.jsonl"

            with FakeInferenceServer([(200, json.dumps(response).encode())]) as server:
                result = self.run_runner(server.endpoint, requests_path, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(server.requests[0]["body"], request)

    def test_posts_request_larger_than_process_argument_limit(self):
        request = {"content": "x" * (os.sysconf("SC_ARG_MAX") + 1_024)}
        response = {"answer": "received"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            requests_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            (root / "ids.txt").write_text("large\n", encoding="utf-8")
            output = root / "raw.jsonl"

            with FakeInferenceServer([(200, json.dumps(response).encode())]) as server:
                result = self.run_runner(server.endpoint, requests_path, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(server.requests[0]["body"], request)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"id": "large", "response": response},
            )

    def test_bounds_concurrency_and_appends_in_completion_order(self):
        first_two_started = threading.Event()
        second_completed = threading.Event()
        start_lock = threading.Lock()
        started = 0

        def response_for(request_body):
            nonlocal started
            case = request_body["case"]
            if case in (1, 2):
                with start_lock:
                    started += 1
                    if started == 2:
                        first_two_started.set()
                if not first_two_started.wait(timeout=5):
                    raise AssertionError("two requests did not run concurrently")
            if case == 2:
                second_completed.set()
            elif case == 1:
                if not second_completed.wait(timeout=5):
                    raise AssertionError("second request did not complete first")
                time.sleep(0.01)
            response = {"answer": case}
            if case == 3:
                response["content"] = "x" * 100_000
            return 200, json.dumps(response).encode()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            requests_path.write_text(
                "".join(json.dumps({"case": case}) + "\n" for case in (1, 2, 3)),
                encoding="utf-8",
            )
            (root / "ids.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            output = root / "raw.jsonl"

            with FakeInferenceServer(response_for=response_for) as server:
                result = self.run_runner(
                    server.endpoint,
                    requests_path,
                    output,
                    extra_args=("--concurrency", "2"),
                )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(server.max_active_requests, 2)
            envelopes = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(envelopes[0]["id"], "two")
            by_id = {envelope["id"]: envelope for envelope in envelopes}
            self.assertEqual(set(by_id), {"one", "two", "three"})
            self.assertEqual(len(by_id["three"]["response"]["content"]), 100_000)
            self.assertEqual(list(root.glob(".lxbench-workers-*")), [])

    def test_rejects_invalid_concurrency_before_http(self):
        for concurrency in ("0", "-1", "many"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(
                concurrency=concurrency
            ):
                root = Path(directory)
                requests_path = root / "requests.jsonl"
                requests_path.write_text("{}\n", encoding="utf-8")
                (root / "ids.txt").write_text("one\n", encoding="utf-8")
                output = root / "raw.jsonl"

                with FakeInferenceServer([]) as server:
                    result = self.run_runner(
                        server.endpoint,
                        requests_path,
                        output,
                        extra_args=("--concurrency", concurrency),
                    )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("concurrency", result.stderr.lower())
                self.assertEqual(server.requests, [])
                self.assertFalse(output.exists())

    def test_rejects_invalid_prepared_contract_before_http(self):
        invalid_contracts = {
            "blank request": ("{}\n\n{}\n", "one\ntwo\nthree\n"),
            "mismatched counts": ("{}\n{}\n", "one\n"),
            "unsafe ID": ("{}\n", "../one\n"),
            "duplicate ID": ("{}\n{}\n", "same\nsame\n"),
        }
        if self.validates_json:
            invalid_contracts.update(
                {
                    "invalid JSON request": ("not-json\n", "one\n"),
                    "non-JSON constant request": ('{"value":NaN}\n', "one\n"),
                    "non-object request": ("[]\n", "one\n"),
                }
            )

        for name, (request_text, id_text) in invalid_contracts.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                requests_path = root / "requests.jsonl"
                requests_path.write_text(request_text, encoding="utf-8")
                (root / "ids.txt").write_text(id_text, encoding="utf-8")
                output = root / "raw.jsonl"

                with FakeInferenceServer([]) as server:
                    result = self.run_runner(server.endpoint, requests_path, output)

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(server.requests, [])
                self.assertFalse(output.exists())

    def test_rejects_malformed_existing_raw_run_before_http(self):
        invalid_runs = {
            "invalid JSON": "not-json\n",
            "blank record": '{"id":"one","response":{}}\n\n',
            "non-object envelope": "[]\n",
            "wrong fields": '{"id":"one","response":{},"attempt":1}\n',
            "non-string ID": '{"id":1,"response":{}}\n',
            "non-object response": '{"id":"one","response":[]}\n',
            "duplicate ID": (
                '{"id":"one","response":{}}\n'
                '{"id":"one","response":{}}\n'
            ),
            "unknown ID": '{"id":"other","response":{}}\n',
        }
        if self.validates_json:
            invalid_runs["non-JSON response constant"] = (
                '{"id":"one","response":{"answer":NaN}}\n'
            )

        for name, raw_text in invalid_runs.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                requests_path = root / "requests.jsonl"
                requests_path.write_text("{}\n{}\n", encoding="utf-8")
                (root / "ids.txt").write_text("one\ntwo\n", encoding="utf-8")
                output = root / "raw.jsonl"
                output.write_text(raw_text, encoding="utf-8")

                with FakeInferenceServer([]) as server:
                    result = self.run_runner(server.endpoint, requests_path, output)

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(server.requests, [])
                self.assertEqual(output.read_text(encoding="utf-8"), raw_text)

    def test_resumes_only_gaps_and_new_output_runs_every_prepared_id(self):
        prepared_requests = [{"case": "one"}, {"case": "two"}, {"case": "three"}]
        existing_envelopes = [
            {"id": "one", "response": {"answer": 1}},
            {"id": "three", "response": {"answer": 3}},
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            requests_path.write_text(
                "".join(json.dumps(request) + "\n" for request in prepared_requests),
                encoding="utf-8",
            )
            (root / "ids.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            resumed_output = root / "resumed.jsonl"
            resumed_output.write_text(
                "\n".join(json.dumps(envelope) for envelope in existing_envelopes),
                encoding="utf-8",
            )

            with FakeInferenceServer([(200, b'{"answer":2}')]) as server:
                resumed = self.run_runner(
                    server.endpoint, requests_path, resumed_output
                )

            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual([item["body"] for item in server.requests], [{"case": "two"}])
            self.assertEqual(
                [
                    json.loads(line)["id"]
                    for line in resumed_output.read_text(encoding="utf-8").splitlines()
                ],
                ["one", "three", "two"],
            )

            new_output = root / "new.jsonl"
            responses = [
                (200, json.dumps({"answer": number}).encode())
                for number in (1, 2, 3)
            ]
            with FakeInferenceServer(responses) as server:
                fresh = self.run_runner(server.endpoint, requests_path, new_output)

            self.assertEqual(fresh.returncode, 0, fresh.stderr)
            self.assertEqual(
                [item["body"] for item in server.requests], prepared_requests
            )

    def test_retries_transient_failures(self):
        transient_responses = {
            "network failure": (None, b""),
            "HTTP 408": (408, b'{"error":"timeout"}'),
            "HTTP 429": (429, b'{"error":"busy"}'),
            "HTTP 500": (500, b'{"error":"server"}'),
        }
        if self.validates_json:
            transient_responses.update(
                {
                    "invalid JSON HTTP 200": (200, b"not-json"),
                    "invalid UTF-8 HTTP 200": (200, b"\xff"),
                    "non-JSON constant HTTP 200": (200, b'{"answer":NaN}'),
                }
            )

        for name, first_response in transient_responses.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                requests_path = root / "requests.jsonl"
                requests_path.write_text('{"prompt":"unchanged"}\n', encoding="utf-8")
                (root / "ids.txt").write_text("one\n", encoding="utf-8")
                output = root / "raw.jsonl"

                with FakeInferenceServer(
                    [first_response, (200, b'{"answer":"ok"}')]
                ) as server:
                    result = self.run_runner(server.endpoint, requests_path, output)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    [item["body"] for item in server.requests],
                    [{"prompt": "unchanged"}, {"prompt": "unchanged"}],
                )
                self.assertEqual(
                    json.loads(output.read_text(encoding="utf-8")),
                    {"id": "one", "response": {"answer": "ok"}},
                )

    def test_exhausted_and_nonretryable_failures_stay_absent_while_later_ids_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            cases = ["exhausted", "bad-request"]
            responses = [
                (500, b"{}"),
                (500, b"{}"),
                (500, b"{}"),
                (500, b"{}"),
                (400, b"{}"),
            ]
            expected_requests = ["exhausted"] * 4 + ["bad-request"]
            if self.validates_json:
                cases.append("non-object")
                responses.append((200, b"[]"))
                expected_requests.append("non-object")
            cases.append("later")
            responses.append((200, b'{"answer":"saved"}'))
            expected_requests.append("later")

            requests_path.write_text(
                "".join(json.dumps({"case": case}) + "\n" for case in cases),
                encoding="utf-8",
            )
            (root / "ids.txt").write_text("\n".join(cases) + "\n", encoding="utf-8")
            output = root / "raw.jsonl"

            with FakeInferenceServer(responses) as server:
                result = self.run_runner(server.endpoint, requests_path, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                [item["body"]["case"] for item in server.requests],
                expected_requests,
            )
            self.assertIn("exhausted", result.stderr)
            self.assertIn("HTTP 500", result.stderr)
            self.assertIn("bad-request", result.stderr)
            self.assertIn("HTTP 400", result.stderr)
            if self.validates_json:
                self.assertIn("non-object", result.stderr)
                self.assertIn("not a JSON object", result.stderr)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"id": "later", "response": {"answer": "saved"}},
            )

    def test_interrupt_finishes_active_requests_without_launching_more(self):
        active_started = threading.Event()
        release_active = threading.Event()
        start_lock = threading.Lock()
        started = 0
        failed_attempts = 0

        def response_for(request_body):
            nonlocal failed_attempts, started
            if request_body["case"] in (1, 2):
                with start_lock:
                    started += 1
                    if started == 2:
                        active_started.set()
                release_active.wait(timeout=5)
            case = request_body["case"]
            if case == 2:
                with start_lock:
                    failed_attempts += 1
                    if failed_attempts == 1:
                        return 500, b'{}'
            return 200, json.dumps({"answer": case}).encode()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            requests_path.write_text(
                "".join(json.dumps({"case": case}) + "\n" for case in (1, 2, 3)),
                encoding="utf-8",
            )
            (root / "ids.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            output = root / "raw.jsonl"

            with FakeInferenceServer(response_for=response_for) as server:
                process = subprocess.Popen(
                    [
                        *self.runner_command,
                        "--endpoint",
                        server.endpoint,
                        "--requests",
                        str(requests_path),
                        "--output",
                        str(output),
                        "--concurrency",
                        "2",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertTrue(active_started.wait(timeout=5))
                process.send_signal(signal.SIGINT)
                time.sleep(0.1)
                release_active.set()
                process.communicate(timeout=5)

            self.assertEqual(process.returncode, 130)
            self.assertCountEqual(
                [request["body"]["case"] for request in server.requests], [1, 2]
            )
            self.assertEqual(
                {
                    json.loads(line)["id"]
                    for line in output.read_text(encoding="utf-8").splitlines()
                },
                {"one"},
            )

    def test_requires_endpoint_requests_and_output_arguments(self):
        result = subprocess.run(
            self.runner_command, capture_output=True, text=True
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--endpoint", result.stderr)
        self.assertIn("--requests", result.stderr)
        self.assertIn("--output", result.stderr)


class PythonRunnerTests(RunnerContract, unittest.TestCase):
    runner_command = (sys.executable, str(PYTHON_RUNNER))
    validates_json = True


class BashRunnerTests(RunnerContract, unittest.TestCase):
    runner_command = ("/bin/bash", str(BASH_RUNNER))

    def run_runner(self, endpoint, requests_path, output, extra_args=()):
        with tempfile.TemporaryDirectory() as directory:
            command_path = Path(directory)
            for command in ("curl", "mkdir", "rm", "sleep"):
                executable = shutil.which(command)
                self.assertIsNotNone(executable)
                (command_path / command).symlink_to(executable)
            environment = os.environ.copy()
            environment["PATH"] = str(command_path)
            return super().run_runner(
                endpoint,
                requests_path,
                output,
                environment=environment,
                extra_args=extra_args,
            )


if __name__ == "__main__":
    unittest.main()
