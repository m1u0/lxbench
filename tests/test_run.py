import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run.py"


class FakeInferenceServer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                owner.requests.append(
                    {
                        "body": json.loads(self.rfile.read(length)),
                        "content_type": self.headers["Content-Type"],
                    }
                )
                status, body = owner.responses.pop(0)
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


class RunnerTests(unittest.TestCase):
    def run_runner(self, endpoint, requests_path, output):
        return subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--endpoint",
                endpoint,
                "--requests",
                str(requests_path),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
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

    def test_rejects_invalid_prepared_contract_before_http(self):
        invalid_contracts = {
            "blank request": ("{}\n\n{}\n", "one\ntwo\nthree\n"),
            "invalid JSON request": ("not-json\n", "one\n"),
            "non-JSON constant request": ('{"value":NaN}\n', "one\n"),
            "non-object request": ("[]\n", "one\n"),
            "mismatched counts": ("{}\n{}\n", "one\n"),
            "unsafe ID": ("{}\n", "../one\n"),
            "duplicate ID": ("{}\n{}\n", "same\nsame\n"),
        }

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
            "non-JSON response constant": (
                '{"id":"one","response":{"answer":NaN}}\n'
            ),
            "duplicate ID": (
                '{"id":"one","response":{}}\n'
                '{"id":"one","response":{}}\n'
            ),
            "unknown ID": '{"id":"other","response":{}}\n',
        }

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

    def test_retries_transient_failures_and_invalid_success_json(self):
        transient_responses = {
            "network failure": (None, b""),
            "HTTP 408": (408, b'{"error":"timeout"}'),
            "HTTP 429": (429, b'{"error":"busy"}'),
            "HTTP 500": (500, b'{"error":"server"}'),
            "invalid JSON HTTP 200": (200, b"not-json"),
            "invalid UTF-8 HTTP 200": (200, b"\xff"),
            "non-JSON constant HTTP 200": (200, b'{"answer":NaN}'),
        }

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
            requests_path.write_text(
                '{"case":"exhausted"}\n'
                '{"case":"bad-request"}\n'
                '{"case":"non-object"}\n'
                '{"case":"later"}\n',
                encoding="utf-8",
            )
            (root / "ids.txt").write_text(
                "exhausted\nbad-request\nnon-object\nlater\n", encoding="utf-8"
            )
            output = root / "raw.jsonl"
            responses = [
                (500, b"{}"),
                (500, b"{}"),
                (500, b"{}"),
                (500, b"{}"),
                (400, b"{}"),
                (200, b"[]"),
                (200, b'{"answer":"saved"}'),
            ]

            with FakeInferenceServer(responses) as server:
                result = self.run_runner(server.endpoint, requests_path, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                [item["body"]["case"] for item in server.requests],
                [
                    "exhausted",
                    "exhausted",
                    "exhausted",
                    "exhausted",
                    "bad-request",
                    "non-object",
                    "later",
                ],
            )
            self.assertIn("exhausted", result.stderr)
            self.assertIn("HTTP 500", result.stderr)
            self.assertIn("bad-request", result.stderr)
            self.assertIn("HTTP 400", result.stderr)
            self.assertIn("non-object", result.stderr)
            self.assertIn("not a JSON object", result.stderr)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"id": "later", "response": {"answer": "saved"}},
            )

    def test_interrupt_after_next_request_starts_preserves_prior_envelope(self):
        second_request_started = threading.Event()
        release_response = threading.Event()

        def blocked_response():
            second_request_started.set()
            release_response.wait(timeout=5)
            return b'{"answer":2}'

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            requests_path.write_text('{"case":1}\n{"case":2}\n', encoding="utf-8")
            (root / "ids.txt").write_text("one\ntwo\n", encoding="utf-8")
            output = root / "raw.jsonl"

            with FakeInferenceServer(
                [(200, b'{"answer":1}'), (200, blocked_response)]
            ) as server:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(RUNNER),
                        "--endpoint",
                        server.endpoint,
                        "--requests",
                        str(requests_path),
                        "--output",
                        str(output),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertTrue(second_request_started.wait(timeout=5))
                process.terminate()
                process.communicate(timeout=5)
                release_response.set()

            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(
                [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()],
                [{"id": "one", "response": {"answer": 1}}],
            )

    def test_requires_endpoint_requests_and_output_arguments(self):
        result = subprocess.run(
            [sys.executable, str(RUNNER)], capture_output=True, text=True
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--endpoint", result.stderr)
        self.assertIn("--requests", result.stderr)
        self.assertIn("--output", result.stderr)


if __name__ == "__main__":
    unittest.main()
