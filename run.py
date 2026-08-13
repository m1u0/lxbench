#!/usr/bin/env python3

import argparse
import http.client
import json
import queue
import re
import sys
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
RETRY_WAITS = (1, 2, 4)


class RequestFailure(ValueError):
    pass


class InvalidJSON(ValueError):
    pass


def reject_json_constant(value):
    raise InvalidJSON(f"invalid JSON constant {value}")


def decode_json(value):
    try:
        return json.loads(value, parse_constant=reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidJSON from error


def split_jsonl(text):
    lines = text.split("\n")
    if lines[-1:] == [""]:
        lines.pop()
    return lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute prepared benchmark requests against an inference endpoint."
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be a positive integer")
    return args


def load_prepared(requests_path):
    request_lines = split_jsonl(requests_path.read_text(encoding="utf-8"))
    if any(not line.strip() for line in request_lines):
        raise ValueError("requests file contains a blank record")

    requests = []
    for line_number, line in enumerate(request_lines, start=1):
        try:
            request_body = decode_json(line)
        except InvalidJSON as error:
            raise ValueError(
                f"request line {line_number} is not valid JSON"
            ) from error
        if not isinstance(request_body, dict):
            raise ValueError(f"request line {line_number} is not a JSON object")
        requests.append(request_body)

    ids = (requests_path.parent / "ids.txt").read_text(encoding="utf-8").splitlines()
    if len(requests) != len(ids):
        raise ValueError("request and ID counts do not match")
    if any(not SAFE_ID.fullmatch(prepared_id) for prepared_id in ids):
        raise ValueError("prepared IDs must be nonblank safe IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("prepared IDs must be unique")

    return requests, ids


def load_completed(output_path, prepared_ids):
    if not output_path.exists():
        return set(), False

    raw_text = output_path.read_text(encoding="utf-8")
    lines = split_jsonl(raw_text)
    if any(not line.strip() for line in lines):
        raise ValueError("raw run contains a blank record")

    completed = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            envelope = decode_json(line)
        except InvalidJSON as error:
            raise ValueError(
                f"raw run line {line_number} is not valid JSON"
            ) from error
        if not isinstance(envelope, dict) or set(envelope) != {"id", "response"}:
            raise ValueError(f"raw run line {line_number} is not a valid envelope")

        completed_id = envelope["id"]
        if not isinstance(completed_id, str):
            raise ValueError(f"raw run line {line_number} has a non-string ID")
        if not isinstance(envelope["response"], dict):
            raise ValueError(f"raw run line {line_number} has a non-object response")
        if completed_id not in prepared_ids:
            raise ValueError(f"raw run contains unknown ID {completed_id!r}")
        if completed_id in completed:
            raise ValueError(f"raw run contains duplicate ID {completed_id!r}")
        completed.add(completed_id)

    return completed, bool(raw_text and not raw_text.endswith("\n"))


def post_with_retries(endpoint, request_body, stop_requested):
    request_data = json.dumps(request_body).encode("utf-8")

    for attempt in range(len(RETRY_WAITS) + 1):
        if stop_requested.is_set():
            raise RequestFailure("interrupted")
        request = Request(
            endpoint,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        retryable = False
        try:
            with urlopen(request) as response:
                response_body = decode_json(response.read())
            if isinstance(response_body, dict):
                return response_body
            reason = "HTTP 2xx response was not a JSON object"
        except InvalidJSON:
            reason = "HTTP 2xx response was not valid JSON"
            retryable = True
        except HTTPError as error:
            reason = f"HTTP {error.code}"
            retryable = error.code in (408, 429) or 500 <= error.code <= 599
        except (URLError, OSError, http.client.HTTPException) as error:
            reason = f"network failure: {error}"
            retryable = True

        if retryable and attempt < len(RETRY_WAITS):
            if stop_requested.wait(RETRY_WAITS[attempt]):
                raise RequestFailure("interrupted")
            continue
        raise RequestFailure(reason)


def run(args):
    requests, ids = load_prepared(args.requests)
    completed, needs_separator = load_completed(args.output, set(ids))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    interrupted = False
    pending = (
        (prepared_id, request_body)
        for prepared_id, request_body in zip(ids, requests)
        if prepared_id not in completed
    )
    finished = queue.Queue()
    in_flight = {}
    stop_requested = threading.Event()

    with args.output.open("a", encoding="utf-8") as output:
        def submit_next(executor):
            try:
                prepared_id, request_body = next(pending)
            except StopIteration:
                return False
            future = executor.submit(
                post_with_retries, args.endpoint, request_body, stop_requested
            )
            in_flight[future] = prepared_id
            future.add_done_callback(finished.put)
            return True

        def record(future):
            nonlocal failed, needs_separator
            prepared_id = in_flight.pop(future)
            try:
                response_body = future.result()
            except CancelledError:
                return
            except RequestFailure as error:
                print(f"{prepared_id}: {error}", file=sys.stderr)
                failed = True
                return
            if needs_separator:
                output.write("\n")
                needs_separator = False
            output.write(
                json.dumps({"id": prepared_id, "response": response_body}) + "\n"
            )
            output.flush()

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            try:
                for _ in range(args.concurrency):
                    if not submit_next(executor):
                        break
                while in_flight:
                    record(finished.get())
                    submit_next(executor)
            except KeyboardInterrupt:
                interrupted = True
                stop_requested.set()
                for future in in_flight:
                    future.cancel()
                while in_flight:
                    record(finished.get())

    if interrupted:
        return 130
    return 1 if failed else 0


def main():
    args = parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
