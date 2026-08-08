# lxbench

`lxbench` executes prepared benchmark requests against an already-running inference
endpoint. Shared execution is benchmark-agnostic: a preparer owns request content,
and the runner treats every request as an opaque JSON object.

## Prepared-to-raw workflow

A prepared benchmark directory must contain these aligned UTF-8 files:

- `requests.jsonl`: one complete JSON request object per line.
- `ids.txt`: one unique stable ID per request line. IDs may contain letters,
  numbers, `.`, `_`, `:`, and `-`, and must begin with a letter or number.

Run the prepared requests with Python 3.10 or newer:

```sh
python3 run.py \
  --endpoint http://board.example:8080/v1/chat/completions \
  --requests prepared/example/requests.jsonl \
  --output results/example/raw.jsonl
```

The runner discovers `ids.txt` beside `requests.jsonl` and creates the output
parent directory when necessary. It sends requests sequentially as JSON `POST`s.
Every successful JSON-object response is immediately appended and flushed as:

```json
{"id": "case-1", "response": {"choices": []}}
```

The raw run is append-only. Reusing an output path validates the existing run,
skips completed IDs, and executes only missing IDs. Choose a different output path
to start an independent run.

Network failures, HTTP 408, HTTP 429, HTTP 5xx, and malformed HTTP-2xx JSON are
retried up to four total attempts, with waits of one, two, and four seconds. Other
HTTP 4xx responses are not retried. A failed ID stays absent, later IDs continue,
and the command exits nonzero if any ID fails.

## Execution boundary

The runner does not know benchmark semantics and does not transform requests. It
does not select or load a model, accept GGUF arguments, manage the inference
server, probe health, add authentication, impose a request timeout, grade results,
or run requests in parallel. Dataset preparation, request profiles, inference
server operation, and grading remain separate responsibilities.
