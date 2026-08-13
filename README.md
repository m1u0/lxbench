# lxbench

`lxbench` executes prepared benchmark requests against an already-running inference
endpoint. Shared execution is benchmark-agnostic: a preparer owns request content,
and the runner treats every request as an opaque JSON object.

## Prepared-to-raw workflow

A prepared benchmark directory must contain these aligned UTF-8 files:

- `requests.jsonl`: one complete JSON request object per line.
- `ids.txt`: one unique stable ID per request line. IDs may contain letters,
  numbers, `.`, `_`, `:`, and `-`, and must begin with a letter or number.
- `cases.jsonl`: one grading case with a top-level string `id` per request line.

Run the prepared requests with Python 3.10 or newer:

```sh
python3 run.py \
  --endpoint http://board.example:8080/v1/chat/completions \
  --requests prepared/example/requests.jsonl \
  --output results/example/raw.jsonl \
  --concurrency 4
```

For a faster, reproducible sampled run, add a positive sample size and optional
nonnegative seed. The seed defaults to `0`:

```sh
python3 run.py \
  --endpoint http://board.example:8080/v1/chat/completions \
  --requests prepared/example/requests.jsonl \
  --output results/example/raw.jsonl \
  --sample-size 50 \
  --seed 7
```

Python and Bash select the same IDs for a given prepared benchmark, size, and
seed. If fewer than the requested number of cases exist, every case is selected.

### Board-local Bash fallback

When the workstation cannot reach the board's inference endpoint, copy the Bash
runner and both aligned prepared files to the board:

```sh
ssh board 'mkdir -p /tmp/lxbench'
scp run.sh prepared/example/requests.jsonl prepared/example/ids.txt board:/tmp/lxbench/
```

On the board, call the complete local endpoint URL and keep the raw run at a
stable path so rerunning the same command resumes missing IDs:

```sh
cd /tmp/lxbench
./run.sh \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --requests requests.jsonl \
  --output raw.jsonl \
  --concurrency 4
```

Retain `raw.jsonl` after the run, then transfer it back to the workstation for
loading and grading:

```sh
scp board:/tmp/lxbench/raw.jsonl results/example/raw.jsonl
```

For a sampled run, also retain and transfer the sibling
`raw.jsonl.manifest.json`. The grader discovers it from the raw response path;
without it, omitted cases would belong to a full run rather than the sample.

The fallback requires Bash, `curl`, `mkdir`, `rm`, and `sleep`; it does not
require a model file or JSON tooling. Because it has no JSON parser, it embeds
compact HTTP 2xx response bodies directly. Workstation loading or grading
performs full JSON response validation.

The runner discovers `ids.txt` beside `requests.jsonl` and creates the output
parent directory when necessary. It sends JSON `POST`s with at most
`--concurrency N` requests active. `N` must be positive and defaults to `1`.
Every successful JSON-object response is immediately appended and flushed in
completion order as:

```json
{"id": "case-1", "response": {"choices": []}}
```

The raw run is append-only. Reusing an output path validates the existing run,
skips completed IDs, and executes only missing IDs. Choose a different output path
to start an independent run.

Each settled request writes one stable progress line to standard error, for
example `[12/50] PASS case-12` or `[13/50] FAIL case-13: HTTP 500`. A sampled
run writes `<output>.manifest.json` before inference with the population size,
sample size, seed, and selected IDs. Resume requires the same manifest; use a
different output path to change the sample.

Each benchmark grader automatically uses a valid sibling manifest as its
denominator. Its saved and printed result identifies the score as sampled,
including the selected count, full population, and seed. Selected requests that
never produced a response remain missing and incorrect.

Network failures, HTTP 408, HTTP 429, and HTTP 5xx are retried up to four total
attempts, with waits of one, two, and four seconds. The Python runner also retries
malformed HTTP-2xx JSON. Other HTTP 4xx responses are not retried. A failed ID
stays absent, later IDs continue, and the command exits nonzero if any ID fails.
On Ctrl-C, the runner stops launching requests, finishes and persists active
requests, and exits interrupted.

## Execution boundary

The runner does not know benchmark semantics and does not transform requests. It
does not select or load a model, accept GGUF arguments, manage the inference
server, probe health, add authentication, impose a request timeout, grade results,
or choose a concurrency limit automatically. Dataset preparation, request
profiles, inference server operation, and grading remain separate responsibilities.

## Benchmarks

- [MMLU-Redux 2.0](benchmarks/mmlu_redux/README.md): prepare the official
  provider dataset, run it through this shared runner, and grade overall and
  per-subject accuracy.
- [IFEval](benchmarks/ifeval/README.md): prepare the official instruction-following
  dataset and grade it with the provider evaluator.
- [LongBench v2](benchmarks/longbench_v2/README.md): prepare the official
  zero-shot direct profile with provider-style middle truncation and grading.

## Prepare all datasets

After installing each benchmark's workstation dependencies, prepare every dataset
up front with the effective context size of the target server:

```sh
python3 prepare.py --longbench-context-size 262144
```

Each preparer validates its existing published output and skips dataset loading
when all aligned files are already complete. Pass `--force` to this command or an
individual preparer to regenerate its output. In particular, use `--force` after
changing the LongBench context size.
