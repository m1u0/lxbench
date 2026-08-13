# LXBench

LXBench is a minimal, modular harness for running deterministic, code-graded LLM
benchmarks against an already-running inference endpoint. Each benchmark owns its
preparation and grading, while a small benchmark-agnostic execution layer runs
all of them.

The Python runner uses only the standard library. When Python is not available, a
Bash fallback needs only `curl` and a handful of basic shell utilities, with no
JSON tooling or benchmark dependencies.

## Design

Benchmark-specific code lives in self-contained directories under `benchmarks/`.
Each directory holds its preparer, grader, dependencies, documentation, tests,
and any provider files it needs. The shared boundary is a small file contract,
not a plugin system or benchmark framework.

LXBench is provider-first: grading follows the benchmark provider where possible.
Each benchmark README records its sources, request profile, and any necessary
deviations. Adding a benchmark should not require changes to the shared runners.

## Workflow

1. A benchmark-specific preparer downloads and caches the official dataset, then
   writes requests, stable IDs, and grading cases.
2. The shared runner sends those requests unchanged to an OpenAI-compatible
   chat-completions endpoint and saves the raw responses.
3. A benchmark-specific grader joins responses to cases and reports the standard
   metrics for that benchmark.

Prepared data can be run more than once. Raw responses are kept separately from
scores, so they can be graded again without repeating inference.

## Benchmarks

| Benchmark | Evaluates | Reports |
| --- | --- | --- |
| [MMLU-Redux 2.0](benchmarks/mmlu_redux/README.md) | Multiple-choice knowledge and reasoning | Overall and per-subject accuracy |
| [IFEval](benchmarks/ifeval/README.md) | Instruction following | Strict and loose prompt-level and instruction-level accuracy |
| [LongBench v2](benchmarks/longbench_v2/README.md) | Long-context understanding | Overall accuracy with difficulty and context-length breakdowns |

See each benchmark's README for its dependencies, provider sources, request
profile, grading details, and deviations from the provider's inference setup.

## Quick start: MMLU-Redux

The workstation workflow requires Python 3.10 or newer and an already-running
inference endpoint. Install the MMLU-Redux dependency:

```sh
python3 -m pip install -r benchmarks/mmlu_redux/requirements.txt
```

Prepare the dataset:

```sh
python3 benchmarks/mmlu_redux/prepare.py \
  --cache-dir data/mmlu-redux-2.0 \
  --output prepared/mmlu-redux-2.0
```

Run it against the endpoint:

```sh
python3 run.py \
  --endpoint http://board.example:8080/v1/chat/completions \
  --requests prepared/mmlu-redux-2.0/requests.jsonl \
  --output results/mmlu-redux-2.0/raw.jsonl \
  --concurrency 4
```

Grade the saved responses:

```sh
python3 benchmarks/mmlu_redux/grade.py \
  --cases prepared/mmlu-redux-2.0/cases.jsonl \
  --responses results/mmlu-redux-2.0/raw.jsonl \
  --output results/mmlu-redux-2.0/score.json
```

The grader prints the result and writes the full JSON summary to the output path.

## Preparing all benchmarks

After installing the dependencies described in each benchmark README, prepare
all current datasets with:

```sh
python3 prepare.py --longbench-context-size 262144
```

The LongBench context size must match the effective context configured for the
target server. Existing valid output is reused. Pass `--force` to regenerate it,
including after changing the LongBench context size.

## Running and resuming

Both runners support concurrency, reproducible sampling, transient-failure
retries, and resume by stable case ID. Successful responses are written
immediately. Reusing an output path skips completed IDs and retries missing ones;
using a new path starts an independent run.

Add `--sample-size N` and, optionally, `--seed N` to run a stable subset. Sampled
runs write `<output>.manifest.json`. Keep this file beside the raw responses so
the grader can recover the selected denominator, including requests that did not
produce a response.

## Running on a stripped-down system

If the workstation cannot reach the endpoint, copy `run.sh` and the prepared
`requests.jsonl` and `ids.txt` files to the target, then run:

```sh
./run.sh \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --requests requests.jsonl \
  --output raw.jsonl \
  --concurrency 4
```

Copy `raw.jsonl` back to the workstation for grading, along with its manifest for
a sampled run. The fallback requires Bash, `curl`, `mkdir`, `rm`, and `sleep`. It
has no JSON parser, so HTTP 2xx response bodies must be compact JSON on a single
line. The workstation grader performs full response validation.

## Adding a benchmark

Every preparer publishes three aligned UTF-8 files:

| File | Purpose |
| --- | --- |
| `requests.jsonl` | Complete JSON request bodies sent unchanged by the runner |
| `ids.txt` | One stable ID per request, aligned by line |
| `cases.jsonl` | Benchmark-specific information used for grading |

The runner reads only requests and IDs. The grader joins cases to raw responses
by ID. A new benchmark supplies a preparer, grader, dependency list, README, and
focused tests under `benchmarks/`. Provider code, when needed, belongs in a
minimal `upstream/` directory.

The contract covers deterministic benchmarks whose requests can run
independently. Interactive or response-dependent benchmarks need a different
execution model.

## Scope

LXBench evaluates an already-running endpoint. It does not download or load
models, manage the inference server, add authentication, compare endpoints,
aggregate benchmark scores, judge regressions, or provide dedicated latency
measurement.
