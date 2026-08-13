# Minimal Modular LLM Benchmark Harness

Status: ready for implementation  
Initial target: `unsloth/Qwen3.6-35B-A3B-GGUF`, BF16  
Inference service: an already-running `llama-server`

## Problem Statement

The project needs a small harness that can run deterministic, code-graded LLM benchmarks against a proprietary RISC-V system and report benchmark scores. The RISC-V board exposes `llama-server`; the harness must not manage the server or require access to its GGUF file.

The normal execution path runs on a development workstation and calls the board over Ethernet. A Bash fallback must also be able to run directly on the board when remote access from the workstation is unavailable. In both modes, completed responses must survive interruption and support ID-based resume.

The central design risk is duplication: benchmark-specific prompts or grading must not leak into shared execution, and each benchmark must not acquire its own runner. The harness therefore needs one small request/result contract that keeps execution generic while leaving preparation and grading with the benchmark that owns them.

## Solution

Build a file-oriented workflow with four phases:

1. A benchmark-specific Python preparer downloads and caches official Hugging Face data, then emits opaque HTTP requests and grading cases.
2. One shared Python runner sends those requests to a required endpoint. A shared Bash runner provides the board-local fallback.
3. Each successful response is appended immediately to a raw JSONL run identified by stable case ID. Reusing the output path resumes missing IDs; choosing a new output path starts a new run.
4. A benchmark-specific Python grader joins the raw run to prepared cases, invokes provider grading code wherever it exists, and reports the benchmark's standard metrics.

The initial suite is MMLU-Redux 2.0, IFEval, and LongBench v2. MMLU-Pro is deferred, but a future deterministic profile must be addable without changing shared execution.

### Design language

These terms are normative throughout the spec:

- **Provider-first**: preserve the benchmark creator's data, prompts, conventions, and grading logic wherever they apply to the selected profile.
- **Opaque request**: a complete `llama-server` JSON request body whose benchmark meaning is unknown to the runner.
- **Prepared benchmark**: the three aligned files `requests.jsonl`, `ids.txt`, and `cases.jsonl`.
- **Raw run**: an append-only JSONL file of successful `{id, response}` envelopes.
- **ID resume**: treat an ID already present in the raw run as complete and execute only absent IDs.

## User Stories

1. As a benchmark operator, I want to download official benchmark data from Hugging Face, so that I do not assemble datasets manually.
2. As a benchmark operator, I want downloaded data cached in the working repository, so that repeated preparation does not redownload it.
3. As a benchmark operator, I want preparation separated from inference, so that the same prepared benchmark can be run more than once.
4. As a benchmark operator, I want to provide one complete endpoint URL, so that the runner does not infer network or server configuration.
5. As a benchmark operator, I want to choose the raw output path, so that a new path creates a new run and an existing path resumes one.
6. As a benchmark operator, I want every successful response flushed immediately, so that a stopped run retains completed work.
7. As a benchmark operator, I want transient failures retried, so that brief network or server problems do not lose a case.
8. As a benchmark operator, I want exhausted failures left absent from the raw run, so that a later invocation retries them naturally.
9. As a benchmark operator, I want completed cases recognized by stable ID rather than line count, so that resume remains correct when earlier failures create gaps.
10. As a workstation user, I want a standard-library Python runner, so that shared inference does not require an HTTP package or framework.
11. As a board user, I want a Bash-and-curl fallback, so that inference can run locally without Python or JSON tooling.
12. As a benchmark operator, I want raw server responses retained, so that I can regrade without rerunning inference.
13. As a benchmark operator, I want creator-supplied grading code used whenever possible, so that scores follow benchmark intent.
14. As a benchmark operator, I want each grader to report its standard metrics and breakdowns, so that benchmark-specific information is not collapsed into one generic score.
15. As a benchmark operator, I want deterministic, no-thinking request profiles, so that scores are stable internal regression signals.
16. As a LongBench operator, I want provider-style middle truncation against an explicit server context size, so that oversized cases exercise a defined policy.
17. As a maintainer, I want to understand a benchmark by reading its directory, so that prompts, dependencies, provider code, and grading are locally discoverable.
18. As a maintainer, I want to add a benchmark through preparation and grading only, so that the shared runners remain unchanged.
19. As a maintainer, I want tiny behavior-level tests, so that retry, resume, contracts, and grader wiring are protected without model or hardware fixtures.
20. As a server operator, I want the harness independent of model paths and GGUF files, so that server provisioning remains a separate responsibility.

## Implementation Decisions

### 1. Responsibility boundary

The workstation owns dataset download, preparation, normal remote execution, raw-result storage, grading, and regrading. The board owns only the independently managed `llama-server` process.

The Bash fallback is copied to the board with the prepared requests and IDs. It calls a local endpoint, writes a raw run, and leaves transfer back to the workstation to the operator. Grading never runs on the board.

Neither runner starts or stops `llama-server`, downloads a model, accepts a GGUF path, probes hardware, compares endpoints, or decides whether a score is a regression.

### 2. Module boundary and repository shape

Shared infrastructure consists of exactly:

- A root Python runner.
- A root Bash fallback runner.
- Root documentation and ignore rules.
- Black-box runner tests and a fake HTTP server.

Each benchmark has one plainly named directory under `benchmarks/` containing:

- `README.md`: provider source, selected profile, commands, dependencies, and deviations.
- `prepare.py`: dataset download/cache, prompt construction, and prepared-contract validation.
- `grade.py`: result loading, provider adapter, metrics, and score-summary output.
- `requirements.txt`: only that benchmark's workstation dependencies.
- `upstream/`, only when required: the smallest usable set of provider grading or prompt files, plus license and source/revision notes.

Generated Hugging Face caches, prepared benchmarks, and raw/graded results live in root working directories named `data/`, `prepared/`, and `results/`; all three are ignored by Git.

There is no benchmark registry, base class, plugin loader, shared prompt builder, dataset framework, or top-level orchestration command. The file contract is the abstraction.

### 3. Prepared benchmark contract

Every preparer emits these aligned files:

| File | Shared meaning | Benchmark-owned content |
|---|---|---|
| `requests.jsonl` | One complete opaque request object per line | Messages, prompt, output cap, and generation settings |
| `ids.txt` | One unique safe ID per request line | Stable provider ID or deterministic derived ID |
| `cases.jsonl` | One grading case with top-level `id` per line | Answer keys, constraints, categories, and breakdown fields |

Contract rules:

- All files are UTF-8 and contain no blank records.
- `requests.jsonl` contains valid JSON objects with no harness wrapper or harness ID.
- Line N of `ids.txt` identifies line N of `requests.jsonl`.
- IDs are unique and match `^[A-Za-z0-9][A-Za-z0-9._:-]*$`.
- Every `cases.jsonl` object contains a string `id`.
- The request count, ID count, and case count match.
- The `cases.jsonl` ID set exactly matches `ids.txt`.
- A preparer returns success only after validating the complete contract.
- Preparation publishes new files only after the entire output validates; a failed preparation must not replace an earlier valid prepared benchmark.

The shared runners read `requests.jsonl` and its sibling `ids.txt`. They never read `cases.jsonl`.

### 4. Dataset cache

Preparers use the official Hugging Face dataset repositories without an explicit revision. The latest revision available on the first download is stored under `data/<benchmark>/`; subsequent runs reuse the cache rather than deliberately redownloading it.

Model weights and GGUF files are outside this cache. LongBench preparation downloads only the Qwen tokenizer files needed for token counting and truncation, never model weights.

### 5. Shared runner interface

Both runners expose the same arguments:

| Argument | Meaning |
|---|---|
| `--endpoint` | Complete inference URL, normally ending in `/v1/chat/completions` |
| `--requests` | Path to prepared `requests.jsonl` |
| `--output` | Path to the raw run JSONL |
| `--concurrency` | Maximum active requests; optional positive integer, default `1` |

`ids.txt` is resolved beside `requests.jsonl`; there is no ID argument. The output's parent directory is created when absent.

Execution is bounded by `--concurrency`. A runner launches missing requests in prepared order as capacity becomes available and sends each opaque request as an HTTP `POST` with `Content-Type: application/json`. No health check, authentication feature, fixed request timeout, model parameter, or request transformation is part of shared execution.

The Python runner supports Python 3.10 or newer and uses only the standard library, including `argparse`, `concurrent.futures`, `json`, `pathlib`, `queue`, `time`, and `urllib`. It avoids language features newer than Python 3.10.

The Bash runner uses Bash, curl, `mkdir`, `rm`, and `sleep`. It does not require Python, Node.js, `jq`, `awk`, or `sed`. Safe IDs and the fixed output-envelope shape allow it to recover completed IDs without interpreting request or response semantics.

Benchmark workstation dependencies stay local to their directories:

| Benchmark | Expected dependency surface |
|---|---|
| MMLU-Redux | Hugging Face `datasets` |
| IFEval | `datasets` plus dependencies required by the official evaluator, including `absl-py`, `langdetect`, `nltk`, and `immutabledict` |
| LongBench v2 | `datasets`, `transformers`, and its tokenizer dependencies |

### 6. Raw run contract

Each output line has exactly two top-level fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable ID from `ids.txt` |
| `response` | JSON object | Successful `llama-server` response |

The runner does not add normalized text, model, usage, timing, hardware, grading, attempt, or error fields. Server-provided timing and usage remain inside `response` when available.

The Python runner preserves response meaning rather than response byte layout: JSON whitespace and key ordering are not contractual. The Bash runner embeds the compact response body directly.

Only a complete successful response is appended. Each append is flushed when its request completes; concurrent responses therefore appear in completion order. Duplicate output IDs are invalid.

### 7. Retry, failure, and ID resume

At startup, a runner validates prepared IDs and loads every valid ID already present in the raw run. It rejects duplicate output IDs or output IDs that are not in the current prepared benchmark. It skips completed IDs and launches absent IDs in prepared order while never exceeding the configured concurrency.

Retry up to three times after the initial attempt, giving four maximum attempts per invocation. Both runners retry:

- Network or transport failure.
- HTTP 408.
- HTTP 429.
- HTTP 5xx.

The Python runner also retries HTTP 2xx responses whose bodies are not valid JSON. The Bash fallback has no JSON parser and treats a compact HTTP 2xx body from `llama-server` as the response object; full validation happens when the workstation loads or grades the raw run.

Use fixed waits of one, two, and four seconds before successive retries. Other HTTP 4xx responses are not retried.

When all attempts fail, write the ID and concise reason to standard error, leave the ID absent from the raw run, and continue. Exit nonzero after processing the remaining IDs if any ID exhausted its attempts. A later invocation with the same output path naturally retries every absent ID.

On Ctrl-C, a runner stops launching requests, lets active requests finish, appends their successful responses, and exits interrupted. Previously appended lines remain a valid, regradable raw run. A different output path always represents a new independent run.

### 8. Shared request profile

Every initial request contains these profile fields:

| Field | Value |
|---|---|
| `temperature` | `0` |
| `stream` | `false` |
| `chat_template_kwargs.enable_thinking` | `false` |

Requests omit explicit `top_p`, `seed`, `model`, and GGUF fields.

This is the named `qwen36-deterministic-no-thinking` profile. It is an internal regression profile, not automatically a leaderboard-comparable provider profile. Each benchmark README names any deviation from its provider's inference settings.

### 9. Benchmark profiles

#### MMLU-Redux 2.0

- Dataset: `edinburgh-dawg/mmlu-redux-2.0`.
- Select only rows whose latest annotation is `ok`.
- Preserve subject information.
- Create one zero-shot user message containing the question and choices A-D, requesting only the answer letter.
- Use `max_tokens: 16` and the shared request profile.
- Extract one unambiguous A-D prediction; missing, malformed, or ambiguous predictions are incorrect.
- Grade against the original answer of the selected valid row.
- Report overall accuracy, per-subject accuracy, and correct/incorrect/invalid/missing/total counts.

The A-D extractor trims the response text, finds case-insensitive standalone `A`, `B`, `C`, or `D` tokens, and accepts the prediction only when exactly one token occurrence is present. The provider project does not supply a grader for this selected direct-answer profile; this extractor is the documented exception to the provider-first grading rule.

#### IFEval

- Dataset: `google/IFEval`.
- Preserve provider key, prompt, instruction IDs, and instruction arguments required by the official evaluator.
- Send the provider prompt unchanged as one user message.
- Use `max_tokens: 8192` and the shared request profile.
- Convert raw envelopes into the provider evaluator's expected prompt/response shape without altering response text.
- Invoke provider-supplied grading logic; project code owns only file adaptation and summary formatting.
- Report strict and loose prompt-level accuracy, strict and loose instruction-level accuracy, and expected/received/missing/invalid counts.

#### LongBench v2

- Dataset: `zai-org/LongBench-v2`.
- Use the provider's official zero-shot direct prompt.
- Require `--context-size` during preparation; this is the effective context configured for the target server.
- Use the Qwen3.6 tokenizer from Hugging Face for token counting. Reserve 128 generation tokens and account for chat-template overhead.
- When the rendered prompt exceeds the input budget, apply provider-style middle truncation by retaining the first and last token segments that fit.
- Store truncation status, category, difficulty, and provider length fields in the grading case.
- Use `max_tokens: 128` and the shared request profile.
- Invoke the provider's direct-answer extractor and metric logic, adapting only the shared file shapes around them.
- Count missing, malformed, or unextractable predictions as incorrect.
- Report overall accuracy and provider-standard difficulty and context-length breakdowns, plus correct/incorrect/invalid/missing/total counts.

The benchmark README explicitly records that the harness changes the provider direct runner's temperature from `0.1` to `0`.

### 10. Provider-first grading policy

Provider code is authoritative when it implements the selected profile:

1. Wrap or invoke provider grading rather than rewriting it.
2. If the code is not installable, copy only required files into that benchmark's `upstream/` directory.
3. Preserve its license and record source URL and copied revision.
4. Keep its dependencies in that benchmark's `requirements.txt`.
5. Limit project adapters to translating prepared cases and raw envelopes into provider shapes and formatting provider metrics.
6. Document every necessary semantic deviation in the benchmark README.

Runtime downloads of executable grading code and Git submodules are outside the design.

### 11. Grading interface

Every grader requires:

| Argument | Meaning |
|---|---|
| `--cases` | Prepared benchmark-specific `cases.jsonl` |
| `--responses` | Existing raw run JSONL |
| `--output` | JSON score-summary path |

The grader joins by ID, rejects duplicate or unknown response IDs, and treats expected IDs absent from the raw run as missing and incorrect in the headline denominator. For the initial OpenAI-style chat endpoint, model text is the string at `response.choices[0].message.content`; a missing or non-string value is invalid. Structurally invalid model answers are incorrect.

The grader prints the headline metric and concise standard breakdowns, then writes a JSON summary containing:

- `benchmark`.
- `profile`, set to `qwen36-deterministic-no-thinking`.
- `expected`.
- `received`.
- `missing`.
- `invalid`.
- Benchmark-specific `metrics`.

Grading never performs inference. The same raw run can be graded repeatedly.

### 12. Adding a benchmark

A future deterministic, code-graded benchmark is complete when its directory:

1. Documents provider source, selected profile, dependencies, commands, and deviations.
2. Downloads/caches provider data and emits the three-file prepared contract.
3. Uses provider grading code under the provider-first policy.
4. Converts raw envelopes into standard benchmark metrics.
5. Includes tiny preparation/grading fixtures.
6. Runs through the unchanged Python and Bash runners.

A benchmark needing several independent generations can emit several request IDs and group them in its grading cases. A benchmark needing interactive, stateful, or response-dependent requests is outside the current single-request contract and must not cause speculative orchestration to be added to the runner.

## Testing Decisions

### Test seams

There are two behavior-level seams:

1. **Runner seam**: invoke each runner through its command-line interface against a fake `llama-server`, then inspect files and exit status. Tests do not reach into runner functions.
2. **Benchmark seam**: invoke each preparer or grader through its command-line interface against tiny local fixtures, then inspect the prepared contract or metrics. Tests do not reproduce provider grading internals.

The runner seam is shared by Python and Bash. This is the highest practical seam: endpoint behavior in, append-only files and exit status out.

### Runner tests

A Python standard-library fake HTTP server supplies HTTP 200, 400, 408, 429, and 500 responses plus invalid JSON with HTTP 200.

The runner tests verify:

- Required arguments and sibling `ids.txt` discovery.
- Positive concurrency validation, sequential default behavior, and the active-request bound.
- Mismatched counts, unsafe IDs, and duplicate prepared IDs fail before requests begin.
- A successful request produces the exact two-field envelope.
- Transient statuses receive the agreed retries in both runners; invalid successful JSON receives them in Python.
- Other 4xx responses are not retried.
- Exhausted IDs are absent, later IDs continue, and final status is nonzero.
- Reusing an output path skips completed IDs and executes missing IDs.
- Changing output path starts every case again.
- Duplicate or unknown IDs in an existing raw run fail before execution.
- Interrupting between requests leaves earlier envelopes valid.
- Ctrl-C stops new launches while preserving responses from active requests.
- The Bash runner matches the Python runner's observable envelope, HTTP-status retry, and resume behavior without Python or JSON-tool dependencies in the script.

### Benchmark tests

Each benchmark carries only enough local fixture data to verify:

- Prepared-contract alignment and unique stable IDs.
- Its selected prompt/profile fields.
- Response extraction and provider adapter wiring.
- Known headline and breakdown metrics.
- Missing and malformed responses.
- LongBench middle truncation preserves the beginning and end and respects the configured budget.

Tests do not download full datasets, models, or GGUF files and do not require real inference hardware. Use Python's `unittest` unless provider code already imposes a test dependency.

## Acceptance Criteria

1. Each initial preparer downloads its official Hugging Face dataset into the working cache and produces a valid prepared benchmark.
2. Existing caches are reused without deliberately redownloading their contents.
3. The Python runner executes any valid prepared benchmark without benchmark imports or third-party packages.
4. The Bash runner executes the same opaque requests with Bash and curl on the board.
5. Both runners require endpoint, requests, and output arguments, infer sibling IDs, and accept an optional positive concurrency limit that defaults to one.
6. Both runners write the exact `{id, response}` raw envelope and flush each success immediately in completion order.
7. Transient failures receive three retries after the initial attempt; exhausted IDs remain absent while later cases continue.
8. A run with exhausted IDs exits nonzero without discarding successful responses.
9. Reusing an output path skips every completed ID and retries every absent ID.
10. A new output path executes a new independent run.
11. A normally interrupted raw run remains valid and can resume or be graded.
12. Each grader scores an existing raw run without inference and writes the agreed summary fields and provider-standard metrics.
13. IFEval invokes the official evaluator and LongBench invokes the provider direct-answer extractor and metric logic; the MMLU-Redux direct-answer exception is documented.
14. LongBench preparation enforces its explicit context budget through Qwen tokenization and middle truncation.
15. Adding a fixture benchmark requires no changes to either shared runner.
16. Neither runner accepts or discovers a model path, GGUF file, benchmark name, comparison endpoint, or server lifecycle command.
17. The minimal behavior-level runner and benchmark tests pass without a model or board.

## Out of Scope

- x86/CUDA-versus-RISC-V comparison orchestration.
- Statistical significance, thresholds, or automated regression verdicts.
- Multiple endpoints in one invocation.
- `llama-server` launch, configuration, health management, or shutdown.
- llama.cpp version pinning, negotiation, or compatibility layers.
- Model/GGUF download, path arguments, checksums, or artifact comparison.
- Hardware discovery and dedicated performance or latency reporting.
- Dataset revision pinning and explicit refresh machinery.
- Automated transfer to or from the board.
- Authentication and custom HTTP headers.
- Cross-benchmark aggregate scores.
- Interactive or response-dependent benchmark orchestration.

## Further Notes

Implementation order is the shared contract and runner seam, MMLU-Redux as the smallest complete vertical slice, IFEval provider integration, then LongBench tokenization and provider integration. A new shared abstraction is justified only after two concrete benchmark implementations show duplicated preparation or grading behavior that the abstraction removes.

Normative upstream references:

- llama.cpp server API: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Qwen3.6 non-thinking guidance: https://huggingface.co/Qwen/Qwen3.6-27B#instruct-or-non-thinking-mode
- Unsloth Qwen3.6 llama.cpp guidance: https://unsloth.ai/docs/models/qwen3.6
- MMLU-Redux: https://github.com/aryopg/mmlu-redux and https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0
- IFEval: https://github.com/google-research/google-research/tree/master/instruction_following_eval and https://huggingface.co/datasets/google/IFEval
- LongBench: https://github.com/THUDM/LongBench and https://huggingface.co/datasets/zai-org/LongBench-v2
