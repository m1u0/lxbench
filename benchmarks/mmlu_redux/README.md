# MMLU-Redux 2.0

This benchmark prepares the official
[`edinburgh-dawg/mmlu-redux-2.0`](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0)
dataset. It uses the current provider revision on the first load (no revision is
pinned), stores Hugging Face's reusable dataset cache under `data/`, and keeps
only questions whose current `error_type` annotation is `ok`.

Install the benchmark-local dependency:

```sh
python3 -m pip install -r benchmarks/mmlu_redux/requirements.txt
```

Prepare the aligned requests, IDs, and grading cases:

```sh
python3 benchmarks/mmlu_redux/prepare.py \
  --cache-dir data/mmlu-redux-2.0 \
  --output prepared/mmlu-redux-2.0
```

Preparation reuses the working Hugging Face cache and publishes the prepared
directory only after all three files validate. Stable IDs combine the provider
subject name and the row's original offset within that subject.

Run the opaque requests with the unchanged shared runner:

```sh
python3 run.py \
  --endpoint http://board.example:8080/v1/chat/completions \
  --requests prepared/mmlu-redux-2.0/requests.jsonl \
  --output results/mmlu-redux-2.0/raw.jsonl
```

Grade an existing raw run:

```sh
python3 benchmarks/mmlu_redux/grade.py \
  --cases prepared/mmlu-redux-2.0/cases.jsonl \
  --responses results/mmlu-redux-2.0/raw.jsonl \
  --output results/mmlu-redux-2.0/score.json
```

## Profile and grading

Requests use the `qwen36-deterministic-no-thinking` profile: temperature `0`,
streaming disabled, Qwen thinking disabled, no model or GGUF fields, and a
16-token output cap. The prompt is zero-shot and asks for only one answer letter.

The [provider project](https://github.com/aryopg/mmlu-redux) does not include a
grader for this selected direct-answer profile. This benchmark therefore uses the
documented provider-first exception: after trimming the chat response content,
it accepts a prediction only when exactly one standalone, case-insensitive A, B,
C, or D token occurs. Missing, malformed, and ambiguous answers are incorrect.
The score reports overall and per-subject accuracy.
