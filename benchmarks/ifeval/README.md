# IFEval

This benchmark prepares and grades the provider's 541-case IFEval dataset. It
uses the `qwen36-deterministic-no-thinking` profile and the official Google
Research evaluator copied into `upstream/`.

## Provider sources

- Dataset: <https://huggingface.co/datasets/google/IFEval>
- Evaluator: <https://github.com/google-research/google-research/tree/master/instruction_following_eval>
- Paper: <https://arxiv.org/abs/2311.07911>
- Copied evaluator revision and files: [upstream/SOURCE.md](upstream/SOURCE.md)
- Upstream license: Apache-2.0, copied as [upstream/LICENSE](upstream/LICENSE)

## Dependencies

Use Python 3.10 or newer on the workstation:

```sh
python -m pip install -r benchmarks/ifeval/requirements.txt
python -m nltk.downloader punkt punkt_tab
```

`datasets` provides the official Hugging Face data. The remaining packages are
the official evaluator's dependencies. Dataset and NLTK data are inputs; no
executable grading code is downloaded at runtime.

## Prepare, run, and grade

Preparation passes no explicit dataset revision. Hugging Face therefore reuses
the supplied working cache after the first download.

```sh
python benchmarks/ifeval/prepare.py \
  --cache-dir data/ifeval \
  --output prepared/ifeval

python run.py \
  --endpoint http://BOARD:PORT/v1/chat/completions \
  --requests prepared/ifeval/requests.jsonl \
  --output results/ifeval.jsonl

python benchmarks/ifeval/grade.py \
  --cases prepared/ifeval/cases.jsonl \
  --responses results/ifeval.jsonl \
  --output results/ifeval-scores.json
```

Preparation writes `requests.jsonl`, `ids.txt`, and `cases.jsonl` into a
versioned sibling directory, validates the complete aligned set, and atomically
switches the output symlink to that version. A validation failure leaves an
earlier prepared benchmark unchanged.

## Selected profile and deviations

Each provider prompt is sent unchanged as one user message with
`max_tokens: 8192`, `temperature: 0`, `stream: false`, and thinking disabled.
Requests contain no model, GGUF, `top_p`, or seed field. This deterministic
internal profile is not automatically comparable with provider leaderboard
inference settings.

The official evaluator normally joins a prompt/response JSONL file by prompt.
The adapter instead joins the harness's raw envelopes to prepared cases by
stable provider key, extracts only the string at
`response.choices[0].message.content`, and passes that string to the unmodified
provider strict and loose functions. It passes an empty response for missing or
structurally invalid envelopes so those cases remain in both prompt-level and
instruction-level denominators. The adapter does not trim or otherwise alter a
valid response.

The score summary reports the standard strict and loose prompt-level and
instruction-level accuracies, plus expected, received, missing, and invalid
counts.

## Tests

The command-level fixtures replace only the Hugging Face download boundary and
unused third-party tokenizer/language packages. They execute the real preparer,
adapter, and copied provider evaluator without a dataset download, model, or
board:

```sh
python -m unittest benchmarks.ifeval.tests.test_ifeval
```
