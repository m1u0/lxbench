# LongBench v2

This benchmark implements the official LongBench v2 zero-shot direct profile
against the shared prepared-request runner.

## Provider sources

- Dataset: [`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2), loaded without an explicit revision so the working Hugging Face cache is reused.
- Prompt and grading: [`THUDM/LongBench`](https://github.com/THUDM/LongBench) at revision `2e00731f8d0bff23dc4325161044d0ed8af94c1e`.
- Tokenizer: [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

The minimal copied provider surface is in `upstream/`: the official `0shot.txt`
prompt, direct-answer extractor, difficulty/context-length metric loop, source
record, and MIT license. Provider inference, model loading, and multiprocessing
code are not copied.

## Dependencies

Use Python 3.10 or newer and install only this benchmark's workstation
dependencies:

```sh
python3 -m pip install -r benchmarks/longbench_v2/requirements.txt
```

Preparation downloads the LongBench dataset into `data/longbench-v2/dataset`
and only these Qwen tokenizer assets into `data/longbench-v2/tokenizer`:
`chat_template.jinja`, `merges.txt`, `tokenizer.json`,
`tokenizer_config.json`, and `vocab.json`. It does not request model weights,
safetensors, or GGUF files.

## Prepare, run, and grade

`--context-size` is required and must be the effective context configured for
the target server. Preparation reserves 128 tokens for generation, counts the
complete non-thinking Qwen chat template, and applies provider-style middle
truncation only when the rendered chat input exceeds the remaining budget.

```sh
python3 benchmarks/longbench_v2/prepare.py \
  --context-size 262144 \
  --cache data/longbench-v2 \
  --output prepared/longbench-v2

python3 run.py \
  --endpoint http://board.example:8080/v1/chat/completions \
  --requests prepared/longbench-v2/requests.jsonl \
  --output results/longbench-v2/raw.jsonl

python3 benchmarks/longbench_v2/grade.py \
  --cases prepared/longbench-v2/cases.jsonl \
  --responses results/longbench-v2/raw.jsonl \
  --output results/longbench-v2/score.json
```

Requests use the `qwen36-deterministic-no-thinking` profile: temperature `0`,
streaming disabled, thinking disabled, and `max_tokens: 128`. They deliberately
omit model and GGUF fields. The only inference-setting deviation from the
provider direct runner is temperature `0` instead of `0.1`, making this an
internal deterministic regression profile rather than an automatically
leaderboard-comparable run.

The grader reads the OpenAI-compatible string at
`response.choices[0].message.content`, joins responses by prepared ID, and uses
the provider answer extractor and standard overall, easy/hard, and
short/medium/long accuracy calculations. Missing, malformed, and unextractable
predictions remain in the denominator as incorrect.
