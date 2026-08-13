## Agent skills

### Issue tracker

Issues are tracked in this repository’s GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the default five-label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This repository uses a single-context domain documentation layout. See `docs/agents/domain.md`.

## Cursor Cloud specific instructions

`lxbench` is a Python 3.10+ CLI benchmark harness (no web app, no build step, no linter configured). The `prepare -> run -> grade` workflow is documented in `README.md` and each `benchmarks/*/README.md`.

- Python deps live in a virtualenv at `.venv` (gitignored). Use `.venv/bin/python` to get the benchmark dependencies; the core scripts (`run.py`, `prepare.py`, `sample_manifest.py`) are stdlib-only and also run under a bare `python3`.
- Tests are stdlib `unittest`, split into two suites: `python3 -m unittest discover -s tests -p 'test_*.py'` (runner/prepare) and `python3 -m unittest discover -s benchmarks -p 'test_*.py'` (per-benchmark, uses in-repo fake dependencies). Neither needs network or an inference server.
- The `tests/` suite takes ~75s because it exercises real retry/backoff sleeps — that runtime is expected, not a hang.
- There is no bundled inference server. `run.py` POSTs to an already-running OpenAI-compatible `/v1/chat/completions` endpoint. For local end-to-end testing without a real board, run a tiny mock HTTP server that returns a JSON chat-completion object and point `--endpoint` at it.
- Real dataset preparation (`prepare.py` / `benchmarks/*/prepare.py`) downloads from Hugging Face; caches go in gitignored `data/`, prepared files in `prepared/`, run outputs and scores in `results/`. IFEval also needs the NLTK `punkt` and `punkt_tab` corpora.
