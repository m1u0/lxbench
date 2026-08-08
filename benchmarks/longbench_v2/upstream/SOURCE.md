# LongBench v2 upstream grading surface

- Source: https://github.com/THUDM/LongBench
- Copied revision: `2e00731f8d0bff23dc4325161044d0ed8af94c1e`
- License: MIT; preserved in `LICENSE`.

`0shot.txt` is copied from `prompts/0shot.txt`. `provider.py` retains the direct
answer extraction from `pred.py` and wraps the difficulty and context-length
counting from `result.py` as a callable function. Inference, model loading,
multiprocessing, and result-directory traversal were intentionally not copied.
