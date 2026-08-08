import os
from pathlib import Path


original_replace = os.replace
failed_swap = False


def fail_atomic_swap(source, destination):
    global failed_swap
    log_path = os.environ.get("MMLU_FIXTURE_REPLACE_LOG")
    if log_path:
        with Path(log_path).open("a", encoding="utf-8") as log_file:
            log_file.write(f"{source}\t{destination}\n")
    if (
        os.environ.get("MMLU_FIXTURE_FAIL_ATOMIC_SWAP")
        and Path(destination).name == "prepared"
        and not failed_swap
    ):
        failed_swap = True
        raise OSError("fixture atomic swap failure")
    return original_replace(source, destination)


os.replace = fail_atomic_swap
