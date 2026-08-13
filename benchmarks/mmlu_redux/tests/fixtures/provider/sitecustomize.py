import os
from pathlib import Path
import shutil


original_replace = os.replace
original_rmtree = shutil.rmtree
failed_swap = False
failed_cleanup = False


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
    result = original_replace(source, destination)
    if (
        os.environ.get("MMLU_FIXTURE_FAIL_AFTER_ATOMIC_SWAP")
        and Path(destination).name == "prepared"
        and not failed_swap
    ):
        failed_swap = True
        raise OSError("fixture interruption after atomic swap")
    return result


def fail_generation_cleanup(path, *args, **kwargs):
    global failed_cleanup
    if (
        os.environ.get("MMLU_FIXTURE_FAIL_GENERATION_CLEANUP")
        and kwargs.get("ignore_errors") is True
        and Path(path).name.startswith("version-")
        and not failed_cleanup
    ):
        failed_cleanup = True
        raise OSError("fixture interruption during old generation cleanup")
    return original_rmtree(path, *args, **kwargs)


os.replace = fail_atomic_swap
shutil.rmtree = fail_generation_cleanup
