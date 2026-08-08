import os
from pathlib import Path


original_write_text = Path.write_text


def fail_ids_write(path, *args, **kwargs):
    if os.environ.get("MMLU_FIXTURE_FAIL_PUBLISH") and path.name == "ids.txt":
        raise OSError("fixture publication failure")
    return original_write_text(path, *args, **kwargs)


Path.write_text = fail_ids_write
