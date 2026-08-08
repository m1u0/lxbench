import os


failure_target = os.environ.get("IFEVAL_FAIL_AFTER_SWAP")
if failure_target:
    original_replace = os.replace
    failed = False

    def replace(source, destination):
        global failed
        original_replace(source, destination)
        if not failed and os.path.abspath(destination) == failure_target:
            failed = True
            raise KeyboardInterrupt("fixture interrupt after atomic swap")

    os.replace = replace
