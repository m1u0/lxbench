from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare


class PrepareAllTests(unittest.TestCase):
    @patch("prepare.subprocess.run")
    def test_prepares_every_benchmark_with_shared_paths(self, run):
        run.return_value = SimpleNamespace(returncode=0)

        result = prepare.main(["--longbench-context-size", "262144"])

        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 3)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            [command[1] for command in commands],
            [
                "benchmarks/ifeval/prepare.py",
                "benchmarks/longbench_v2/prepare.py",
                "benchmarks/mmlu_redux/prepare.py",
            ],
        )
        self.assertIn("262144", commands[1])
        self.assertTrue(all(call.kwargs["cwd"] == Path(prepare.ROOT) for call in run.call_args_list))

    @patch("prepare.subprocess.run")
    def test_force_is_forwarded_and_failure_stops_the_sequence(self, run):
        run.side_effect = [SimpleNamespace(returncode=0), SimpleNamespace(returncode=7)]

        result = prepare.main(
            ["--longbench-context-size", "4096", "--force"]
        )

        self.assertEqual(result, 7)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all("--force" in call.args[0] for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
