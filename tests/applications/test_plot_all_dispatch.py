"""Check shell plot dispatch without installing dependencies or rendering plots."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

FAKE_PYTHON = """import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["PLOT_DISPATCH_CALLS"], "a") as output:
    output.write(json.dumps(args) + "\\n")
if args[:2] == ["-m", "venv"]:
    activate = pathlib.Path(args[2]) / "bin" / "activate"
    activate.parent.mkdir(parents=True, exist_ok=True)
    activate.write_text("# Stub environment: keep the fake Python on PATH.\\n")
elif args and args[0].endswith(".py"):
    sys.exit(int(os.environ.get("PLOT_DISPATCH_STATUS", "0")))
"""


class PlotAllDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="plot dispatch ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        shutil.copy2(SCRIPTS / "plot-all.sh", scripts / "plot-all.sh")
        for name in ("application-scaling", "database-topn", "token-sampling"):
            (scripts / f"plot-{name}.py").write_text("# Dispatch target only.\n")
        commands = self.root / "commands"
        commands.mkdir()
        python = commands / "python3"
        python.write_text(f"#!{sys.executable}\n{FAKE_PYTHON}")
        python.chmod(0o755)
        self.calls = self.root / "calls.jsonl"
        self.env = dict(
            os.environ,
            PATH=f"{commands}{os.pathsep}{os.environ['PATH']}",
            PLOT_DISPATCH_CALLS=str(self.calls),
        )

    def dispatch(self, *arguments, status=0):
        self.calls.unlink(missing_ok=True)
        run = subprocess.run(
            ["bash", "scripts/plot-all.sh", *arguments],
            cwd=self.root,
            env=dict(self.env, PLOT_DISPATCH_STATUS=str(status)),
            text=True,
            capture_output=True,
        )
        calls = (
            [json.loads(line) for line in self.calls.read_text().splitlines()]
            if self.calls.exists()
            else []
        )
        return run, calls

    def test_selected_plot_receives_all_arguments_with_spaces_preserved(self):
        arguments = (
            "results with spaces/index.json",
            "--output-dir",
            "plots with spaces",
        )
        run, calls = self.dispatch("application-scaling", *arguments)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(
            calls,
            [
                ["-m", "venv", ".venv"],
                ["-m", "pip", "install", "-r", "scripts/requirements.txt"],
                ["scripts/plot-application-scaling.py", *arguments],
            ],
        )

    def test_selected_application_scaling_discovers_without_index(self):
        run, calls = self.dispatch("application-scaling")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(calls[-1], ["scripts/plot-application-scaling.py"])

    def test_discovery_options_are_forwarded_without_index(self):
        arguments = ("--data-dir", "saved studies", "--output-dir", "saved plots")
        run, calls = self.dispatch("application-scaling", *arguments)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(calls[-1], ["scripts/plot-application-scaling.py", *arguments])

    def test_all_and_default_include_application_scaling(self):
        for arguments in ((), ("all",)):
            with self.subTest(arguments=arguments):
                run, calls = self.dispatch(*arguments)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertCountEqual(
                    calls[2:],
                    [
                        ["scripts/plot-application-scaling.py"],
                        ["scripts/plot-database-topn.py"],
                        ["scripts/plot-token-sampling.py"],
                    ],
                )

    def test_help_explains_optional_index_without_environment_setup(self):
        run, calls = self.dispatch("help")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("application-scaling discovers saved indices", run.stderr)
        self.assertIn("also included in all", run.stderr)
        self.assertIn("application-scaling [INDEX] [--output-dir PATH]", run.stderr)
        self.assertEqual(calls, [])

    def test_invalid_dispatch_fails_before_environment_setup(self):
        for arguments in (("missing",), ("all", "unexpected")):
            with self.subTest(arguments=arguments):
                run, calls = self.dispatch(*arguments)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(calls, [])

    def test_selected_plot_failure_is_returned(self):
        run, calls = self.dispatch("database-topn", "input.csv", status=7)
        self.assertEqual(run.returncode, 7, run.stderr)
        self.assertEqual(calls[-1], ["scripts/plot-database-topn.py", "input.csv"])


if __name__ == "__main__":
    unittest.main()
