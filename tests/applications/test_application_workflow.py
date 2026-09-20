"""Check application workflow dispatch without Python workloads or a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
NODE_BUILDS = {
    "ampere": "build-ampere",
    "adalovelace": "build-adalovelace",
    "hopper": "build-hopper",
    "bw": "build-bw",
    "volta": "build-volta",
}
FAKE_COMMAND = """import json
import os
from pathlib import Path
import subprocess
import sys

name = Path(sys.argv[0]).name
arguments = sys.argv[1:]
with open(os.environ['WORKFLOW_CALLS'], 'a') as output:
    output.write(json.dumps({
        'command': name,
        'arguments': arguments,
        'worker': os.environ.get('worker'),
        'build_dir': os.environ.get('build_dir'),
        'CUDA_ARCHITECTURES': os.environ.get('CUDA_ARCHITECTURES'),
        'cwd': os.getcwd(),
    }) + '\\n')
if name == os.environ.get('WORKFLOW_FAIL_COMMAND'):
    print('simulated workflow failure', file=sys.stderr)
    sys.exit(37)
if name == 'srun':
    print('unexpected srun', file=sys.stderr)
    sys.exit(86)
if name == 'sbatch':
    script_index = next(i for i, arg in enumerate(arguments) if arg.endswith('.sh'))
    environment = dict(os.environ, SLURM_JOB_ID='24680')
    environment['SLURM_SUBMIT_DIR'] = os.getcwd()
    result = subprocess.run(['bash', *arguments[script_index:]], env=environment)
    sys.exit(result.returncode)
"""


def option(arguments, name):
    """Match argparse's last occurrence when callers override a default."""
    positions = [i for i, argument in enumerate(arguments) if argument == name]
    if not positions:
        raise AssertionError(f"Missing {name} in {arguments!r}")
    return arguments[positions[-1] + 1]


class ApplicationWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="application workflow ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("executor.sh", "config.sh", "run-application-scaling.sh"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        for name in ("local", *NODE_BUILDS):
            shutil.copyfile(ROOT / f"{name}-build.sh", self.root / f"{name}-build.sh")
        for name in (
            "prepare-application-scaling.py",
            "run-application-scaling.py",
        ):
            (scripts / name).touch()
        self.calls_path = self.root / "calls.jsonl"
        binaries = self.root / "fake-bin"
        binaries.mkdir()
        for name in ("python3", "sbatch", "srun"):
            path = binaries / name
            path.write_text(f"#!{sys.executable}\n{FAKE_COMMAND}")
            path.chmod(0o755)
        self.suite = self.root / "data/application-inputs/scaling/suite.json"
        self.suite.parent.mkdir(parents=True)
        self.suite.write_text("{}")
        self.environment = dict(os.environ)
        for name in (
            "worker",
            "builder",
            "build_dir",
            "root_dir",
            "CUDA_ARCHITECTURES",
            "SLURM_JOB_ID",
            "SLURM_SUBMIT_DIR",
            "WORKFLOW_FAIL_COMMAND",
        ):
            self.environment.pop(name, None)
        self.environment.update(
            PATH=f"{binaries}{os.pathsep}{os.environ['PATH']}",
            WORKFLOW_CALLS=str(self.calls_path),
        )

    def run_wrapper(self, wrapper, action, *arguments, fail_command=None):
        self.calls_path.unlink(missing_ok=True)
        environment = dict(self.environment)
        if fail_command:
            environment["WORKFLOW_FAIL_COMMAND"] = fail_command
        result = subprocess.run(
            ["bash", f"{wrapper}-build.sh", "selected-node", "90", action, *arguments],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
        )
        calls = (
            [json.loads(line) for line in self.calls_path.read_text().splitlines()]
            if self.calls_path.exists()
            else []
        )
        return result, calls

    def assert_path(self, actual, expected):
        self.assertEqual((self.root / actual).resolve(), Path(expected).resolve())

    def test_preparation_stays_local_and_preserves_quoted_arguments(self):
        cache = str(self.root / "model cache")
        for wrapper in ("local", "hopper"):
            with self.subTest(wrapper=wrapper):
                result, calls = self.run_wrapper(
                    wrapper,
                    "applications-prepare",
                    "--cache-directory",
                    cache,
                    "--local-files-only",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([call["command"] for call in calls], ["python3"])
                arguments = calls[0]["arguments"]
                self.assert_path(
                    arguments[0], self.root / "scripts/prepare-application-scaling.py"
                )
                self.assert_path(
                    option(arguments, "--database-python"),
                    self.root / ".venv/bin/python",
                )
                self.assert_path(
                    option(arguments, "--model-python"),
                    self.root / ".venv-model/bin/python",
                )
                self.assertEqual(
                    arguments[-3:], ["--cache-directory", cache, "--local-files-only"]
                )

    def test_local_run_uses_selected_build_and_forwards_runner_options(self):
        arguments = ["--ks", "32", "64", "--degrees", "8", "32", "--dry-run"]
        result, calls = self.run_wrapper("local", "applications-run", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call["command"] for call in calls], ["python3"])
        invocation = calls[0]["arguments"]
        self.assert_path(
            invocation[0], self.root / "scripts/run-application-scaling.py"
        )
        self.assert_path(invocation[1], self.suite)
        self.assert_path(
            option(invocation, "--build-dir"), self.root / "build-selected-node"
        )
        self.assertEqual(invocation[-len(arguments) :], arguments)
        output = Path(option(invocation, "--output-dir"))
        self.assert_path(output.parent, self.root / "data/application-scaling")
        self.assertTrue(output.name.startswith("selected-node-"))

    def test_node_runs_export_selected_worker_and_architecture_specific_build(self):
        for wrapper, build in NODE_BUILDS.items():
            with self.subTest(wrapper=wrapper):
                result, calls = self.run_wrapper(
                    wrapper, "applications-run", "--dry-run"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    [call["command"] for call in calls], ["sbatch", "python3"]
                )
                submission, native = calls
                self.assertEqual(option(submission["arguments"], "-w"), "selected-node")
                exports = next(
                    arg
                    for arg in submission["arguments"]
                    if arg.startswith("--export=")
                )
                self.assertIn("worker", exports.split("=", 1)[1].split(","))
                self.assertIn("build_dir", exports.split("=", 1)[1].split(","))
                self.assertEqual(submission["worker"], "selected-node")
                self.assertEqual(submission["CUDA_ARCHITECTURES"], "90")
                self.assertEqual(submission["build_dir"], build)
                self.assert_path(
                    option(native["arguments"], "--build-dir"), self.root / build
                )
                output = Path(option(native["arguments"], "--output-dir"))
                self.assert_path(output.parent, self.root / "data/application-scaling")
                self.assertIn("selected-node", output.name)
                self.assertIn("24680", output.name)

    def test_local_default_run_directories_are_distinct(self):
        directories = []
        for _ in range(2):
            result, calls = self.run_wrapper("local", "applications-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            directories.append(option(calls[0]["arguments"], "--output-dir"))
        self.assertNotEqual(*directories)

    def test_resume_preserves_explicit_suite_and_output_paths(self):
        suite = self.root / "custom input suite.json"
        suite.write_text("{}")
        output = self.root / "previous output"
        for wrapper in ("local", "hopper"):
            with self.subTest(wrapper=wrapper):
                result, calls = self.run_wrapper(
                    wrapper,
                    "applications-run",
                    "--suite",
                    str(suite),
                    "--output-dir",
                    str(output),
                    "--resume",
                    "--repeat",
                    "3",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = calls[-1]["arguments"]
                self.assert_path(arguments[1], suite)
                self.assert_path(option(arguments, "--output-dir"), output)
                self.assertIn("--resume", arguments)
                self.assertEqual(option(arguments, "--repeat"), "3")
                self.assertNotIn("--suite", arguments)

    def test_resume_without_output_is_rejected_before_runner(self):
        result, calls = self.run_wrapper("local", "applications-run", "--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_equals_form_preserves_quoted_paths(self):
        output = self.root / "existing output"
        result, calls = self.run_wrapper(
            "local",
            "applications-run",
            f"--suite={self.suite}",
            f"--output-dir={output}",
            "--resume",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = calls[0]["arguments"]
        self.assert_path(arguments[1], self.suite)
        self.assert_path(option(arguments, "--output-dir"), output)
        self.assertIn("--resume", arguments)

    def test_missing_option_paths_are_rejected_before_runner(self):
        for arguments in (
            ("--suite",),
            ("--output-dir",),
            ("--suite=",),
            ("--output-dir=",),
            ("--suite", "--resume"),
        ):
            with self.subTest(arguments=arguments):
                result, calls = self.run_wrapper(
                    "local", "applications-run", *arguments
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])

    def test_run_help_prints_locally_without_submitting_work(self):
        for wrapper in ("local", "hopper"):
            for help_flag in ("--help", "-h"):
                with self.subTest(wrapper=wrapper, help_flag=help_flag):
                    result, calls = self.run_wrapper(
                        wrapper, "applications-run", help_flag
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Usage:", result.stdout)
                    self.assertIn("--suite", result.stdout)
                    self.assertEqual(calls, [])

    def test_command_failures_propagate_to_wrappers(self):
        cases = (
            ("local", "applications-prepare", (), "python3"),
            ("local", "applications-run", (), "python3"),
            ("hopper", "applications-run", (), "sbatch"),
        )
        for wrapper, action, arguments, command in cases:
            with self.subTest(wrapper=wrapper, action=action):
                result, calls = self.run_wrapper(
                    wrapper, action, *arguments, fail_command=command
                )
                self.assertEqual(result.returncode, 37, result.stderr)
                self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
