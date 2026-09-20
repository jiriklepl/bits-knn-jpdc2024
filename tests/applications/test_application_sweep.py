"""Exercise all application batch entry points with a CPU-only fake runner."""

from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
APPLICATIONS = ("database-topn", "token-sampling", "gradient-compression")
BITS = ("bits-prefetch", "bits-sq")
BASELINES = ("air-topk", "grid-select", "block-select")

FAKE_RUNNER = """import csv
import json
import os
import sys

arguments = sys.argv[1:]
options = dict(zip(arguments[1::2], arguments[2::2]))
with open(os.environ['SWEEP_CALLS'], 'a') as output:
    output.write(json.dumps(arguments) + '\\n')
writer = csv.writer(sys.stdout)
writer.writerow(['backend', 'k', 'block_size'])
for backend in options['--backends'].split(','):
    writer.writerow([backend, options['--k'], options['--bits-block-size']])
if (options['--k'], options['--bits-block-size']) == ('64', '256'):
    if os.environ.get('SWEEP_FAIL'):
        print('simulated verification failure', file=sys.stderr)
        sys.exit(7)
"""


class ApplicationSweepTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("config.sh", "tensor-benchmark.sh"):
            shutil.copyfile(SCRIPTS / name, scripts / name)
        (scripts / "run-applications.py").write_text(FAKE_RUNNER)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text("{}")
        self.calls = self.root / "calls.jsonl"
        self.env = dict(os.environ)
        self.env.pop("SLURM_JOB_ID", None)
        self.env.pop("SWEEP_FAIL", None)
        self.env.update(
            SLURM_SUBMIT_DIR=str(self.root),
            build_dir=str(self.root / "build"),
            SWEEP_CALLS=str(self.calls),
        )

    def run_batch(self, application, *ks, fail=False):
        self.calls.unlink(missing_ok=True)
        env = dict(self.env)
        if fail:
            env["SWEEP_FAIL"] = "1"
        run = subprocess.run(
            ["bash", str(SCRIPTS / f"run-{application}.sh"), str(self.manifest), *ks],
            text=True,
            capture_output=True,
            env=env,
        )
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        return run, calls

    def test_bits_variants_and_single_baselines_have_one_csv_header(self):
        for application in APPLICATIONS:
            with self.subTest(application=application):
                run, calls = self.run_batch(application, "32", "64")
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(len(calls), 6)
                configurations = []
                for arguments in calls:
                    self.assertEqual(arguments[0], str(self.manifest))
                    options = dict(zip(arguments[1::2], arguments[2::2]))
                    self.assertEqual(
                        options["--binary"], str(self.root / "build" / application)
                    )
                    self.assertEqual(options["--warmup"], "10")
                    self.assertEqual(options["--repeat"], "30")
                    self.assertEqual(options["--degree"], "32")
                    self.assertNotIn("--items-per-thread", options)
                    configurations.extend(
                        (backend, options["--k"], options["--bits-block-size"])
                        for backend in options["--backends"].split(",")
                    )
                expected = [
                    (backend, k, block)
                    for k in ("32", "64")
                    for backend in BITS
                    for block in ("128", "256", "512")
                ] + [(backend, k, "512") for k in ("32", "64") for backend in BASELINES]
                self.assertEqual(Counter(configurations), Counter(expected))
                self.assertEqual(run.stdout.count("backend,k,block_size"), 1)
                rows = csv.DictReader(io.StringIO(run.stdout))
                self.assertEqual(
                    Counter((r["backend"], r["k"], r["block_size"]) for r in rows),
                    Counter(expected),
                )

    def test_failed_variant_prevents_publishing_any_timings(self):
        for application in APPLICATIONS:
            with self.subTest(application=application):
                run, calls = self.run_batch(application, "32", "64", fail=True)
                self.assertEqual(run.returncode, 7, run.stderr)
                self.assertEqual(len(calls), 5)
                self.assertEqual(run.stdout, "")
                self.assertIn("simulated verification failure", run.stderr)


if __name__ == "__main__":
    unittest.main()
