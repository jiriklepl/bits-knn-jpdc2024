"""Check automatic study discovery without rendering plots or running benchmarks."""

from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "plot_application_scaling_discovery", SCRIPTS / "plot-application-scaling.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ScalingDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        previous = Path.cwd()
        os.chdir(temporary.name)
        self.addCleanup(os.chdir, previous)
        self.data = Path("data/application-scaling")

    def index(self, name, *, status="complete", root=None):
        path = (root or self.data) / name / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": [{"status": status}]}))
        return path

    @staticmethod
    def rows(index, *, annotate):
        return [
            dict(
                source_csv=str(index.parent / f"{operator}-{tier}.csv"),
                operator=operator,
                workload_id=f"{operator}-{tier}",
                size_tier=tier,
                mode="fixed-k",
            )
            for operator in cli.SIZE_LABELS
            for tier in ("small", "middle", "large")
        ]

    def invoke(self, *arguments, load=None):
        stderr = io.StringIO()
        with patch.object(sys, "argv", [cli.__file__, *map(str, arguments)]), (
            redirect_stderr(stderr)
        ), patch.object(cli, "load_study", side_effect=load or self.rows) as read, (
            patch.object(cli.subprocess, "run")
        ) as execute:
            status = 0
            try:
                cli.main()
            except SystemExit as error:
                status = error.code
        return status, stderr.getvalue(), read, execute

    def test_default_walk_preserves_nested_names_and_dispatches_nine_cases(self):
        second = self.index("hopper/same-name")
        first = self.index("blackwell/same-name")
        status, _, read, execute = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(
            [call.args[0] for call in read.call_args_list], [first, second]
        )
        self.assertTrue(
            all(call.kwargs == {"annotate": False} for call in read.call_args_list)
        )
        self.assertEqual(execute.call_count, 18)
        for call in execute.call_args_list:
            command = call.args[0]
            relative = Path(command[2]).parent.relative_to(self.data)
            self.assertEqual(
                command[-2:],
                ["--output-dir", str(Path("plots/application-scaling") / relative)],
            )
            self.assertEqual(call.kwargs, {"check": True})

    def test_custom_roots_skip_unfinished_studies(self):
        root = Path("saved studies")
        complete = self.index("done", root=root)
        incomplete = [
            self.index(status, status=status, root=root)
            for status in ("pending", "running", "failed")
        ]
        status, stderr, read, execute = self.invoke(
            "--data-dir", root, "--output-dir", "custom plots"
        )
        self.assertEqual(status, 0)
        read.assert_called_once_with(complete, annotate=False)
        for index in incomplete:
            self.assertIn(f"Skipping incomplete study: {index}", stderr)
        self.assertEqual(execute.call_count, 9)
        self.assertTrue(
            all(
                call.args[0][-2:] == ["--output-dir", "custom plots/done"]
                for call in execute.call_args_list
            )
        )

    def test_explicit_index_uses_exact_output_and_does_not_scan(self):
        explicit = self.index("chosen")
        self.index("unrelated")
        for arguments, output in (
            ([], "plots/chosen"),
            (["--output-dir", "exact output"], "exact output"),
        ):
            with self.subTest(arguments=arguments):
                status, _, read, execute = self.invoke(explicit, *arguments)
                self.assertEqual(status, 0)
                read.assert_called_once_with(explicit, annotate=False)
                self.assertEqual(execute.call_count, 9)
                self.assertTrue(
                    all(
                        call.args[0][-2:] == ["--output-dir", output]
                        for call in execute.call_args_list
                    )
                )

    def test_explicit_unfinished_index_still_fails_validation(self):
        index = self.index("running", status="running")
        status, stderr, read, execute = self.invoke(
            index, load=ValueError("run is running")
        )
        self.assertEqual(status, 1)
        self.assertIn("run is running", stderr)
        self.assertNotIn("Skipping", stderr)
        read.assert_called_once_with(index, annotate=False)
        execute.assert_not_called()

    def test_no_indices_is_a_successful_noop(self):
        status, stderr, read, execute = self.invoke()
        self.assertEqual(status, 0)
        self.assertIn("No study indices found under data/application-scaling", stderr)
        read.assert_not_called()
        execute.assert_not_called()
        self.assertFalse(Path("plots").exists())

    def test_invalid_studies_are_reported_without_stopping_other_studies(self):
        invalid = self.index("a-invalid-json")
        invalid.write_text("{bad json")
        corrupt = self.index("b-bad-provenance")
        complete = self.index("c-complete")

        def validate(index, *, annotate):
            if index == corrupt:
                raise ValueError("CSV hash mismatch")
            return self.rows(index, annotate=annotate)

        status, stderr, read, execute = self.invoke(load=validate)
        self.assertEqual(status, 1)
        self.assertIn(str(invalid), stderr)
        self.assertIn(f"{corrupt}: CSV hash mismatch", stderr)
        self.assertEqual(
            [call.args[0] for call in read.call_args_list], [corrupt, complete]
        )
        self.assertEqual(execute.call_count, 9)

    def test_discovery_cannot_write_plots_into_raw_study_directories(self):
        self.index("complete")
        status, stderr, _, execute = self.invoke("--output-dir", self.data)
        self.assertEqual(status, 1)
        self.assertIn("Choose a plot directory separate from raw study outputs", stderr)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
