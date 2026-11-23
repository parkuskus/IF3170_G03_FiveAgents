import importlib
import unittest
from pathlib import Path
from unittest.mock import patch


KNN_EXPERIMENTS = Path(__file__).resolve().parents[1] / "src/experiments/knn"
ENTRY_POINTS = (
    "run_experiments.py",
    "run_manhattan_k_experiment.py",
    "run_feature_engineering_experiment.py",
    "run_cost_sensitive_experiment.py",
    "create_submission.py",
)


class KNNExperimentEntrypointTest(unittest.TestCase):
    def test_entrypoints_use_canonical_names(self):
        for filename in ENTRY_POINTS:
            with self.subTest(filename=filename):
                self.assertTrue((KNN_EXPERIMENTS / filename).is_file())

    def test_entrypoints_import_as_project_modules(self):
        for filename in ENTRY_POINTS:
            module_name = filename.removesuffix(".py")
            with self.subTest(module=module_name):
                importlib.import_module(f"src.experiments.knn.{module_name}")

    def test_entrypoints_write_outputs_to_knn_directory(self):
        for module_name, path_name in (
            ("run_experiments", "RESULT_CSV"),
            ("run_manhattan_k_experiment", "RESULT_CSV"),
            ("run_feature_engineering_experiment", "RESULT_CSV"),
            ("create_submission", "SUBMISSION_CSV"),
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(
                    f"src.experiments.knn.{module_name}"
                )
                self.assertEqual(getattr(module, path_name).parent.name, "knn")

    def test_manhattan_entrypoint_reads_arguments_before_data(self):
        module = importlib.import_module(
            "src.experiments.knn.run_manhattan_k_experiment"
        )
        with (
            patch.object(module.Path, "cwd", return_value=module.REPO_ROOT),
            patch.object(module.pd, "read_csv", side_effect=RuntimeError("reached data")),
        ):
            with self.assertRaisesRegex(RuntimeError, "reached data"):
                module.main()
