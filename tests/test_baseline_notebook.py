import json
import ast
import unittest
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "Notebook_FiveAgents_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb"


class BaselineNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        cls.by_id = {cell["id"]: "".join(cell["source"]) for cell in cls.cells}

    def test_all_existing_sections_remain(self):
        headings = "\n".join(self.by_id.values())
        for number in range(1, 12):
            self.assertIn(f'<a name="{number}"></a>', headings)

    def test_notebook_cell_ids_are_unique(self):
        notebook_paths = [NOTEBOOK, NOTEBOOK.parent / "Model_Pipeline_Template.ipynb"]
        notebook_paths.extend(NOTEBOOK.parent.glob("Model_Pipeline_Template_*.ipynb"))
        for path in notebook_paths:
            with self.subTest(notebook=path.name):
                cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
                ids = [cell["id"] for cell in cells]
                self.assertEqual(len(ids), len(set(ids)))

    def test_scratch_variants_and_comparators_are_run(self):
        modeling = "\n".join(self.by_id[key] for key in
                             ("7febed14", "4194347e", "20d20452", "4b9a54ff", "b1f35284", "52546b44", "1ee780f2"))
        for variant in ("knn", "dtl_id3", "dtl_c45", "dtl_cart",
                        "logistic_l1", "logistic_l2", "logistic_elasticnet",
                        "svm_linear", "svm_poly", "svm_rbf", "gaussian_nb"):
            with self.subTest(variant=variant):
                self.assertIn(f"'{variant}'", modeling)
        self.assertIn("comparison_results", modeling)

    def test_data_understanding_uses_charts_without_target_leakage(self):
        charts = self.by_id["data-understanding-charts"]
        self.assertIn("import matplotlib.pyplot as plt", self.by_id["55e390cf"])
        self.assertIn("matplotlib", self.by_id["ff5e60e9"])
        for signal in ("target", "missing", "days_before_arrival", "nightly_rate_idr",
                       "categorical_features", "arrival_month"):
            with self.subTest(signal=signal):
                self.assertIn(signal, charts)
        self.assertIn("train_set", charts)
        self.assertNotIn("val_set", charts)
        drift = self.by_id["data-understanding-drift"]
        self.assertIn("train_df", drift)
        self.assertIn("test_df", drift)
        self.assertNotIn("is_canceled", drift)

    def test_numeric_drift_handles_zero_variance_and_missing_samples(self):
        source = self.by_id["data-understanding-drift"]
        tree = ast.parse(source)
        helpers = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "standardized_mean_difference"
        ]
        self.assertEqual(len(helpers), 1)
        if not helpers:
            return
        helper = helpers[0]
        namespace = {"np": np, "pd": __import__("pandas")}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "drift", "exec"), namespace)
        calculate = namespace["standardized_mean_difference"]

        self.assertTrue(np.isinf(calculate([0, 0], [1, 1])))
        self.assertEqual(calculate([1, 1], [1, 1]), 0)
        self.assertTrue(np.isnan(calculate([np.nan], [np.nan])))
        self.assertTrue(np.isnan(calculate([1], [np.nan])))

    def test_baseline_models_live_in_their_notebook_sections(self):
        for cell_id, name in (("7febed14", "KNNBaseline"), ("4194347e", "CARTBaseline"),
                              ("20d20452", "LogisticBaseline"), ("4b9a54ff", "SVMBaseline"),
                              ("b1f35284", "GaussianNBBaseline")):
            with self.subTest(name=name):
                self.assertIn(f"class {name}:", self.by_id[cell_id])
        self.assertNotIn("from baseline_models import", "\n".join(self.by_id.values()))

    def test_baseline_and_pipeline_code_lines_are_readable(self):
        code_cell_ids = (
            "baseline-preprocessing", "7febed14", "4194347e", "20d20452",
            "4b9a54ff", "b1f35284", "52546b44", "1ee780f2",
            "fd95fdd6", "d2bed3e8", "baseline-error-analysis",
        )
        for cell_id in code_cell_ids:
            with self.subTest(cell_id=cell_id):
                long_lines = [
                    (number, len(line))
                    for number, line in enumerate(self.by_id[cell_id].splitlines(), 1)
                    if len(line) > 100
                ]
                self.assertEqual(long_lines, [])

    def test_split_metric_and_preprocessing_contract(self):
        cells = "\n".join(self.by_id.values())
        self.assertIn("fit_transform(X_train_raw)", cells)
        self.assertIn("transform(X_val_raw)", cells)
        self.assertIn("labels=[0, 1]", cells)
        self.assertIn("zero_division=0", cells)
        self.assertIn("valid_idx", cells)
        self.assertNotIn("assert ", "\n".join("".join(c["source"]) for c in self.cells if c["cell_type"] == "code"))
        self.assertIn("scratch_confusions[name]", self.by_id["52546b44"])
        self.assertIn("comparison_confusions[name]", self.by_id["1ee780f2"])

    def test_improvements_are_in_template_not_main(self):
        self.assertIn("Model_Pipeline_Template.ipynb", self.by_id["464f6abe"])
        self.assertNotIn("Write your code here", self.by_id["98d1e28c"])
        prep = "\n".join("".join(c["source"]) for c in self.cells
                         if c["id"] in ("insight2a", "321af912", "59d6f0f6", "b3856287",
                                        "82227603", "de7f5edb", "46dd91dd", "8f285bcf"))
        self.assertNotIn("tiap branch", prep)
        self.assertNotIn("model branch", prep)

    def test_winner_refits_and_persists_and_submits(self):
        self.assertIn("train_df[feature_columns]", self.by_id["fd95fdd6"])
        self.assertIn("dump(", self.by_id["fd95fdd6"])
        self.assertIn("load(", self.by_id["fd95fdd6"])
        self.assertIn("winner_name.startswith('dtl_')", self.by_id["fd95fdd6"])
        self.assertIn("test_df['id']", self.by_id["d2bed3e8"])
        self.assertIn("submission.csv", self.by_id["d2bed3e8"])
        self.assertIn("y_val", self.by_id["baseline-error-analysis"])

    def test_error_analysis_covers_validation_slices_and_reports_insights(self):
        analysis = self.by_id["baseline-error-analysis"]
        insights = self.by_id.get("error-analysis-insights", "")
        for signal in (
            "false_positive", "false_negative", "confusion_matrix",
            "booking_channel", "days_before_arrival", "missing_feature_count",
            "model_disagreement",
        ):
            with self.subTest(signal=signal):
                self.assertIn(signal, analysis)
        self.assertIn("overall_error_rate", analysis)
        self.assertIn("overall_error_rate", insights)
        self.assertIn("class 1 = canceled", insights)
        self.assertNotIn("test_df", analysis + insights)

    def test_error_analysis_plots_confusion_matrices_for_each_scratch_baseline(self):
        analysis = self.by_id["baseline-error-analysis"]
        self.assertIn("scratch_confusions.items()", analysis)
        self.assertIn("Confusion matrices for all baseline variants (validation set)", analysis)
        self.assertNotIn("fig.colorbar", analysis)
        self.assertNotIn("display(pd.DataFrame(\n    confusion", analysis)

    def test_key_data_and_error_findings_are_written_in_markdown(self):
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in self.cells
            if cell["cell_type"] == "markdown"
        )
        for finding in (
            "32,8%", "25 kombinasi fitur", "Seluruh 12 pemeriksaan kualitas",
            "17,7%", "False negative (542)", "false positive (427)", "69,8%", "31,2%",
            "28,8%", "19,0%", "4,5%",
        ):
            with self.subTest(finding=finding):
                self.assertIn(finding, markdown)

    def test_sklearn_comparison_discloses_cart_representation_difference(self):
        comparison_note = self.by_id["cee02d45"]
        self.assertIn("split dan evaluasi yang sama", comparison_note)
        self.assertIn("CART scratch memakai fitur raw", comparison_note)
        self.assertIn("memakai fitur one-hot encoded", comparison_note)

    def test_confusion_matrix_tables_are_not_duplicated(self):
        self.assertNotIn("display(scratch_confusion_table)", self.by_id["52546b44"])
        self.assertNotIn("display(comparison_confusion_table)", self.by_id["1ee780f2"])
        for cell_id in ("52546b44", "1ee780f2"):
            with self.subTest(cell_id=cell_id):
                cell = next(cell for cell in self.cells if cell["id"] == cell_id)
                self.assertEqual(len(cell["outputs"]), 2)

    def test_saved_bundle_does_not_require_an_external_python_model_module(self):
        source = self.by_id["fd95fdd6"]
        self.assertIn("'model_state': final_model.__dict__", source)
        self.assertIn("restored_model.__dict__.update", source)
        self.assertNotIn("'model': final_model", source)

    def test_model_specific_templates_keep_shared_contract(self):
        templates = {
            "DTL": ("DTL_Final-Pipeline.ipynb", "'C4.5': {'criterion': 'gain_ratio'}"),
            "Logistic Regression": (
                "Logistic_Regression_Final-Pipeline.ipynb",
                "'penalty': 'elasticnet'",
            ),
            "SVM": ("SVM_Final-Pipeline.ipynb", "'kernel': 'rbf'"),
            "Naive Bayes": (
                "Naive_Bayes_Final-Pipeline.ipynb",
                "'variance_smoothing': 1e-9",
            ),
        }
        for model_name, (filename, configuration) in templates.items():
            with self.subTest(model=model_name):
                path = NOTEBOOK.parent / filename
                self.assertTrue(path.is_file())
                template_notebook = json.loads(path.read_text(encoding="utf-8"))
                cells = template_notebook["cells"]
                self.assertEqual(
                    template_notebook["metadata"]["model_template"]["filename"],
                    path.stem,
                )
                source = "\n".join("".join(cell["source"]) for cell in cells)
                for section in range(1, 12):
                    self.assertIn(f'<a name="{section}"></a>', source)
                self.assertIn("SEED = 42", source)
                self.assertIn("average='macro'", source)
                self.assertIn(model_name.lower(), source.lower())
                self.assertIn(configuration, source)
                readme = (NOTEBOOK.parent.parent / "README.md").read_text(
                    encoding="utf-8"
                )
                self.assertIn(f"notebooks/{filename}", readme)

    def test_dtl_template_includes_id3_c45_and_cart(self):
        path = NOTEBOOK.parent / "DTL_Final-Pipeline.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        for variant in ("ID3", "C4.5", "CART"):
            with self.subTest(variant=variant):
                self.assertIn(variant, source)

if __name__ == "__main__":
    unittest.main()
