"""Shared reading order for the five final model notebooks."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1] / "notebooks"
MODELS = ("DTL", "KNN", "Logistic_Regression", "Naive_Bayes", "SVM")
SECTIONS = (
    "Introduction", "Initialization", "Data Understanding",
    "Training and Validation Split", "Data Preparation", "Model Pipeline",
    "Modeling and Validation", "Comparison with scikit-learn",
    "Saving and Loading Models", "Kaggle Submission", "Error Analysis",
    "Cost-Sensitive Classification",
)
BONUSES = {
    "DTL": "Gambar Percabangan Tree",
    "KNN": "Eksperimen Biaya KNN",
    "Logistic_Regression": "Video Kontur Log-Loss",
    "Naive_Bayes": "Graf Bayesian Network",
    "SVM": "Bidang Pemisah SVM",
}


class NotebookStructureTest(unittest.TestCase):
    def test_naive_bayes_settings_imports_from_notebooks_directory(self):
        notebook = json.loads((ROOT / "Naive_Bayes_Final-Pipeline.ipynb").read_text(encoding="utf-8"))
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        code = "\n".join(cells[cell_id] for cell_id in ("template-imports", "template-settings"))
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sections_follow_template_and_bonus_follows_cost(self):
        for name in MODELS:
            with self.subTest(name=name):
                notebook = json.loads((ROOT / f"{name}_Final-Pipeline.ipynb").read_text(encoding="utf-8"))
                cells = notebook["cells"]
                headings = [
                    (index, "".join(cell["source"]))
                    for index, cell in enumerate(cells)
                    if cell["cell_type"] == "markdown" and '<a name="' in "".join(cell["source"])
                ]
                self.assertEqual(len(headings), 13)
                for number, ((_, source), title) in enumerate(zip(headings, SECTIONS), 1):
                    self.assertIn(f"# {title} <a name=\"{number}\"></a>", source)
                self.assertIn("# Bonus ", headings[-1][1])
                self.assertIn(BONUSES[name], headings[-1][1])
                toc = "".join(cells[1]["source"])
                for number in range(1, 14):
                    self.assertIn(f"](#{number})", toc)
                self.assertEqual(
                    notebook["metadata"]["model_template"]["filename"],
                    f"{name}_Final-Pipeline",
                )

    def test_dtl_narrative_does_not_expose_internal_versions(self):
        notebook = json.loads((ROOT / "DTL_Final-Pipeline.ipynb").read_text(encoding="utf-8"))
        text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        self.assertNotIn("DTL-v08", text)
        self.assertNotIn("DTL-v09", text)
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertNotIn("DTL-v08", code)

    @unittest.skipUnless(shutil.which("ffmpeg"), "Rendering MP4 requires ffmpeg")
    def test_logistic_bonus_renders_a_real_video_without_changing_final_model(self):
        notebook = json.loads((ROOT / "Logistic_Regression_Final-Pipeline.ipynb").read_text(encoding="utf-8"))
        bonus = next(cell for cell in notebook["cells"] if cell.get("id") == "lr-contour-video")
        with tempfile.TemporaryDirectory() as folder:
            namespace = {
                "np": np, "plt": plt,
                "X_train_model": np.array([[-2.0], [-1.0], [1.0], [2.0]] * 15),
                "y_train": np.array([0, 0, 1, 1] * 15),
                "CFG": type("Settings", (), {"SEED": 42, "OUTPUT_DIR": Path(folder)})(),
                "final_model": object(),
            }
            final_model = namespace["final_model"]
            exec("".join(bonus["source"]), namespace)
            video = Path(folder) / "logistic_regression_log_loss_trajectory.mp4"
            self.assertTrue(video.exists())
            self.assertGreater(video.stat().st_size, 1000)
            self.assertIs(namespace["final_model"], final_model)


if __name__ == "__main__":
    unittest.main()
