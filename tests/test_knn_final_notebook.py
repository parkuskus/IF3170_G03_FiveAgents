"""End-to-end contract for the final KNN notebook."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks" / "KNN_Final-Pipeline.ipynb"
)
KNN_SOURCE = Path(__file__).resolve().parents[1] / "src" / "experiments" / "knn"


class KNNFinalNotebookTest(unittest.TestCase):
    def test_runs_and_writes_final_ordered_submission(self):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            source = root / "src" / "experiments" / "knn"
            source.mkdir(parents=True)
            for filename in (
                "knn.py",
                "preprocessing.py",
                "run_cost_sensitive_experiment.py",
                "run_experiments.py",
            ):
                shutil.copy2(KNN_SOURCE / filename, source / filename)

            labels = np.array([0, 1] * 60, dtype=np.int64)
            train = pd.DataFrame({
                "id": [f"train-{index}" for index in range(len(labels))],
                "amount": np.where(labels == 1, 20.0, 2.0),
                "channel": np.where(labels == 1, "web", "desk"),
                "is_canceled": labels,
            })
            test = pd.DataFrame({
                "id": ["test-z", "test-a", "test-b"],
                "amount": [20.0, 2.0, None],
                "channel": ["web", "desk", "new"],
            })
            train.to_csv(root / "data" / "train.csv", index=False)
            test.to_csv(root / "data" / "test.csv", index=False)

            old_cwd = Path.cwd()
            old_path = sys.path[:]
            old_src_modules = {
                name: module
                for name, module in sys.modules.items()
                if name == "src" or name.startswith("src.")
            }
            namespace = {"__name__": "__main__", "display": lambda *_: None}
            try:
                os.chdir(root)
                sys.path.insert(0, str(root))
                for name in old_src_modules:
                    del sys.modules[name]
                for index, cell in enumerate(cells):
                    if cell["cell_type"] == "code":
                        exec(
                            compile(
                                "".join(cell["source"]),
                                f"notebook cell {index}",
                                "exec",
                            ),
                            namespace,
                        )
            finally:
                os.chdir(old_cwd)
                sys.path[:] = old_path
                if "saved_arrays" in namespace:
                    namespace["saved_arrays"].close()
                for name in tuple(sys.modules):
                    if name == "src" or name.startswith("src."):
                        del sys.modules[name]
                sys.modules.update(old_src_modules)

            submission = pd.read_csv(
                root / "output" / "knn" / "submission_knn.csv",
                dtype={"id": str},
            )
            self.assertEqual(submission["id"].tolist(), test["id"].tolist())
            self.assertEqual(submission.columns.tolist(), ["id", "is_canceled"])
            self.assertTrue(submission["is_canceled"].isin([0, 1]).all())

            model_dir = root / "output" / "final" / "model" / "knn"
            config = json.loads(
                (model_dir / "konfigurasi.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {
                    "k": config["k"],
                    "weights": config["weights"],
                    "metric": config["metric"],
                    "cat_weight": config["cat_weight"],
                    "threshold": config["threshold"],
                },
                {
                    "k": 21,
                    "weights": "distance",
                    "metric": "manhattan",
                    "cat_weight": 1.0,
                    "threshold": 0.5,
                },
            )
            self.assertEqual(
                config["macro_f1_validasi"],
                round(float(namespace["validation_macro_f1"]), 6),
            )
            self.assertTrue((model_dir / "knn_model.npz").exists())
            self.assertTrue((model_dir / "preprocessing.json").exists())
            self.assertTrue((root / "output" / "knn" / "knn_cost_sensitive.png").exists())
            self.assertFalse((root / "output" / "submission_knn.csv").exists())


if __name__ == "__main__":
    unittest.main()
