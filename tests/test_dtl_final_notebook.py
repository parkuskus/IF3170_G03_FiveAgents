"""End-to-end contract for the standalone DTL notebook on a small dataset."""

import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "DTL_Final-Pipeline.ipynb"


class FinalNotebookTest(unittest.TestCase):
    def test_runs_without_src_and_writes_ordered_submission(self):
        cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        source = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        self.assertNotIn("from src", source)
        self.assertNotIn("import src", source)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            labels = np.array([0, 1] * 50, dtype=int)
            train = pd.DataFrame({
                "id": [f"train-{i}" for i in range(100)],
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
            pd.DataFrame({"id": test["id"], "is_canceled": 0}).to_csv(
                root / "data" / "sample_submission.csv", index=False
            )
            old_cwd = Path.cwd()
            namespace = {"__name__": "__main__", "display": lambda *_: None}
            try:
                os.chdir(root)
                for index, cell in enumerate(cells):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"notebook cell {index}", "exec"), namespace)
                        if index == 63:
                            # In a real notebook these definitions live in __main__.
                            sys.modules["__main__"].SharedPreprocessor = namespace["SharedPreprocessor"]
                            sys.modules["__main__"].BaselineTree = namespace["BaselineTree"]
                            sys.modules["__main__"].Node = namespace["Node"]
                            with self.assertRaises(ValueError):
                                namespace["BaselineTree"](max_depth=0).fit(
                                    pd.DataFrame({"amount": [1, 2]}),
                                    np.array([0, 256], dtype=np.int16),
                                )
            finally:
                os.chdir(old_cwd)
            result = pd.read_csv(root / "output" / "submission_dtl.csv", dtype={"id": str})
            self.assertEqual(result["id"].tolist(), test["id"].tolist())
            self.assertEqual(result.columns.tolist(), ["id", "is_canceled"])
            self.assertTrue(result["is_canceled"].isin([0, 1]).all())
            self.assertEqual(namespace["selected_variant"], "CART")
            self.assertEqual(len(namespace["tree_scores"]), 3)
            self.assertEqual(len(namespace["comparison_results"]), 3)
            self.assertTrue((root / "output" / "final" / "model" / "dtl" / "bundle.pkl").exists())
            diagram = root / "output" / "final" / "model" / "dtl" / "tree_top_10.svg"
            self.assertTrue(diagram.exists())
            self.assertIn("<svg", diagram.read_text(encoding="utf-8"))
            self.assertIn("amount", diagram.read_text(encoding="utf-8"))
            render = namespace["render_tree_svg"]
            node = namespace["Node"]
            tree = node(0, 0.5, 8, 8.0, feature="amount", kind="numeric", threshold=3.0)
            tree.branches = {
                "le": node(0, 0.25, 4, 4.0, feature="channel", kind="binary_category", categories=frozenset({"a&b"})),
                "gt": node(1, 0.75, 4, 4.0),
            }
            tree.branches["le"].branches = {
                "left": node(0, 0.0, 2, 2.0),
                "right": node(1, 0.5, 2, 2.0),
            }
            limited = root / "one.svg"
            render(tree, 1, limited)
            text = limited.read_text(encoding="utf-8")
            self.assertIn("amount", text)
            self.assertNotIn("channel", text)
            self.assertIn("cut", text)
            render(tree, 2, limited)
            self.assertIn("channel", limited.read_text(encoding="utf-8"))
            self.assertIn("a&amp;b", limited.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                render(tree, 0, limited)
            tree.branches["le"].categories = frozenset({
                "Airline", "Complimentary", "Corporate", "Offline TA/TO", "Online TA",
            })
            render(tree, 2, limited)
            svg = ET.parse(limited).getroot()
            tag = "{http://www.w3.org/2000/svg}"
            boxes = [
                (float(box.get("x")), float(box.get("y")), float(box.get("width")), float(box.get("height")))
                for box in svg.findall(f"{tag}rect") if box.get("stroke")
            ]
            tops = {(x + width / 2, y) for x, y, width, height in boxes}
            bottoms = {(x + width / 2, y + height) for x, y, width, height in boxes}
            for edge in svg.findall(f"{tag}path"):
                start, end = edge.get("d").removeprefix("M ").split(" L ")
                self.assertIn(tuple(map(float, start.split())), bottoms)
                self.assertIn(tuple(map(float, end.split())), tops)
            for label in svg.findall(f"{tag}text"):
                x, y = float(label.get("x")), float(label.get("y"))
                self.assertTrue(any(left <= x <= left + width and top <= y <= top + height
                                    for left, top, width, height in boxes))
            self.assertIn("in {Airline, +4}", limited.read_text(encoding="utf-8"))
            self.assertIn("Complimentary", "".join(
                title.text or "" for title in svg.findall(f".//{tag}title")
            ))


if __name__ == "__main__":
    unittest.main()
