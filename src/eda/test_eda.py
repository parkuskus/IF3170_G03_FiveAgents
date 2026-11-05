import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.eda.eda import default_output_dir, generate, histogram_limit


class EDATest(unittest.TestCase):
    def test_default_output_is_outside_source_tree(self):
        self.assertEqual(default_output_dir(), Path(__file__).resolve().parents[2] / "output" / "eda")

    def test_rare_nonzero_values_remain_visible_when_p99_is_zero(self):
        self.assertEqual(histogram_limit(pd.Series([0] * 100 + [13])), 13)

    def test_report_keeps_train_and_test_separate_and_flags_bad_rows(self):
        train = pd.DataFrame({
            "id": ["123e4567", "b", "b", "c"],
            "num_adults": [2, 2, 2, 0],
            "num_children": [None, 0, 0, 0],
            "weekend_nights": [1, 1, 1, 0],
            "weekday_nights": [2, 2, 2, 0],
            "arrival_year": [2024] * 4,
            "arrival_month": [2, 2, 2, 2],
            "arrival_day": [29, 29, 29, 30],
            "nightly_rate_idr": [1000000, 1000000, 1000000, 0],
            "meal_package": ["Breakfast"] * 4,
            "is_canceled": [0, 1, 1, 1],
        })
        test = train.drop(columns="is_canceled").iloc[[0, 3]].copy()
        test.loc[test.index[0], "num_children"] = 0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generate(train, test, output)
            report = (output / "laporan.md").read_text(encoding="utf-8")
            self.assertIn("4 baris latih dan 2 baris test", report)
            self.assertIn("`num_children` | 1 (25.00%) | 0 (0.00%)", report)
            self.assertIn("ID duplikat pada train | 1", report)
            self.assertIn("ID yang muncul di kedua berkas | 2", report)
            self.assertIn("Tanggal kedatangan tidak sah | 1 | 1", report)
            self.assertTrue((output / "distribusi_target.png").is_file())
            self.assertIn("## Kedatangan menurut tahun dan bulan", report)
            self.assertTrue((output / "kedatangan_bulanan.png").is_file())

    def test_submission_template_preserves_test_ids_and_expected_columns(self):
        train = pd.DataFrame({"id": ["t1", "t2"], "arrival_year": [2024, 2024],
                              "arrival_month": [1, 2], "arrival_day": [1, 1],
                              "weekend_nights": [1, 1], "weekday_nights": [1, 1],
                              "num_adults": [2, 2], "num_children": [0, 0],
                              "nightly_rate_idr": [100, 100], "is_canceled": [0, 1]})
        test = train.drop(columns="is_canceled").copy()
        test["id"] = ["123e4567", "z"]
        sample = pd.DataFrame({"id": ["123e4567", "z"], "is_canceled": [0, 0]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generate(train, test, output, sample)
            report = (output / "laporan.md").read_text(encoding="utf-8")
            self.assertIn("Header sesuai `id,is_canceled` | ya", report)
            self.assertIn("Urutan ID sama dengan test | ya", report)


if __name__ == "__main__":
    unittest.main()
