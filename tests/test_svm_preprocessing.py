import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


path = Path(__file__).resolve().parents[1] / "src/experiments/svm/preprocessing.py"
spec = importlib.util.spec_from_file_location("svm_preprocessing", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SVMPreprocessorCacheTest(unittest.TestCase):
    def test_same_rate_sum_does_not_reuse_training_labels_for_different_rows(self):
        prep = module.SVMPreprocessor()
        train = pd.DataFrame({"nightly_rate_idr": [10.0, 20.0], "days_before_arrival": [1, 2]})
        different = pd.DataFrame({"nightly_rate_idr": [20.0, 10.0], "days_before_arrival": [9, 9]})
        prep.is_fitted_ = True
        prep._fitted_frame_ = train
        prep._train_oof_cache_ = np.array([[0.0], [1.0]])

        np.testing.assert_array_equal(prep.transform(train.copy()), prep._train_oof_cache_)
        with patch.object(prep, "_build_base_blocks", side_effect=RuntimeError("different rows reached transform")):
            with self.assertRaisesRegex(RuntimeError, "different rows reached transform"):
                prep.transform(different)
