import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/experiments/logistic_regression'
sys.path.insert(0, str(SOURCE))

from experiments import (
    FINAL_CONFIGS,
    fit_final_models,
    fit_preprocessor,
    load_model_bundle,
    make_submissions,
    save_model_bundle,
    train_model,
    transform_features,
    verify_fresh_process,
)
from logistic_regression import LogisticRegressionScratch


class LogisticRegressionPipelineTest(unittest.TestCase):
    def test_sga_matches_two_examples_from_lecture(self):
        model = LogisticRegressionScratch(
            learning_rate=0.1, max_iter=1, reg_strength=0, random_state=42,
        ).fit(np.array([[52, 41], [62, 58]]), np.array([0, 1]))
        np.testing.assert_allclose(model.coef_, [-2.1, -1.2], atol=1e-12)
        self.assertAlmostEqual(model.intercept_, -0.05)

    def raw_sample(self):
        return pd.read_csv(ROOT / 'data/train.csv', nrows=40, dtype={'id': str})

    def test_validation_values_do_not_refit_preprocessing(self):
        raw = self.raw_sample().drop(columns=['id', 'is_canceled'])
        state = fit_preprocessor(raw)
        lower = state['clip_bounds']['lower'].copy()
        upper = state['clip_bounds']['upper'].copy()
        scaler = state['transformer'].named_transformers_['numeric']['scaler']
        mean = scaler.mean_.copy()
        categories = [
            values.copy() for values in
            state['transformer'].named_transformers_['categorical']['encoder'].categories_
        ]
        validation = raw.iloc[:3].copy()
        validation['nightly_rate_idr'] = 1e12
        validation['booking_channel'] = 'Unseen channel'
        transformed = transform_features(state, validation)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(len(transformed), len(validation))
        self.assertEqual(lower, state['clip_bounds']['lower'])
        self.assertEqual(upper, state['clip_bounds']['upper'])
        np.testing.assert_array_equal(mean, scaler.mean_)
        encoder = state['transformer'].named_transformers_['categorical']['encoder']
        for before, after in zip(categories, encoder.categories_):
            np.testing.assert_array_equal(before, after)

    def test_saved_parameters_predict_identically_in_fresh_process(self):
        sample = self.raw_sample()
        raw = sample.drop(columns=['id', 'is_canceled'])
        state = fit_preprocessor(raw)
        X = transform_features(state, raw)
        model = train_model(X, sample['is_canceled'].to_numpy(), FINAL_CONFIGS['l2'])
        expected = model.predict(X)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            bundle_path = folder / 'model_bundle.pkl'
            test_path = folder / 'test.csv'
            sample.drop(columns='is_canceled').to_csv(test_path, index=False)
            save_model_bundle(bundle_path, state, model, 'l2')
            restored_state, restored_model = load_model_bundle(bundle_path)
            np.testing.assert_array_equal(
                restored_model.predict(transform_features(restored_state, raw)), expected,
            )
            verify_fresh_process(bundle_path, test_path, expected)

    def test_outputs_are_consolidated_under_final_model_folder(self):
        sample = self.raw_sample()
        columns = sample.drop(columns=['id', 'is_canceled']).columns.tolist()
        test = sample.drop(columns='is_canceled')
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            test_path = folder / 'test.csv'
            test.to_csv(test_path, index=False)
            paths = fit_final_models(sample, columns, folder)
            checks, selected = make_submissions(
                paths, test, test_path, folder, 'elasticnet',
            )
            canonical = folder / 'final/model/logistic_regression'
            self.assertEqual(selected, canonical / 'submission.csv')
            for name in FINAL_CONFIGS:
                self.assertEqual(paths[name], canonical / name / 'model_bundle.pkl')
                self.assertEqual(
                    folder / checks.loc[name, 'csv'], canonical / name / 'submission.csv',
                )
                self.assertFalse(Path(checks.loc[name, 'csv']).is_absolute())
                self.assertEqual(checks.loc[name, 'rows'], len(test))
            self.assertEqual(checks.columns.tolist(), ['rows', 'csv'])
            self.assertEqual(
                selected.read_bytes(), (canonical / 'elasticnet/submission.csv').read_bytes(),
            )
            self.assertFalse((folder / 'logistic_regression').exists())
            self.assertFalse((folder / 'submission_logistic_regression.csv').exists())


if __name__ == '__main__':
    unittest.main()
