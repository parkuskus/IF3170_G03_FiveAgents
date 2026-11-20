import argparse
import hashlib
import inspect
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from logistic_regression import LogisticRegressionScratch
from preprocessing import FEATURE_CONFIG, fit_preprocessing, transform_preprocessing


VARIANT_NAMES = {'l2': 'L2', 'l1': 'L1', 'elasticnet': 'Elastic Net'}
COMMON_CONFIG = {
    'learning_rate': 0.001,
    'max_iter': 200,
    'reg_strength': 0.0001,
    'threshold': 0.47,
    'random_state': 42,
}
FINAL_CONFIGS = {
    name: {**COMMON_CONFIG, 'l1_ratio': ratio}
    for name, ratio in [('l2', 0.0), ('l1', 1.0), ('elasticnet', 0.5)]
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_hashes(data_dir):
    return {
        'train_sha256': file_sha256(Path(data_dir) / 'train.csv'),
        'test_sha256': file_sha256(Path(data_dir) / 'test.csv'),
    }


def make_contract_split(y, seed=42):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    valid = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(y == label))[
            :int(np.ceil(0.2 * np.sum(y == label)))
        ] for label in (0, 1)
    ]))
    return np.setdiff1d(np.arange(len(y)), valid), valid


def fit_preprocessor(raw):
    transformer, bounds, _ = fit_preprocessing(raw, FEATURE_CONFIG)
    return {
        'transformer': transformer,
        'clip_bounds': bounds,
        'feature_config': deepcopy(FEATURE_CONFIG),
        'feature_columns': raw.columns.tolist(),
    }


def transform_features(state, raw):
    return transform_preprocessing(
        raw[state['feature_columns']], state['feature_config'],
        state['transformer'], state['clip_bounds'],
    )


def train_model(X, y, config):
    parameters = {key: value for key, value in config.items() if key != 'penalty'}
    with threadpool_limits(limits=1):
        return LogisticRegressionScratch(**parameters).fit(X, y)


def predict_model(model, X):
    return model.predict(X)


def validation_metrics(y, prediction):
    report = classification_report(
        y, prediction, labels=[0, 1], output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        'macro_f1': float(f1_score(
            y, prediction, labels=[0, 1], average='macro', zero_division=0,
        )),
        **{
            f'{metric}_{label}': float(report[str(label)][metric])
            for label in (0, 1) for metric in ('precision', 'recall', 'f1-score')
        },
        'true_negative': int(matrix[0, 0]),
        'false_positive': int(matrix[0, 1]),
        'false_negative': int(matrix[1, 0]),
        'true_positive': int(matrix[1, 1]),
    }


def evaluate_variants(X_train, y_train, X_valid, y_valid):
    models, predictions, rows = {}, {}, []
    for name, config in FINAL_CONFIGS.items():
        started = time.perf_counter()
        model = train_model(X_train, y_train, config)
        pred = model.predict(X_valid)
        models[name], predictions[name] = model, pred
        rows.append({
            'variant': name, **validation_metrics(y_valid, pred),
            'training_seconds': time.perf_counter() - started,
        })
    return models, predictions, pd.DataFrame(rows).set_index('variant')


def cross_validate(raw, y):
    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
    probabilities = {name: np.empty(len(y)) for name in FINAL_CONFIGS}
    fold_rows = []
    for fold, (a, v) in enumerate(splitter.split(raw, y), start=1):
        state = fit_preprocessor(raw.iloc[a])
        X_train = transform_features(state, raw.iloc[a])
        X_valid = transform_features(state, raw.iloc[v])
        for name, config in FINAL_CONFIGS.items():
            model = train_model(X_train, y[a], config)
            probabilities[name][v] = model.predict_proba(X_valid)
            fold_rows.append({
                'variant': name, 'fold': fold,
                'macro_f1': validation_metrics(y[v], model.predict(X_valid))['macro_f1'],
            })
    scores = pd.Series({
        name: validation_metrics(y, p >= FINAL_CONFIGS[name]['threshold'])['macro_f1']
        for name, p in probabilities.items()
    }, name='macro_f1_internal_cv')
    return pd.DataFrame(fold_rows), scores


def build_comparison_model(variant, n_train):
    config = FINAL_CONFIGS[variant]
    params = {
        'C': 1 / (n_train * config['reg_strength']),
        'solver': 'lbfgs' if variant == 'l2' else 'saga',
        'max_iter': 10000,
        'tol': 1e-4 if variant == 'l1' else 1e-7,
        'random_state': 42,
    }
    parameter = inspect.signature(LogisticRegression).parameters.get('penalty')
    if parameter is None or parameter.default == 'deprecated':
        params['l1_ratio'] = config['l1_ratio']
    else:
        params['penalty'] = variant
        if variant == 'elasticnet':
            params['l1_ratio'] = config['l1_ratio']
    return LogisticRegression(**params)


def compare_sklearn(X_train, y_train, X_valid, y_valid):
    predictions, rows = {}, []
    for name, config in FINAL_CONFIGS.items():
        model = build_comparison_model(name, len(y_train))
        with threadpool_limits(limits=1):
            model.fit(X_train, y_train)
        one = int(np.flatnonzero(model.classes_ == 1)[0])
        prediction = (
            model.predict_proba(X_valid)[:, one] >= config['threshold']
        ).astype(np.int64)
        predictions[name] = prediction
        rows.append({'variant': name, **validation_metrics(y_valid, prediction)})
    return predictions, pd.DataFrame(rows).set_index('variant')


def select_variant(scores):
    return sorted(scores.index, key=lambda name: (-scores.loc[name, 'macro_f1'], name))[0]


def save_model_bundle(path, preprocessor, model, variant, metadata=None):
    bundle = {
        'variant': variant, 'config': deepcopy(FINAL_CONFIGS[variant]),
        'preprocessing': preprocessor,
        'model_state': {
            'coef': model.coef_.copy(), 'intercept': model.intercept_,
            'n_features_in': model.n_features_in_, 'n_iter': model.n_iter_,
            'loss_history': model.loss_history_,
        },
        'versions': {
            'python': sys.version.split()[0],
            **{name: version(name) for name in ['numpy', 'pandas', 'scikit-learn']},
        },
        'optimizer': 'stochastic_gradient_ascent',
        **(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(bundle, handle)
    return bundle


def load_model_bundle(path):
    with Path(path).open('rb') as handle:
        bundle = pickle.load(handle)
    model = LogisticRegressionScratch(**bundle['config'])
    state = bundle['model_state']
    model.coef_ = state['coef']
    model.intercept_ = state['intercept']
    model.n_features_in_ = state['n_features_in']
    model.n_iter_ = state['n_iter']
    model.loss_history_ = state['loss_history']
    return bundle['preprocessing'], model


def fit_final_models(train, feature_columns, output_dir, metadata=None):
    raw = train[feature_columns]
    y = train['is_canceled'].to_numpy(dtype=np.int64)
    preprocessor = fit_preprocessor(raw)
    X = transform_features(preprocessor, raw)
    paths = {}
    for name, config in FINAL_CONFIGS.items():
        model = train_model(X, y, config)
        path = Path(output_dir) / 'final/model/logistic_regression' / name / 'model_bundle.pkl'
        save_model_bundle(path, preprocessor, model, name, metadata)
        paths[name] = path
    return paths


def verify_fresh_process(bundle_path, test_path, expected):
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import pandas as pd
from experiments import load_model_bundle, transform_features
prep, model = load_model_bundle(sys.argv[2])
raw = pd.read_csv(sys.argv[3], dtype={'id': str})
np.save(sys.argv[4], model.predict(transform_features(prep, raw)))
'''
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'prediction.npy'
        subprocess.run([
            sys.executable, '-c', script, str(Path(__file__).resolve().parent),
            str(bundle_path), str(test_path), str(path),
        ], check=True, capture_output=True, text=True)
        np.testing.assert_array_equal(np.load(path), expected)


def make_submissions(paths, test, test_path, output_dir, selected):
    folder = Path(output_dir) / 'final/model/logistic_regression'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in paths.items():
        prep, model = load_model_bundle(path)
        prediction = model.predict(transform_features(prep, test))
        submission = pd.DataFrame({
            'id': test['id'].to_numpy(), 'is_canceled': prediction,
        })
        csv_path = folder / name / 'submission.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(csv_path, index=False)
        written = pd.read_csv(csv_path, dtype={'id': str})
        pd.testing.assert_frame_equal(written, submission)
        if not written['id'].equals(test['id'].reset_index(drop=True)):
            raise ValueError('Urutan ID submission berubah.')
        if not np.issubdtype(prediction.dtype, np.integer):
            raise ValueError('Prediksi harus berupa integer.')
        if len(prediction) != len(test) or not np.isin(prediction, [0, 1]).all():
            raise ValueError('Prediksi test harus satu label 0/1 per baris.')
        verify_fresh_process(path, test_path, prediction)
        rows.append({
            'variant': name, 'rows': len(written),
            'csv': str(csv_path.relative_to(Path(output_dir))),
        })
        print(f'Submission {VARIANT_NAMES[name]} disimpan.')
    selected_path = folder / 'submission.csv'
    shutil.copyfile(folder / selected / 'submission.csv', selected_path)
    return pd.DataFrame(rows).set_index('variant'), selected_path


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description='Pipeline final tiga regularisasi LR.')
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--output-dir', type=Path, default=root / 'output')
    args = parser.parse_args()
    train = pd.read_csv(args.data_dir / 'train.csv', dtype={'id': str})
    test = pd.read_csv(args.data_dir / 'test.csv', dtype={'id': str})
    columns = train.drop(columns=['id', 'is_canceled']).columns.tolist()
    y = train['is_canceled'].to_numpy(dtype=np.int64)
    a, v = make_contract_split(y)
    raw_train, raw_valid = train.iloc[a][columns], train.iloc[v][columns]
    state = fit_preprocessor(raw_train)
    X, V = transform_features(state, raw_train), transform_features(state, raw_valid)
    _, cv = cross_validate(raw_train, y[a])
    _, _, scores = evaluate_variants(X, y[a], V, y[v])
    _, reference = compare_sklearn(X, y[a], V, y[v])
    summary = scores.join(cv).join(reference.add_prefix('sklearn_'))
    folder = args.output_dir / 'final/model/logistic_regression'
    folder.mkdir(parents=True, exist_ok=True)
    selected = select_variant(scores)
    hashes = data_hashes(args.data_dir)
    paths = fit_final_models(train, columns, args.output_dir, {
        'validation': summary.to_dict(orient='index'), 'data_hashes': hashes,
    })
    checks, selected_path = make_submissions(
        paths, test, args.data_dir / 'test.csv', args.output_dir, selected,
    )
    summary.join(checks).to_csv(folder / 'validation.csv')
    print(summary.to_string())
    print(checks.to_string())
    print('Model terpilih:', VARIANT_NAMES[selected])
    print('Submission:', selected_path.relative_to(args.output_dir))


if __name__ == '__main__':
    main()
