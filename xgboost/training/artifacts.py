"""Step 7: save and reload a trained run as a self-describing artifact.

    from training.artifacts import save_model_artifact, load_model_artifact
    save_model_artifact(model, features, out_dir='outputs/artifacts/<run_id>', ...)

What is saved, and why each piece is needed to reproduce a prediction:

    model.joblib        the fitted XGBRegressor
    run_info.json       everything needed to interpret the model, in one place:
                          - feature order (positional; get it wrong and predictions are silent garbage)
                          - hyperparameters and n_estimators as actually fitted
                          - the arm and outer fold the configuration came from
                          - target transform (log1p in, expm1 out) and the residual convention
                          - seeds
                          - partition cell assignments and the fold maps (so a run can be re-scored)
                          - library versions (xgboost/scikit-learn/pandas/numpy change defaults)
                          - a checksum of the training matrix shape and columns
    features.json       the same feature list on its own, for quick diffing

`load_model_artifact` refuses to return a model whose feature list does not match the caller's, because
that is the one failure mode that produces plausible-looking wrong numbers rather than an error.
"""
import json
import os
import platform
import sys

import numpy as np

import data

TARGET_TRANSFORM = 'log1p on fit, expm1 on predict'
RESIDUAL_CONVENTION = 'actual - predicted'


def _library_versions():
    import importlib.metadata as md
    out = {}
    for pkg in ['xgboost', 'scikit-learn', 'scikit-optimize', 'pandas', 'numpy', 'matplotlib', 'shap']:
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def save_model_artifact(model, features, out_dir=None, *, arm=None, best_fold=None,
                        params=None, n_estimators=None, partitions=None, fold_doc=None,
                        training_shape=None, extra=None, run_id=None):
    """Write the model plus its run_info.json. Returns the artifact directory."""
    import joblib

    out_dir = out_dir or data.out('artifacts', run_id or 'latest')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'model.joblib')
    joblib.dump(model, model_path)

    info = {
        'created': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
        'artifact_dir': os.path.abspath(out_dir),
        'model_file': 'model.joblib',
        'model_class': type(model).__name__,
        'arm': arm,
        'params_from_outer_fold': best_fold,
        'hyperparameters': params,
        'n_estimators': int(n_estimators) if n_estimators is not None else None,
        'feature_names': list(features),
        'n_features': len(features),
        'target_transform': TARGET_TRANSFORM,
        'residual_convention': RESIDUAL_CONVENTION,
        'random_state': getattr(model, 'random_state', None),
        'training_rows': int(training_shape[0]) if training_shape is not None else None,
        'training_columns': int(training_shape[1]) if training_shape is not None else None,
        'partitions': partitions,
        'fold_maps': (fold_doc or {}).get('folds') if fold_doc else None,
        'fold_partitions_file': os.path.relpath(data.FOLD_JSON, data.ROOT),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'libraries': _library_versions(),
        },
    }
    if extra:
        info.update(extra)
    with open(os.path.join(out_dir, 'run_info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh, indent=2, sort_keys=True, default=str)
    with open(os.path.join(out_dir, 'features.json'), 'w', encoding='utf-8') as fh:
        json.dump({'features': list(features), 'n_features': len(features)}, fh, indent=2)
    return out_dir


def load_model_artifact(out_dir, expect_features=None):
    """Load (model, info). Raises if the artifact's feature list does not match `expect_features`."""
    import joblib

    info_path = os.path.join(out_dir, 'run_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"no run_info.json in {out_dir}. Build the artifact with training/artifacts.py "
            f"(run_pipeline.py --stage save), or point out_dir at an existing artifact.")
    with open(info_path, encoding='utf-8') as fh:
        info = json.load(fh)
    model = joblib.load(os.path.join(out_dir, info.get('model_file', 'model.joblib')))
    if expect_features is not None:
        saved = list(info['feature_names'])
        if list(expect_features) != saved:
            raise ValueError(
                'feature mismatch between the artifact and the caller; refusing to predict.\n'
                f'  artifact ({len(saved)}): {saved}\n'
                f'  caller   ({len(expect_features)}): {list(expect_features)}')
    return model, info


def predict_raw(model, X, info=None):
    """Predictions in raw ug/m3 (the model is fitted on log1p)."""
    return np.expm1(model.predict(X))
