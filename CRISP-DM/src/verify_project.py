"""Replay and verify the Telco notebook with the invoking Python environment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from importlib.metadata import version

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

from telco import FEATURES, NUMERIC, clean_features

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / 'notebooks/telco_churn_crisp_dm.ipynb'


def code_hash(notebook):
    source = '\n\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    return hashlib.sha256(source.encode()).hexdigest()


def execute_notebook():
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    # Override the registered kernel executable so replay uses this Python.
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    try:
        NotebookClient(notebook, km=manager, timeout=300,
                       resources={'metadata': {'path': str(NOTEBOOK.parent)}}).execute()
    finally:
        nbformat.write(notebook, NOTEBOOK)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


def verify():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert [c.execution_count for c in cells] == list(range(1, len(cells) + 1))
    for cell in cells:
        assert not [o for o in cell.outputs if o.output_type == 'error']
        assert not [o for o in cell.outputs if o.output_type == 'stream' and o.name == 'stderr']
    manifest = json.loads((PROJECT / 'outputs/run_manifest.json').read_text())
    for package, recorded in manifest['packages'].items():
        assert version(package) == recorded, f'Environment differs for {package}; replay before comparing artifacts.'
    metrics = json.loads((PROJECT / 'reports/metrics.json').read_text())
    raw_path = PROJECT / 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == manifest['sha256']
    assert hashlib.sha256((PROJECT / 'src/telco.py').read_bytes()).hexdigest() == manifest['normalization_source_sha256']
    assert code_hash(notebook) == manifest['notebook_code_sha256']
    model_path = PROJECT / 'outputs/churn_pipeline.joblib'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == manifest['model_sha256']
    # Load only locally generated, trusted artifacts. A checksum is not authentication.
    model = joblib.load(model_path)
    raw = pd.read_csv(raw_path)
    train_rows = manifest['train_source_rows']
    test_rows = manifest['test_source_rows']
    assert not set(train_rows) & set(test_rows)
    assert set(train_rows) | set(test_rows) == set(raw.index)
    assert raw.customerID.is_unique
    features = raw[FEATURES]
    training = clean_features(features.loc[train_rows])
    numeric = model.named_steps['prepare'].named_transformers_['numeric']
    np.testing.assert_allclose(numeric.named_steps['impute'].statistics_, training[NUMERIC].median())
    test = features.loc[test_rows]
    probability = model.predict_proba(test)[:, 1]
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    predictions = pd.read_csv(PROJECT / 'outputs/test_predictions.csv')
    np.testing.assert_array_equal(predictions.source_row, test_rows)
    np.testing.assert_allclose(predictions.churn_score, probability, atol=1e-14, rtol=1e-14)
    labels = raw.loc[test_rows, 'Churn'].map({'Yes': 1, 'No': 0})
    predicted = probability >= manifest['threshold']
    for key, score in {'roc_auc': roc_auc_score(labels, probability),
                       'average_precision': average_precision_score(labels, probability),
                       'precision': precision_score(labels, predicted),
                       'recall': recall_score(labels, predicted)}.items():
        assert np.isclose(metrics[key], score), key
    cv = pd.read_csv(PROJECT / 'reports/cross_validation.csv')
    assert cv.loc[cv.average_precision_mean.idxmax(), 'model'] == manifest['selected_model']
    folds = pd.read_csv(PROJECT / 'reports/fold_preprocessing_audit.csv')
    assert len(folds) == len(cv) * 5 and folds.training_only_transforms.all()
    for key, interval in metrics['bootstrap_95_intervals'].items():
        assert 0 <= interval['lower'] <= interval['upper'] <= 1, key
    # Exercise the public input contract, including normalization and imputation.
    sample = test.head(5).copy()
    np.testing.assert_allclose(model.predict_proba(sample[FEATURES[::-1]]), model.predict_proba(sample), rtol=0, atol=0)
    normalized = clean_features(sample)
    np.testing.assert_allclose(model.predict_proba(normalized), model.predict_proba(sample), rtol=0, atol=0)
    missing = sample.copy()
    missing.loc[:, 'TotalCharges'] = ' '
    missing.loc[:, 'Partner'] = np.nan
    assert np.isfinite(model.predict_proba(missing)).all()
    bad_inputs = [sample.drop(columns='tenure'), sample.assign(Churn='Yes'),
                  sample.assign(MonthlyCharges=-1), sample.assign(TotalCharges='corrupt'),
                  sample.assign(MonthlyCharges=np.inf), sample.assign(SeniorCitizen=7),
                  sample.assign(Contract='month-to-month typo'), sample.assign(tenure=0.5),
                  sample.assign(PhoneService='No', MultipleLines='Yes'),
                  sample.assign(InternetService='No', OnlineSecurity='Yes')]
    for bad in bad_inputs:
        try:
            model.predict_proba(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('Invalid input was accepted')
    demo = pd.read_csv(PROJECT / 'reports/prediction_demo.csv')
    np.testing.assert_allclose(demo.churn_score, probability[:5], atol=1e-14, rtol=1e-14)
    assert metrics['acceptance_checks'] == {
        'ap_above_prevalence': metrics['average_precision'] > metrics['test_prevalence'],
        'roc_auc_at_least_080': metrics['roc_auc'] >= 0.80,
        'recall_at_least_060': metrics['recall'] >= 0.60,
    }
    assert json.loads((PROJECT / 'reports/decision.json').read_text())['operational_release'] == 'not approved'
    print(f'PASS: {len(cells)} executed code cells; dataset/code/model hashes; training/test isolation;')
    print(f'PASS: {len(folds)} fitted-fold preprocessing checks; independent reload and metric recomputation;')
    print(f'PASS: raw/reordered/normalized/missing input scoring and {len(bad_inputs)} invalid-input rejections.')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Execute the notebook before verification.')
    args = parser.parse_args()
    if args.execute:
        execute_notebook()
    verify()


if __name__ == '__main__':
    main()
