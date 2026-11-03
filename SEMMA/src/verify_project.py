"""Execute and verify the time-aware Seoul Bike SEMMA notebook."""
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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, root_mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
from seoul import (SHA256, INPUTS, DESIGN, WEATHER, load_data, build_supervised, design_matrix, make_day_inputs)

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / 'notebooks/seoul_bike_semma.ipynb'


def execute():
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    notebook = nbformat.read(NOTEBOOK, as_version=4)
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
    assert [c.execution_count for c in cells] == list(range(1,len(cells)+1))
    for cell in cells:
        assert not [o for o in cell.outputs if o.output_type == 'error']
        assert not [o for o in cell.outputs if o.output_type == 'stream' and o.name == 'stderr']
    manifest = json.loads((PROJECT/'outputs/run_manifest.json').read_text())
    for package, recorded in manifest['packages'].items():
        assert version(package) == recorded, f'Package version differs: {package}'
    assert hashlib.sha256((PROJECT/'src/seoul.py').read_bytes()).hexdigest() == manifest['source_sha256']
    source = '\n\n'.join(c.source for c in cells)
    assert hashlib.sha256(source.encode()).hexdigest() == manifest['notebook_code_sha256']
    csv = PROJECT/'data/raw/SeoulBikeData.csv'
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == SHA256 == manifest['dataset_sha256']
    raw = load_data(csv)
    data = build_supervised(raw)
    cutoff = pd.Timestamp(manifest['split']['test_start'])
    dev, test = data[data.timestamp < cutoff], data[data.timestamp >= cutoff]
    assert dev.timestamp.max() < test.timestamp.min()
    assert len(test) == 1344 and len(dev)+len(test) == 8592
    for lag in [24,168]:
        assert (data.timestamp-pd.Timedelta(hours=lag) < data.timestamp.dt.normalize()).all()
        expected = raw.set_index('timestamp').loc[pd.DatetimeIndex(data.timestamp-pd.Timedelta(hours=lag)), 'rentals']
        np.testing.assert_array_equal(data[f'rentals_lag{lag}'],expected)
    for column in WEATHER.values():
        expected = raw.set_index('timestamp').loc[pd.DatetimeIndex(data.timestamp-pd.Timedelta(hours=24)),column]
        np.testing.assert_array_equal(data[f'{column}_lag24'],expected)
    # A future target change must not alter any same-day inputs (day-ahead batch).
    day = cutoff + pd.Timedelta(days=2)
    changed = raw.copy()
    changed.loc[changed.timestamp >= day,'rentals'] += 10000
    changed_data = build_supervised(changed)
    pd.testing.assert_frame_equal(data.loc[data.timestamp < day+pd.Timedelta(days=1), INPUTS],
                                  changed_data.loc[changed_data.timestamp < day+pd.Timedelta(days=1), INPUTS])
    model_path = PROJECT/'outputs/demand_pipeline.joblib'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == manifest['model_sha256']
    model = joblib.load(model_path)  # Trusted locally generated artifact only.
    prediction = model.predict(test[INPUTS])
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    saved = pd.read_csv(PROJECT/'outputs/test_predictions.csv')
    np.testing.assert_allclose(prediction,saved.prediction,rtol=1e-13,atol=1e-12)
    pd.testing.assert_series_equal(pd.to_datetime(saved.timestamp),test.timestamp.reset_index(drop=True),check_names=False)
    np.testing.assert_array_equal(saved.actual,test.rentals)
    np.testing.assert_array_equal(saved.previous_day,test.rentals_lag24)
    metrics = json.loads((PROJECT/'reports/metrics.json').read_text())
    for key,value in {'mae':mean_absolute_error(test.rentals,prediction),
                      'rmse':root_mean_squared_error(test.rentals,prediction),
                      'r2':r2_score(test.rentals,prediction)}.items():
        assert np.isclose(metrics['selected'][key],value),key
    comparison = pd.read_csv(PROJECT/'reports/model_comparison.csv')
    assert comparison.loc[comparison.mae_mean.idxmin(),'model'] == manifest['model']
    splits = pd.read_csv(PROJECT/'reports/time_splits.csv')
    assert len(splits) == 3
    assert (pd.to_datetime(splits.train_end)+pd.Timedelta(hours=24) < pd.to_datetime(splits.validation_start)).all()
    folds = pd.read_csv(PROJECT/'reports/validation_folds.csv')
    assert len(folds) == 9 and folds.train_only_preprocessing_verified.all()
    if 'impute' in model.named_steps:
        design = design_matrix(dev[INPUTS])
        np.testing.assert_allclose(model.named_steps['impute'].statistics_,design.median())
        np.testing.assert_allclose(model.named_steps['scale'].mean_,design.mean(),atol=1e-12)
    examples = pd.read_csv(PROJECT/'outputs/example_profiles.csv')
    demo = pd.read_csv(PROJECT/'reports/prediction_demo.csv')
    np.testing.assert_allclose(model.predict(examples),demo.predicted_rentals,rtol=1e-13,atol=1e-12)
    np.testing.assert_allclose(model.predict(examples[INPUTS[::-1]]),model.predict(examples),rtol=0,atol=0)
    if 'impute' in model.named_steps:
        missing = examples.copy()
        missing['temperature_lag24'] = np.nan
        assert np.isfinite(model.predict(missing)).all()
    invalid = [examples.assign(rentals=1),examples.drop(columns='rentals_lag24'),
               examples.assign(holiday='Unknown'),examples.assign(rentals_lag24=-1),
               examples.assign(temperature_lag24=np.inf),examples.assign(timestamp='not-a-date'),
               examples.assign(humidity_lag24=101), examples.assign(rentals_lag24=np.nan),
               examples.assign(rentals_lag168=1.5)]
    for bad in invalid:
        try:
            model.predict(bad)
        except (ValueError,TypeError):
            pass
        else:
            raise AssertionError('Invalid inference input accepted')
    # Recompute all comparator metrics, slice contributions, and the paired interval.
    def errors(actual, predicted):
        return {'mae': float(mean_absolute_error(actual, predicted)),
                'rmse': float(root_mean_squared_error(actual, predicted)),
                'r2': float(r2_score(actual, predicted)) if np.unique(actual).size > 1 else None,
                'rmsle': float(root_mean_squared_log_error(actual, predicted))}
    for name, values in [('selected', prediction), ('previous_day', test.rentals_lag24.to_numpy())]:
        for key, computed in errors(test.rentals, values).items():
            assert np.isclose(metrics[name][key], computed), (name,key)
    signed_difference = np.abs(saved.actual-saved.prediction)-np.abs(saved.actual-saved.previous_day)
    assert np.isclose(metrics['mae_difference'],signed_difference.mean())
    assert np.isclose(metrics['mae_reduction_percent'],100*(1-metrics['selected']['mae']/metrics['previous_day']['mae']))
    slices = pd.read_csv(PROJECT/'reports/error_slices.csv')
    for row in slices.itertuples():
        labels = pd.to_datetime(saved.timestamp).dt.strftime('%Y-%m') if row.field == 'month' else saved.functioning
        mask = labels.eq(row.value)
        assert row.hours == int(mask.sum())
        for prefix, values in [('',saved.prediction[mask]),('baseline_',saved.previous_day[mask])]:
            for key, computed in errors(saved.actual[mask], values).items():
                recorded = getattr(row,prefix+key)
                assert pd.isna(recorded) if computed is None else np.isclose(recorded,computed)
        assert np.isclose(row.mae_difference,signed_difference[mask].mean())
        assert np.isclose(row.weighted_mae_difference_contribution,mask.mean()*signed_difference[mask].mean())
    np.testing.assert_array_equal(saved.functioning, test.functioning)
    for _, group in slices.groupby('field'):
        assert group.hours.sum() == len(test)
        assert np.isclose(group.weighted_mae_difference_contribution.sum(),metrics['mae_difference'])
    daily = pd.DataFrame({'timestamp':pd.to_datetime(saved.timestamp),
                          'absolute_error':np.abs(saved.actual-saved.prediction),
                          'baseline_absolute_error':np.abs(saved.actual-saved.previous_day)}).set_index('timestamp').resample('D').mean()
    recorded_daily = pd.read_csv(PROJECT/'reports/daily_errors.csv',index_col=0,parse_dates=True)
    np.testing.assert_allclose(daily,recorded_daily)
    weekly = (daily.absolute_error-daily.baseline_absolute_error).to_numpy().reshape(8,7).mean(axis=1)
    bootstrap = np.random.default_rng(manifest['bootstrap']['seed']).choice(weekly,size=(1000,8),replace=True).mean(axis=1)
    np.testing.assert_allclose([metrics['paired_week_block_95_interval'][k] for k in ['lower','upper']],np.quantile(bootstrap,[.025,.975]))
    for row in comparison.itertuples():
        score_rows = folds[folds.model.eq(row.model)]
        for metric in ['mae','rmse','r2','rmsle']:
            assert np.isclose(getattr(row,metric+'_mean'),score_rows[metric].mean())
            assert np.isclose(getattr(row,metric+'_std'),score_rows[metric].std())
        baseline_mae = comparison.loc[comparison.model.eq('previous_day'),'mae_mean'].iloc[0]
        assert np.isclose(row.mae_skill_vs_persistence,1-row.mae_mean/baseline_mae)
    reconstructed = TimeSeriesSplit(n_splits=3,test_size=28*24,gap=24)
    for row, (fit,validation) in zip(splits.itertuples(),reconstructed.split(dev)):
        assert row.train_rows == len(fit) and row.validation_rows == len(validation)
        assert pd.Timestamp(row.train_end) == dev.timestamp.iloc[fit].max()
        assert pd.Timestamp(row.validation_start) == dev.timestamp.iloc[validation].min()
        assert pd.Timestamp(row.validation_end) == dev.timestamp.iloc[validation].max()
    assert pd.Timestamp(manifest['split']['development_end'])+pd.Timedelta(hours=1) == cutoff
    assert manifest['cv']['final_fit_gap_hours'] == 0
    origin = cutoff
    history = raw[raw.timestamp < origin]
    batch = make_day_inputs(history,origin,examples.holiday.iloc[0])
    pd.testing.assert_frame_equal(batch,test.head(24)[INPUTS].reset_index(drop=True),check_dtype=False)
    np.testing.assert_allclose(model.predict(batch),prediction[:24],atol=1e-12,rtol=1e-13)
    for bad_history in [raw.iloc[::-1],raw.drop(index=100),raw.assign(timestamp=raw.timestamp+pd.Timedelta(minutes=30))]:
        try:
            build_supervised(bad_history)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid lag history accepted')
    for bad_history in [raw,history.iloc[:-1]]:
        try:
            make_day_inputs(bad_history,origin,examples.holiday.iloc[0])
        except ValueError:
            pass
        else:
            raise AssertionError('Future or incomplete as-of history accepted')
    assessment = json.loads((PROJECT/'reports/assessment.json').read_text())
    assert assessment['development_selected_candidate'] == manifest['model']
    assert assessment['test_primary_metric_improved'] == (metrics['selected']['mae'] < metrics['previous_day']['mae'])
    assert assessment['ridge_promotion_supported'] is False
    assert assessment['test_primary_metric_improved'] is False
    print('PASS: all candidate/comparator test metrics, paired slices, validation means, bootstrap and assessment recomputed.')
    print('PASS: 24-hour as-of scoring batch; three invalid lag histories and two invalid as-of histories rejected.')
    print(f'PASS: {len(cells)} executed code cells, artifact hashes and saved-model inference.')
    print('PASS: chronological split, all causal lag alignments, future-target perturbation, nine validation fits and metrics.')
    print(f'PASS: saved/reordered inputs and {len(invalid)} invalid-input rejections.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if arguments.execute:
        execute()
    verify()
