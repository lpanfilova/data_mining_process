"""Replay and verify the nine-step Mall Customers KDD study."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from importlib.metadata import version

import joblib
import nbformat
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from mall import FEATURES, SHA256, make_pipeline, assign_customers

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / 'notebooks/mall_customers_kdd.ipynb'


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
    headings = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type=='markdown' and re.match(r'^## [1-9]\.', c.source)]
    assert [int(re.search(r'\d',h).group()) for h in headings] == list(range(1,10))
    cells = [c for c in notebook.cells if c.cell_type=='code']
    assert [c.execution_count for c in cells] == list(range(1,len(cells)+1))
    for cell in cells:
        assert not [o for o in cell.outputs if o.output_type=='error']
        assert not [o for o in cell.outputs if o.output_type=='stream' and o.name=='stderr']
    manifest = json.loads((PROJECT/'outputs/run_manifest.json').read_text())
    for package,recorded in manifest['packages'].items():
        assert version(package)==recorded,package
    csv = PROJECT/'data/raw/Mall_Customers.csv'
    assert hashlib.sha256(csv.read_bytes()).hexdigest()==SHA256==manifest['dataset_sha256']
    assert hashlib.sha256((PROJECT/'src/mall.py').read_bytes()).hexdigest()==manifest['source_sha256']
    source = '\n\n'.join(c.source for c in cells)
    assert hashlib.sha256(source.encode()).hexdigest()==manifest['notebook_code_sha256']
    model_path = PROJECT/'outputs/customer_segments.joblib'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==manifest['model_sha256']
    model = joblib.load(model_path)  # Trusted locally generated artifact only.
    raw = pd.read_csv(csv)
    X = raw[FEATURES]
    scaler,estimator = model.named_steps['scale'],model.named_steps['kmeans']
    np.testing.assert_allclose(scaler.mean_,X.mean())
    np.testing.assert_allclose(scaler.var_,X.var(ddof=0))
    scaled = scaler.transform(X)
    labels = model.predict(X)
    assignments = pd.read_csv(PROJECT/'outputs/cluster_assignments.csv')
    pd.testing.assert_frame_equal(assignments[raw.columns],raw)
    np.testing.assert_array_equal(assignments.cluster,labels)
    np.testing.assert_allclose(assignments.silhouette,silhouette_samples(scaled,labels))
    selection = pd.read_csv(PROJECT/'reports/cluster_selection.csv')
    assert selection.k.tolist()==list(range(1,11))
    np.testing.assert_array_equal(selection.eligible,(selection.k>=2)&(selection.smallest_cluster>=10))
    eligible = selection[selection.eligible].sort_values(['silhouette','k'],ascending=[False,True])
    k = int(eligible.iloc[0].k)
    assert k==manifest['selected_k']==estimator.n_clusters
    assert estimator.n_init==20 and estimator.random_state==42
    metrics = json.loads((PROJECT/'reports/metrics.json').read_text())
    for key,value in {'silhouette':silhouette_score(scaled,labels),
                      'davies_bouldin':davies_bouldin_score(scaled,labels),
                      'calinski_harabasz':calinski_harabasz_score(scaled,labels),
                      'inertia':np.square(model.transform(X).min(axis=1)).sum()}.items():
        assert np.isclose(metrics[key],value),key
        assert np.isclose(selection.loc[selection.k.eq(k),key].iloc[0],value),key
    centroids = pd.read_csv(PROJECT/'reports/centroids.csv',index_col=0)
    np.testing.assert_allclose(centroids,scaler.inverse_transform(estimator.cluster_centers_))
    grouped = raw[FEATURES].assign(cluster=labels).groupby('cluster').mean()
    np.testing.assert_allclose(centroids,grouped,atol=1e-10)
    profiles = pd.read_csv(PROJECT/'reports/cluster_profiles.csv',index_col=0)
    np.testing.assert_array_equal(profiles.customers,np.bincount(labels,minlength=k))
    np.testing.assert_allclose(profiles.income_mean,grouped[FEATURES[0]])
    np.testing.assert_allclose(profiles.score_mean,grouped[FEATURES[1]])
    stability = pd.read_csv(PROJECT/'reports/stability.csv')
    assert len(stability)==40
    rng = np.random.default_rng(42)
    cluster_checks = pd.read_csv(PROJECT/'reports/cluster_stability.csv')
    k_candidates = pd.read_csv(PROJECT/'reports/subsample_k_candidates.csv')
    k_choices = pd.read_csv(PROJECT/'reports/k_stability.csv')
    assert len(cluster_checks)==30*k and len(k_candidates)==90 and len(k_choices)==10
    for row in stability.itertuples():
        if row.check=='initialization':
            subset,seed = X,row.replicate
        else:
            fit_rows = rng.choice(len(X),size=160,replace=False)
            subset,seed = X.iloc[fit_rows],42
        refitted = make_pipeline(k,seed).fit(subset)
        np.testing.assert_allclose(refitted.named_steps['scale'].mean_,subset.mean())
        predicted = refitted.predict(X)
        assert np.isclose(row.ari,adjusted_rand_score(labels,predicted))
        if row.check=='subsample_80pct':
            omitted = np.setdiff1d(np.arange(len(X)),fit_rows)
            assert row.omitted_rows==40
            assert np.isclose(row.omitted_ari,adjusted_rand_score(labels[omitted],predicted[omitted]))
            counts = contingency_matrix(labels,predicted)
            left,right = linear_sum_assignment(-counts)
            for reference_id,refit_id in zip(left,right):
                intersection = counts[reference_id,refit_id]
                expected = intersection/(counts[reference_id,:].sum()+counts[:,refit_id].sum()-intersection)
                recorded = cluster_checks[(cluster_checks.replicate==row.replicate)&(cluster_checks.reference_cluster==reference_id)]
                assert len(recorded)==1 and np.isclose(recorded.jaccard.iloc[0],expected)
            if row.replicate<10:
                candidates = k_candidates[k_candidates.replicate==row.replicate]
                assert candidates.k.tolist()==list(range(2,11))
                for entry in candidates.itertuples():
                    diagnostic = refitted if entry.k==k else make_pipeline(entry.k,42).fit(subset)
                    diagnostic_labels = diagnostic.predict(subset)
                    matrix = diagnostic.named_steps['scale'].transform(subset)
                    minimum = np.bincount(diagnostic_labels,minlength=entry.k).min()
                    assert entry.size_guard==8 and entry.smallest_cluster==minimum
                    assert entry.eligible == (minimum>=8)
                    assert np.isclose(entry.silhouette,silhouette_score(matrix,diagnostic_labels))
                choice = candidates[candidates.eligible].sort_values(['silhouette','k'],ascending=[False,True]).iloc[0]
                assert k_choices.loc[k_choices.replicate.eq(row.replicate),'k'].iloc[0]==choice.k
    examples = pd.read_csv(PROJECT/'outputs/example_profiles.csv')
    demo = pd.read_csv(PROJECT/'reports/assignment_demo.csv')
    result = assign_customers(model,examples,manifest['training_bounds'])
    np.testing.assert_array_equal(demo.cluster,result.cluster)
    np.testing.assert_allclose(demo.distance_to_centroid,result.distance_to_centroid)
    np.testing.assert_allclose(demo.distance_gap_to_second,result.distance_gap_to_second)
    np.testing.assert_array_equal(demo.outside_training_range,result.outside_training_range)
    assert result.outside_training_range.iloc[-1]
    np.testing.assert_array_equal(model.predict(examples[FEATURES[::-1]]),model.predict(examples))
    bad_inputs = [examples.assign(CustomerID=1),examples.drop(columns=FEATURES[0]),
                  examples.assign(**{FEATURES[0]:-1}),examples.assign(**{FEATURES[1]:101}),
                  examples.assign(**{FEATURES[1]:0}),examples.assign(**{FEATURES[0]:np.nan}),
                  examples.assign(**{FEATURES[0]:np.inf}),examples.assign(**{FEATURES[0]:'corrupt'})]
    for frame in bad_inputs:
        try:
            model.predict(frame)
        except (ValueError,TypeError):
            pass
        else:
            raise AssertionError('Invalid inference input accepted')
    for entry in selection.itertuples():
        candidate = make_pipeline(entry.k,42).fit(X)
        candidate_labels = candidate.predict(X)
        matrix = candidate.named_steps['scale'].transform(X)
        assert entry.smallest_cluster==np.bincount(candidate_labels,minlength=entry.k).min()
        assert np.isclose(entry.inertia,candidate.named_steps['kmeans'].inertia_)
        if entry.k>1:
            assert np.isclose(entry.silhouette,silhouette_score(matrix,candidate_labels))
            assert np.isclose(entry.davies_bouldin,davies_bouldin_score(matrix,candidate_labels))
            assert np.isclose(entry.calinski_harabasz,calinski_harabasz_score(matrix,candidate_labels))
    evidence = json.loads((PROJECT/'reports/selection_evidence.json').read_text())
    assert evidence['runner_up_k']==int(eligible.iloc[1].k)
    assert np.isclose(evidence['silhouette_gap_to_runner_up'],eligible.iloc[0].silhouette-eligible.iloc[1].silhouette)
    assert evidence['davies_bouldin_best_k']==int(selection[selection.k>1].sort_values(['davies_bouldin','k']).iloc[0].k)
    assert evidence['calinski_harabasz_best_k']==int(selection[selection.k>1].sort_values(['calinski_harabasz','k'],ascending=[False,True]).iloc[0].k)
    sensitivity = pd.read_csv(PROJECT/'reports/representation_sensitivity.csv')
    sensitivity_candidates = pd.read_csv(PROJECT/'reports/sensitivity_candidates.csv')
    assert len(sensitivity)==4 and len(sensitivity_candidates)==36
    for scenario, entries in sensitivity_candidates.groupby('scenario',sort=False):
        values = X.astype(float).copy()
        if entries.rank_score.iloc[0]:
            values[FEATURES[1]] = values[FEATURES[1]].rank(method='average',pct=True)
        matrix = StandardScaler().fit_transform(values)
        matrix[:,1] *= np.sqrt(entries.score_squared_distance_weight.iloc[0])
        partitions = {}
        for entry in entries.itertuples():
            fitted = KMeans(n_clusters=entry.k,n_init=20,random_state=42,max_iter=300,tol=1e-4,algorithm='lloyd').fit(matrix)
            partitions[entry.k] = fitted.labels_
            minimum = np.bincount(fitted.labels_,minlength=entry.k).min()
            assert entry.smallest_cluster==minimum and entry.eligible==(minimum>=10)
            assert np.isclose(entry.silhouette,silhouette_score(matrix,fitted.labels_))
        best = entries[entries.eligible].sort_values(['silhouette','k'],ascending=[False,True]).iloc[0]
        row = sensitivity[sensitivity.scenario.eq(scenario)].iloc[0]
        assert row.selected_k==best.k
        assert np.isclose(row.fixed_k_ari_to_primary,adjusted_rand_score(labels,partitions[k]))
        assert np.isclose(row.selected_k_ari_to_primary,adjusted_rand_score(labels,partitions[best.k]))
    quality = json.loads((PROJECT/'reports/quality_summary.json').read_text())
    assert quality['rows_removed']==0 and quality['analysis_rows']==len(raw)
    assert quality['repeated_two_feature_profiles']==int(X.duplicated().sum())
    assert quality['distinct_two_feature_profiles']==len(X.drop_duplicates())
    q1,q3 = X[FEATURES[0]].quantile([.25,.75]); iqr=q3-q1
    retained = X[FEATURES[0]].between(q1-1.5*iqr,q3+1.5*iqr)
    influence = make_pipeline(k,42).fit(X.loc[retained])
    assert quality['income_iqr_exclusion_diagnostic']['fit_rows']==int(retained.sum())
    assert np.isclose(quality['income_iqr_exclusion_diagnostic']['all_row_ari_to_primary'],adjusted_rand_score(labels,influence.predict(X)))
    assert np.isclose(metrics['subsample_ari_mean'],stability.loc[stability.check.eq('subsample_80pct'),'ari'].mean())
    assert np.isclose(metrics['omitted_subsample_ari_mean'],stability.omitted_ari.mean())
    assert np.isclose(metrics['per_cluster_jaccard_minimum'],cluster_checks.jaccard.min())
    assert metrics['subsample_selected_k_counts']=={str(int(key)):int(value) for key,value in k_choices.k.value_counts().items()}
    assert not result.outside_training_range.iloc[-2]
    assessment = json.loads((PROJECT/'reports/assessment.json').read_text())
    assert assessment['primary_k']==k and assessment['score_measurement_validated'] is False
    assert assessment['natural_customer_types_established'] is False and assessment['marketing_benefit_established'] is False
    print('PASS: all primary/sensitivity candidates, omitted-row ARIs, matched Jaccards, k-reselection and influence checks.')
    knowledge = json.loads((PROJECT/'reports/knowledge_summary.json').read_text())
    assert sum(f['customers'] for f in knowledge['findings'])==len(raw)
    for path in [PROJECT/'README.md']:
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',path.read_text()):
            if not target.startswith('https://'):
                assert (path.parent/target).exists(),target
    print(f'PASS: nine KDD steps, {len(cells)} executed code cells, source/code/model hashes.')
    print('PASS: selection rule, recomputed indices, assignments, centers, sizes and all 40 stability refits.')
    print(f'PASS: saved-model inference, reordered columns, extrapolation flag and {len(bad_inputs)} invalid-input rejections.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    verify()
