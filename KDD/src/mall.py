"""Input contract and reproducible K-Means customer-segmentation pipeline."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DATASET = 'vjchoudhary7/customer-segmentation-tutorial-in-python'
SOURCE_URL = f'https://www.kaggle.com/datasets/{DATASET}'
DOWNLOAD_URL = f'https://www.kaggle.com/api/v1/datasets/download/{DATASET}?datasetVersionNumber=1'
SHA256 = '416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77'
FEATURES = ['Annual Income (k$)', 'Spending Score (1-100)']


def feature_frame(frame):
    """Accept exactly two finite numeric features, with no implicit imputation."""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or set(frame.columns) != set(FEATURES):
        raise ValueError(f'Expected exactly these named columns: {FEATURES}')
    result = frame[FEATURES].apply(pd.to_numeric, errors='raise').astype(float)
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise ValueError('Features must contain finite observed values.')
    if result[FEATURES[0]].lt(0).any() or not result[FEATURES[1]].between(1,100).all():
        raise ValueError('Income must be nonnegative; spending score must be between 1 and 100.')
    return result


def make_pipeline(k, seed=42):
    return Pipeline([
        ('features', FunctionTransformer(feature_frame, validate=False)),
        ('scale', StandardScaler()),
        ('kmeans', KMeans(n_clusters=k, init='k-means++', n_init=20,
                          max_iter=300, tol=1e-4, random_state=seed, algorithm='lloyd')),
    ])


def assign_customers(model, frame, training_bounds):
    """Return nearest-cluster assignments with descriptive distance/range diagnostics."""
    values = feature_frame(frame)
    distances = model.transform(values)
    ordered = np.sort(distances, axis=1)
    outside = np.zeros(len(values), dtype=bool)
    for column in FEATURES:
        lower, upper = training_bounds[column]
        outside |= ~values[column].between(lower,upper).to_numpy()
    return pd.DataFrame({
        'cluster': model.predict(values),
        'distance_to_centroid': ordered[:,0],
        'distance_gap_to_second': ordered[:,1]-ordered[:,0],
        'outside_training_range': outside,
    }, index=frame.index)


def representation_sensitivity(frame, reference_labels, k, include_rank=False):
    """Small fixed sensitivity analysis; never replaces the primary representation."""
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    scenarios = [('score_weight_0.5',0.5,False),('score_weight_1',1.0,False),('score_weight_2',2.0,False)]
    if include_rank:
        scenarios.append(('score_rank_equal_weight',1.0,True))
    candidates, summaries = [], []
    for name, weight, use_rank in scenarios:
        values = feature_frame(frame)
        if use_rank:
            values[FEATURES[1]] = values[FEATURES[1]].rank(method='average',pct=True)
        matrix = StandardScaler().fit_transform(values)
        matrix[:,1] *= np.sqrt(weight)
        fits = {}
        for count in range(2,11):
            estimator = KMeans(n_clusters=count,n_init=20,random_state=42,max_iter=300,tol=1e-4,algorithm='lloyd').fit(matrix)
            labels = estimator.labels_
            row = {'scenario':name,'k':count,'score_squared_distance_weight':weight,
                   'rank_score':use_rank,'silhouette':float(silhouette_score(matrix,labels)),
                   'smallest_cluster':int(np.bincount(labels,minlength=count).min())}
            row['eligible'] = row['smallest_cluster'] >= 10
            candidates.append(row)
            fits[count] = labels
        eligible = [row for row in candidates if row['scenario']==name and row['eligible']]
        best = sorted(eligible,key=lambda row:(-row['silhouette'],row['k']))[0]
        summaries.append({'scenario':name,'selected_k':best['k'],'selected_silhouette':best['silhouette'],
                          'fixed_primary_k':k,'fixed_k_ari_to_primary':float(adjusted_rand_score(reference_labels,fits[k])),
                          'selected_k_ari_to_primary':float(adjusted_rand_score(reference_labels,fits[best['k']]))})
    return pd.DataFrame(candidates),pd.DataFrame(summaries)
