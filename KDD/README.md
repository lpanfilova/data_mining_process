# KDD — Mall Customers

Discover customer groups with K-Means in the [nine-step KDD notebook](notebooks/mall_customers_kdd.ipynb),
from domain understanding and data preparation through mining, interpretation,
and a saved-model assignment demonstration.

## Data and usage

[Kaggle Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python),
version 1: 200 customers and five columns. The notebook downloads and checksums
`data/raw/Mall_Customers.csv`. Clustering uses standardized annual income and
spending score; age and gender are descriptive only, and customer ID is excluded.

Use Python 3.11. From the repository root:

```sh
python -m pip install -r KDD/requirements.txt
python -m jupyterlab
```

Run the notebook cells in order, or execute and verify programmatically:

```sh
python KDD/src/verify_project.py --execute
```

## Results

Highest silhouette among eligible `k=2..10` solutions selected **five clusters**;
each candidate required at least 10 customers per cluster.

| Diagnostic | Result |
| --- | --- |
| Internal silhouette | 0.555 |
| Davies–Bouldin index | 0.572 |
| Cluster sizes | 81, 39, 22, 35, 23 |
| Mean ARI across 10 initialization checks | 1.000 |
| Mean ARI across 30 subsampling checks | 0.989 |
| Mean omitted-row ARI in those checks | 0.985 |

Five was reselected in all ten subsample searches, but is conditional on the
representation: halving score weight selects seven. Ranking score retains five
but changes assignments (ARI 0.915); these checks do not validate score semantics.

See [cluster selection](reports/cluster_selection.csv), [profiles](reports/cluster_profiles.csv),
[centroids](reports/centroids.csv), [segment figures](reports/customer_segments.png),
[metrics](reports/metrics.json), [stability](reports/stability.csv), and [knowledge summary](reports/knowledge_summary.json).
See [representation sensitivity](reports/representation_sensitivity.csv),
[assessment](reports/assessment.json), and the [two-round review](reports/review_log.md).
The [saved pipeline](outputs/customer_segments.joblib), [customer assignments](outputs/cluster_assignments.csv),
[synthetic example inputs](outputs/example_profiles.csv), [assignment demonstration](reports/assignment_demo.csv),
[run manifest](outputs/run_manifest.json), and [input contract](reports/input_contract.json)
are included. Keep `src/mall.py` available when loading the trusted model artifact.

Cluster IDs are arbitrary. Internal metrics and stability reuse this small sample;
they do not establish natural customer types, future behavior, fairness, or
marketing benefit. Income/score weighting and score geometry are assumptions;
sampling and score construction remain unverified. New profiles always receive
a cluster, with out-of-range examples flagged. Direct dependencies are pinned;
a fresh installation and full transitive lock have not been verified.
