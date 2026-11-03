# CRISP-DM — Telco Customer Churn

Retrospective customer churn classification using all six CRISP-DM phases:
business understanding, data understanding, preparation, modeling, evaluation,
and a local scoring demonstration.

## Data and usage

[Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
version 1: 7,043 customers and 21 columns. The
[notebook](notebooks/telco_churn_crisp_dm.ipynb) downloads and checksums the CSV;
alternatively place the version 1 CSV in `data/raw/`.

Use Python 3.11. From the repository root:

```sh
python -m pip install -r CRISP-DM/requirements.txt
python -m jupyterlab
```

Run the notebook cells in order. To execute and verify it programmatically:

```sh
python CRISP-DM/src/verify_project.py --execute
```

The project requirements pin direct dependencies from the verified environment;
transitive dependencies are not fully locked. Raw data in `data/` is downloaded
separately. The [saved pipeline](outputs/churn_pipeline.joblib),
[example inputs](outputs/example_profiles.csv), [test predictions](outputs/test_predictions.csv),
and [run manifest](outputs/run_manifest.json) are included. Keep `src/telco.py`
available when loading the trusted model artifact.

## Results

Five-fold training cross-validation selected unweighted logistic regression from
four simple candidates. On the reused 1,409-customer benchmark at threshold 0.50:

| Metric | Value |
| --- | --- |
| ROC AUC | 0.842 |
| Average precision | 0.634 |
| Churn precision | 0.658 |
| Churn recall | 0.561 |
| Churn F1 | 0.606 |

The inherited illustrative recall target of 0.60 is **not met**. Of 319 flagged
customers, 210 churned and 109 did not; 164 churners were missed. Selecting the
highest-ranked 20% identifies half the churners, an illustrative workload scenario.

See [evaluation and uncertainty](reports/metrics.json), [model comparison](reports/cross_validation.csv),
[evaluation figures](reports/test_evaluation.png),
[subgroup errors](reports/subgroup_metrics.csv), [prediction examples](reports/prediction_demo.csv),
[provenance](reports/provenance.json), [input contract](reports/input_contract.json),
[decision](reports/decision.json), and [methodological review](reports/review_log.md).

This static snapshot does not establish pre-churn feature availability, a future
prediction window, intervention benefit, or equal subgroup performance. The test
set has been inspected repeatedly; its results are not independent confirmation.
Business acceptance is unresolved; intended use is offline analysis and local scoring.
