# SEMMA — Seoul Bike Sharing Demand

Day-ahead hourly rental prediction following Sample, Explore, Modify, Model, and
Assess in the [executed notebook](notebooks/seoul_bike_semma.ipynb). Inputs use
calendar information and prior-day/week observations, excluding same-day rentals,
weather, and actual operating status.

## Data and usage

[Kaggle dataset](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction),
version 1: 8,760 hourly records from December 2017 through November 2018. The
notebook downloads and checksums `data/raw/SeoulBikeData.csv` (Latin-1 encoding).
The first seven days supply lag history.

Use Python 3.11. From the repository root:

```sh
python -m pip install -r SEMMA/requirements.txt
python -m jupyterlab
```

Run the notebook cells in order, or execute and verify programmatically:

```sh
python SEMMA/src/verify_project.py --execute
```

## Results

Three expanding-window validations (28 days each, 24-hour gaps) selected Ridge
regression over previous-day persistence and Poisson gradient boosting. The final
56 days, October 6–November 30, 2018, provide 1,344 chronological test hours.

| Test metric | Ridge | Previous-day persistence |
| --- | --- | --- |
| MAE, rentals/hour | 330.76 | 301.37 |
| RMSE, rentals/hour | 439.15 | 516.96 |
| R² | 0.413 | 0.187 |

Ridge does **not** beat persistence on the primary metric: MAE is 9.8% higher.
The paired weekly-block interval for the MAE difference includes zero; no robust
superiority claim is supported.
Ridge is retained as the archived validation-selected candidate, not recommended
over persistence. The reviewed test period is a reused benchmark.

See [model comparison](reports/model_comparison.csv), [metrics](reports/metrics.json),
[time splits](reports/time_splits.csv), [exploration](reports/development_eda.png),
[evaluation figures](reports/test_evaluation.png), and [error slices](reports/error_slices.csv).
See the [assessment decision](reports/assessment.json) and [two-round review](reports/review_log.md).
The [saved pipeline](outputs/demand_pipeline.joblib), [example inputs](outputs/example_profiles.csv),
[prediction demonstration](reports/prediction_demo.csv), [test predictions](outputs/test_predictions.csv),
[run manifest](outputs/run_manifest.json), and [input contract](reports/input_contract.json)
are included. Keep `src/seoul.py` available when loading the trusted model artifact.

Forecasts assume prior-day observations are complete at local midnight without
reporting delay; history updates daily while the fitted model stays fixed. One
year and eight test weeks limit seasonal and uncertainty claims. Unannounced
closures and weather changes remain difficult. Direct dependencies are pinned;
a fresh installation and full transitive lock have not been verified.
