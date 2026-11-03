"""Causal day-ahead features and small reusable Seoul demand estimators."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

DATASET = 'saurabhshahane/seoul-bike-sharing-demand-prediction'
SOURCE_URL = f'https://www.kaggle.com/datasets/{DATASET}'
DOWNLOAD_URL = f'https://www.kaggle.com/api/v1/datasets/download/{DATASET}?datasetVersionNumber=1'
SHA256 = '373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60'
WEATHER = {
    'Temperature(°C)': 'temperature', 'Humidity(%)': 'humidity',
    'Wind speed (m/s)': 'wind_speed', 'Visibility (10m)': 'visibility',
    'Dew point temperature(°C)': 'dew_point', 'Solar Radiation (MJ/m2)': 'solar',
    'Rainfall(mm)': 'rainfall', 'Snowfall (cm)': 'snowfall',
}
LAGS = ['rentals_lag24', 'rentals_lag168'] + [f'{c}_lag24' for c in WEATHER.values()]
INPUTS = ['timestamp', 'holiday'] + LAGS
DESIGN = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'year_sin', 'year_cos', 'holiday'] + LAGS


def load_data(path):
    frame = pd.read_csv(Path(path), encoding='latin1')
    expected = {'Date', 'Rented Bike Count', 'Hour', *WEATHER, 'Seasons', 'Holiday', 'Functioning Day'}
    if set(frame.columns) != expected or frame.isna().any().any():
        raise ValueError('Unexpected source schema or missing source values.')
    if not frame.Hour.between(0, 23).all() or not frame.Hour.mod(1).eq(0).all():
        raise ValueError('Invalid source hour.')
    frame['timestamp'] = pd.to_datetime(frame.Date, format='%d/%m/%Y') + pd.to_timedelta(frame.Hour, unit='h')
    frame = frame.sort_values('timestamp').reset_index(drop=True).rename(columns={
        'Rented Bike Count': 'rentals', 'Holiday': 'holiday', 'Functioning Day': 'functioning', **WEATHER})
    expected_hours = pd.date_range(frame.timestamp.min(), frame.timestamp.max(), freq='h')
    if not frame.timestamp.is_unique or not np.array_equal(frame.timestamp.to_numpy(), expected_hours.to_numpy()):
        raise ValueError('Source must have one record for every consecutive hour.')
    if not frame.rentals.ge(0).all() or not frame.rentals.mod(1).eq(0).all():
        raise ValueError('Rental counts must be nonnegative integers.')
    if not frame.holiday.isin(['Holiday', 'No Holiday']).all() or not frame.functioning.isin(['Yes', 'No']).all():
        raise ValueError('Unexpected holiday or functioning-day categories.')
    if not frame.loc[frame.functioning.eq('No'), 'rentals'].eq(0).all():
        raise ValueError('Closed hours have nonzero recorded rentals.')
    return frame


def build_supervised(frame):
    """Lag only past observations; omit the initial seven-day warm-up."""
    timestamps = pd.DatetimeIndex(frame.timestamp)
    if len(timestamps) <= 168 or timestamps.hasnans or timestamps.tz is not None or not timestamps.is_unique or not timestamps.floor('h').equals(timestamps):
        raise ValueError('Need more than seven days of unique local hourly history.')
    if not np.array_equal(timestamps.to_numpy(), pd.date_range(timestamps.min(), timestamps.max(), freq='h').to_numpy()):
        raise ValueError('Lag construction requires ordered, consecutive hours.')
    result = frame[['timestamp', 'holiday', 'rentals', 'functioning']].copy()
    result['rentals_lag24'] = frame.rentals.shift(24)
    result['rentals_lag168'] = frame.rentals.shift(168)
    for column in WEATHER.values():
        result[f'{column}_lag24'] = frame[column].shift(24)
    return result.iloc[168:].reset_index(drop=True)


def design_matrix(frame):
    """Validate the inference contract and derive only known calendar variables."""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or set(frame.columns) != set(INPUTS):
        raise ValueError(f'Expected exactly these columns: {INPUTS}')
    dates = pd.to_datetime(frame.timestamp, format='ISO8601', errors='raise')
    if dates.isna().any() or dates.dt.tz is not None or not dates.dt.floor('h').eq(dates).all():
        raise ValueError('Expected complete, timezone-naive local hourly timestamps.')
    if not frame.holiday.isin(['Holiday', 'No Holiday']).all():
        raise ValueError('Unknown holiday category.')
    result = pd.DataFrame(index=frame.index)
    for label, values, period in [('hour', dates.dt.hour, 24), ('weekday', dates.dt.dayofweek, 7),
                                  ('year', dates.dt.dayofyear - 1, 365.25)]:
        result[f'{label}_sin'] = np.sin(2*np.pi*values/period)
        result[f'{label}_cos'] = np.cos(2*np.pi*values/period)
    result['holiday'] = frame.holiday.eq('Holiday').astype(float)
    for column in LAGS:
        values = pd.to_numeric(frame[column], errors='raise').astype(float)
        if column.startswith('rentals_') and (values.isna().any() or not values.dropna().mod(1).eq(0).all()):
            raise ValueError('Historical rental counts must be observed whole counts; no implicit imputation.')
        if np.isinf(values).any():
            raise ValueError(f'Nonfinite {column}')
        if not column.startswith(('temperature_', 'dew_point_')) and values.dropna().lt(0).any():
            raise ValueError(f'Negative {column}')
        if column == 'humidity_lag24' and values.dropna().gt(100).any():
            raise ValueError('Humidity must not exceed 100 percent.')
        result[column] = values
    return result[DESIGN]


class PreviousDayRegressor(RegressorMixin, BaseEstimator):
    """Persistence baseline for a fully observed previous day."""
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        prediction = X['rentals_lag24'].to_numpy(dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError('Previous-day baseline requires observed rentals_lag24.')
        return prediction


class NonnegativeRegressor(RegressorMixin, BaseEstimator):
    """Apply the same predeclared nonnegative-count rule during scoring and replay."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(0, self.estimator_.predict(X))


def make_day_inputs(history, forecast_day, holiday):
    """Build a midnight 24-hour batch from prior observations and a known calendar label."""
    origin = pd.Timestamp(forecast_day)
    if origin.tz is not None or origin != origin.normalize():
        raise ValueError('Forecast origin must be a timezone-naive local midnight.')
    times = pd.DatetimeIndex(history.timestamp)
    if not times.is_unique or times.hasnans or times.tz is not None or (times >= origin).any():
        raise ValueError('History must contain unique timestamps strictly before the forecast day.')
    source = history.set_index('timestamp')
    forecasts = pd.date_range(origin, periods=24, freq='h')
    result = pd.DataFrame({'timestamp': forecasts, 'holiday': holiday})
    for lag in [24, 168]:
        lookup = forecasts - pd.Timedelta(hours=lag)
        if not lookup.isin(source.index).all():
            raise ValueError(f'Missing required {lag}-hour historical rental observations.')
        result[f'rentals_lag{lag}'] = source.loc[lookup, 'rentals'].to_numpy(dtype=float)
    lookup = forecasts - pd.Timedelta(hours=24)
    for column in WEATHER.values():
        result[f'{column}_lag24'] = source.loc[lookup, column].to_numpy(dtype=float)
    design_matrix(result[INPUTS])
    return result[INPUTS]
