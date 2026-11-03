"""Deterministic Telco input normalization shared by training and inference."""
import numpy as np
import pandas as pd

FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges',
]
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]

YES_NO = {'Yes', 'No'}
INTERNET_OPTIONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
CATEGORY_VALUES = {
    'gender': {'Female', 'Male'}, 'SeniorCitizen': YES_NO,
    **{c: YES_NO for c in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']},
    'MultipleLines': {'Yes', 'No', 'No phone service'},
    'InternetService': {'DSL', 'Fiber optic', 'No'},
    **{c: {'Yes', 'No', 'No internet service'} for c in INTERNET_OPTIONS},
    'Contract': {'Month-to-month', 'One year', 'Two year'},
    'PaymentMethod': {'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'},
}


def clean_features(frame):
    """Accept original CSV predictor columns; reject schema or numeric corruption."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('Inputs must be a pandas DataFrame with named columns.')
    missing = set(FEATURES) - set(frame.columns)
    extra = set(frame.columns) - set(FEATURES)
    if missing or extra or not frame.columns.is_unique:
        raise ValueError(f'Invalid feature schema: missing={sorted(missing)}, extra={sorted(extra)}')
    result = frame.loc[:, FEATURES].copy()
    for column in NUMERIC:
        source = result[column].replace(r'^\s*$', np.nan, regex=True)
        numeric = pd.to_numeric(source, errors='raise').astype(float)
        if ((numeric.dropna() < 0) | ~np.isfinite(numeric.dropna())).any():
            raise ValueError(f'{column} must be finite and nonnegative when present.')
        result[column] = numeric
    senior = result['SeniorCitizen']
    mapped = senior.map({0: 'No', 1: 'Yes', '0': 'No', '1': 'Yes', 'No': 'No', 'Yes': 'Yes'})
    if (senior.notna() & mapped.isna()).any():
        raise ValueError('SeniorCitizen must be 0/1 or No/Yes.')
    result['SeniorCitizen'] = mapped
    if not result['tenure'].dropna().mod(1).eq(0).all():
        raise ValueError('tenure must contain whole months.')
    for column, allowed in CATEGORY_VALUES.items():
        if not result[column].dropna().isin(allowed).all():
            raise ValueError(f'Unknown category in {column}; review input schema before scoring.')
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    phone_known = result['PhoneService'].notna() & result['MultipleLines'].notna()
    mismatch = result['PhoneService'].eq('No') != result['MultipleLines'].eq('No phone service')
    if (phone_known & mismatch).any():
        raise ValueError('PhoneService and MultipleLines disagree.')
    for column in INTERNET_OPTIONS:
        known = result['InternetService'].notna() & result[column].notna()
        mismatch = result['InternetService'].eq('No') != result[column].eq('No internet service')
        if (known & mismatch).any():
            raise ValueError(f'InternetService and {column} disagree.')
    return result
