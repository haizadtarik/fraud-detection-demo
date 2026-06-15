import pandas as pd
from fraud_detection.config import load_params
from fraud_detection.data.features import (
    build_features,
    FEATURE_COLUMNS,
    EXCLUDED_COLUMNS,
    TARGET,
)


def _toy_transactions():
    """Small hand-built history so cumulative features have something to accumulate."""
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 3, 3, 4],
            "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT"],
            "amount": [100.0, 200.0, 150.0, 300.0, 50.0, 400.0],
            "nameOrig": ["C1", "C2", "C3", "C1", "C4", "C5"],
            "nameDest": ["C9", "C9", "M1", "C9", "C8", "C8"],
            "oldbalanceOrg": [1000.0, 2000.0, 1500.0, 700.0, 500.0, 900.0],
            "newbalanceOrig": [900.0, 1800.0, 1350.0, 400.0, 450.0, 500.0],
            "oldbalanceDest": [0.0, 100.0, 0.0, 300.0, 0.0, 50.0],
            "newbalanceDest": [100.0, 300.0, 0.0, 600.0, 50.0, 450.0],
            "isFraud": [0, 0, 0, 1, 0, 1],
        }
    )


def test_no_leaky_columns_in_features():
    """The data-card-forbidden balance columns must never reach the model."""
    params = load_params()
    feats = build_features(_toy_transactions(), params)
    for col in EXCLUDED_COLUMNS:
        assert col not in feats.columns, f"LEAKAGE: forbidden column '{col}' in features"


def test_feature_schema_is_exact():
    params = load_params()
    feats = build_features(_toy_transactions(), params)
    assert list(feats.columns) == FEATURE_COLUMNS + ["day", TARGET]


def test_recipient_history_is_point_in_time():
    """dest_prior_txn_count must count only PRIOR rows (no look-ahead leakage)."""
    params = load_params()
    feats = build_features(_toy_transactions(), params)
    # C9 receives at steps 1,1,3. The first C9 row must have 0 priors.
    c9_rows = feats[_toy_transactions()["nameDest"] == "C9"].sort_values("day")
    assert c9_rows["dest_prior_txn_count"].iloc[0] == 0, "first occurrence must have 0 priors"
    assert c9_rows["dest_prior_txn_count"].is_monotonic_increasing


def test_dest_is_new_flag():
    params = load_params()
    feats = build_features(_toy_transactions(), params)
    # M1 is seen exactly once -> must be flagged new.
    m1 = feats[_toy_transactions()["nameDest"] == "M1"]
    assert (m1["dest_is_new"] == 1).all()
