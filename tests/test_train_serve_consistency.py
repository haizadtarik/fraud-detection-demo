import pandas as pd
from fraud_detection.config import load_params
from fraud_detection.data.features import FEATURE_COLUMNS, build_features
from fraud_detection.serving.app import assemble_features
from fraud_detection.serving.schemas import Transaction


def _training_features_for_one(raw_row: dict, params) -> pd.Series:
    """Run the TRAINING feature pipeline on a one-row frame, return the model-input row."""
    df = pd.DataFrame([raw_row])
    # build_features needs sorted history for cumcount; a single row is trivially sorted.
    df["step"] = raw_row["step"]
    feats = build_features(df.assign(isFraud=0), params)
    return feats[FEATURE_COLUMNS].iloc[0]


def test_stateless_features_match():
    """Stateless features (amount, type, time) must be identical train vs serve."""
    params = load_params()
    raw = {"type": "TRANSFER", "amount": 50000.0,
           "nameOrig": "C111", "nameDest": "M222", "step": 14}

    train_row = _training_features_for_one(raw, params)
    txn = Transaction(transaction_type="TRANSFER", amount=50000.0,
                      name_orig="C111", name_dest="M222", step=14)
    # For a never-seen recipient the store returns the cold-start aggregates.
    store_feats = {"dest_prior_txn_count": 0, "dest_prior_amount_sum": 0.0,
                   "dest_prior_amount_mean": 0.0, "dest_is_new": 1}
    serve_row = assemble_features(txn, store_feats).iloc[0]

    for f in ["amount", "is_transfer", "dest_is_merchant", "hour", "is_night"]:
        assert train_row[f] == serve_row[f], (
            f"TRAIN-SERVE SKEW on '{f}': train={train_row[f]} serve={serve_row[f]}")


def test_hour_derivation_matches():
    """The step->hour transform must be identical on both paths (a classic skew source)."""
    params = load_params()
    for step in [0, 5, 6, 13, 23, 24, 47, 100]:
        raw = {"type": "CASH_OUT", "amount": 100.0,
               "nameOrig": "C1", "nameDest": "C2", "step": step}
        train_row = _training_features_for_one(raw, params)
        txn = Transaction(transaction_type="CASH_OUT", amount=100.0,
                          name_orig="C1", name_dest="C2", step=step)
        serve_row = assemble_features(
            txn, {"dest_prior_txn_count": 0, "dest_prior_amount_sum": 0.0,
                  "dest_prior_amount_mean": 0.0, "dest_is_new": 1}).iloc[0]
        assert train_row["hour"] == serve_row["hour"] == step % 24
        assert train_row["is_night"] == serve_row["is_night"]