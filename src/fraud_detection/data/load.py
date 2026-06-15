import pandas as pd
from fraud_detection.config import load_params, PROJECT_ROOT

MODELLED_TYPES = ["TRANSFER", "CASH_OUT"]  # Phase 1 finding: fraud lives only here
BALANCE_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

RAW_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def load_raw(source: str = "sample") -> pd.DataFrame:
    params = load_params()
    key = "sample" if source == "sample" else "raw"
    path = PROJECT_ROOT / params["paths"][key]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. For the full set run the Kaggle download "
            f"(see README), or use --source sample to run on the committed sample."
        )
    return pd.read_csv(path, dtype=RAW_DTYPES)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df[BALANCE_COLS] = df[BALANCE_COLS].fillna(0)
    df = df[df["type"].isin(MODELLED_TYPES)].copy()
    df = df[df["amount"] >= 0]
    return df.sort_values("step").reset_index(drop=True)
