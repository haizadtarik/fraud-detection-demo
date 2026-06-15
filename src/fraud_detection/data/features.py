# Never enter the model
EXCLUDED_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]

FEATURE_COLUMNS = [
    # --- transaction-intrinsic: serve trivially, no state needed ---
    "amount",  # transaction size
    "is_transfer",  # TRANSFER vs CASH_OUT
    "dest_is_merchant",  # merchants are not the mule pattern
    "hour",  # hour-of-day
    "is_night",  # 00:00–05:59 risk window
    # --- recipient history: require an online feature store at serving time ---
    "dest_prior_txn_count",  # fan-in count — mules receive from many
    "dest_prior_amount_sum",  # cumulative prior inflow to recipient
    "dest_prior_amount_mean",  # average prior inflow size to recipient
    "dest_is_new",  # first-ever payment to this beneficiary
    # --- origin history: degenerate in PaySim; validated + retained for transparency ---
    "orig_prior_txn_count",  # origin velocity — the workhorse on real recurring customers
]
TARGET = "isFraud"
SPLIT_KEY = "day"  # temporal-split key; NOT a model input


def add_temporal(df, night_start, night_end):
    df["hour"] = (df["step"] % 24).astype("int8")
    df["day"] = (df["step"] // 24).astype("int16")  # split key only
    df["is_night"] = ((df["hour"] >= night_start) & (df["hour"] < night_end)).astype("int8")
    return df


def add_intrinsic(df):
    df["is_transfer"] = (df["type"] == "TRANSFER").astype("int8")
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").fillna(False).astype("int8")
    return df


def add_recipient_history(df):
    df["dest_prior_txn_count"] = df.groupby("nameDest").cumcount().astype("int32")
    prior_sum = df.groupby("nameDest")["amount"].cumsum() - df["amount"]  # exclude self
    df["dest_prior_amount_sum"] = prior_sum.astype("float64")
    df["dest_prior_amount_mean"] = (
        (prior_sum / df["dest_prior_txn_count"].where(df["dest_prior_txn_count"] > 0))
        .fillna(0.0)
        .astype("float64")
    )  # 0 when no history yet
    df["dest_is_new"] = (df["dest_prior_txn_count"] == 0).astype("int8")
    return df


def add_origin_history(df):
    df["orig_prior_txn_count"] = df.groupby("nameOrig").cumcount().astype("int32")
    return df


def build_features(df, params):
    nf = params["features"]
    df = add_temporal(df, nf["night_start_hour"], nf["night_end_hour"])
    df = add_intrinsic(df)
    df = add_recipient_history(df)
    df = add_origin_history(df)

    drop_days = nf.get("drop_days", [])
    if drop_days:
        before = len(df)
        df = df[~df["day"].isin(drop_days)]
        # (logged by the pipeline, not here, to keep features.py import-clean)
    return df[FEATURE_COLUMNS + [SPLIT_KEY, TARGET]].copy()
