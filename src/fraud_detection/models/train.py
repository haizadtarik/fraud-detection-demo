import logging
import numpy as np
import lightgbm as lgb
from fraud_detection.data.features import FEATURE_COLUMNS, TARGET, SPLIT_KEY

log = logging.getLogger("train")


def temporal_split(df, train_max_day):
    train = df[df[SPLIT_KEY] <= train_max_day]
    test = df[df[SPLIT_KEY] > train_max_day]
    log.info(
        f"Temporal split @ day {train_max_day}: "
        f"train={len(train):,} (fraud {train[TARGET].mean():.3%}) | "
        f"test={len(test):,} (fraud {test[TARGET].mean():.3%})"
    )
    return train, test


def make_scale_pos_weight(y_train, strategy="sqrt"):
    """Imbalance lever. Raw neg/pos can be so extreme it saturates the first tree
    and stalls boosting; sqrt() caps it to a stable value while still upweighting fraud."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    raw = neg / pos
    spw = float(np.sqrt(raw)) if strategy == "sqrt" else float(raw)
    log.info(f"scale_pos_weight = {spw:.1f}  (raw neg/pos = {raw:.1f}, strategy={strategy})")
    return spw


def train_model(df, params, use_early_stopping=True):
    train, test = temporal_split(df, params["split"]["train_max_day"])
    X_tr, y_tr = train[FEATURE_COLUMNS], train[TARGET]
    X_te, y_te = test[FEATURE_COLUMNS], test[TARGET]

    mp = dict(params["model"]["params"])
    mp.pop("weight_strategy", None)  # not a LightGBM param; strip before passing
    strategy = params["model"].get("weight_strategy", "sqrt")
    mp["scale_pos_weight"] = make_scale_pos_weight(y_tr, strategy)

    model = lgb.LGBMClassifier(**mp)
    callbacks = [lgb.log_evaluation(0)]
    if use_early_stopping:
        callbacks.insert(0, lgb.early_stopping(params["model"]["early_stopping_rounds"]))

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric=["auc", "average_precision"],  # AUC first
        callbacks=[
            lgb.early_stopping(
                params["model"]["early_stopping_rounds"], first_metric_only=True
            ),  # <-- stop on AUC ONLY
            lgb.log_evaluation(0),
        ]
        if use_early_stopping
        else [lgb.log_evaluation(0)],
    )

    log.info(f"Best iteration: {model.best_iteration_ or model.n_estimators_}")
    return model, (X_tr, y_tr, X_te, y_te)
