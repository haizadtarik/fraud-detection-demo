import argparse
import logging
import joblib
import mlflow
from sklearn.calibration import calibration_curve

from fraud_detection.config import load_params, PROJECT_ROOT
from fraud_detection.data.features import FEATURE_COLUMNS
from fraud_detection.models.train import train_model
from fraud_detection.models.evaluate import (
    core_metrics,
    precision_at_k,
    cost_optimal_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("training_pipeline")


def main(source_label: str):
    params = load_params()

    feat_path = PROJECT_ROOT / params["paths"]["features"]
    import pandas as pd

    df = pd.read_parquet(feat_path)

    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"lgbm_{source_label}"):
        mlflow.log_params(params["model"]["params"])
        mlflow.log_param("split_strategy", params["split"]["strategy"])
        mlflow.log_param("train_max_day", params["split"]["train_max_day"])
        mlflow.log_param("data_source", source_label)

        model, (X_tr, y_tr, X_te, y_te) = train_model(df, params)
        y_score = model.predict_proba(X_te)[:, 1]

        # --- metrics ---
        m = core_metrics(y_te, y_score)
        m.update(precision_at_k(y_te, y_score, params["evaluate"]["precision_at_k"]))
        cm = params["evaluate"]["cost_matrix"]
        thr = cost_optimal_threshold(y_te, y_score, cm["cost_fn"], cm["cost_fp"])

        log.info(
            f"ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} "
            f"| test prevalence={m['prevalence']:.3%}"
        )
        log.info(
            f"Cost-optimal threshold={thr['threshold']:.4f} -> "
            f"precision={thr['precision']:.3f}, recall={thr['recall']:.3f}, "
            f"FP={thr['fp']}, FN={thr['fn']}"
        )
        for k in params["evaluate"]["precision_at_k"]:
            kk = min(k, len(y_te))
            log.info(
                f"  precision@{kk}={m[f'precision_at_{kk}']:.3f} "
                f"recall@{kk}={m[f'recall_at_{kk}']:.3f}"
            )

        mlflow.log_metrics(m)
        mlflow.log_metrics({f"thr_{k}": v for k, v in thr.items() if isinstance(v, (int, float))})

        # --- calibration (reliability) ---
        frac_pos, mean_pred = calibration_curve(
            y_te, y_score, n_bins=params["evaluate"]["calibration_bins"]
        )
        log.info(
            "Calibration (mean_pred -> observed_freq): "
            + ", ".join(f"{p:.2f}->{f:.2f}" for p, f in zip(mean_pred, frac_pos))
        )

        # --- feature importance (grouped read in talking points) ---
        imp = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1])
        log.info("Feature importance (gain): " + ", ".join(f"{n}={v}" for n, v in imp))

        # --- persist ---
        out = PROJECT_ROOT / params["paths"]["model"]
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "features": FEATURE_COLUMNS, "threshold": thr["threshold"]}, out
        )
        mlflow.log_artifact(str(out))
        log.info(f"Saved model bundle -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sample", "full"], default="full")
    main(ap.parse_args().source)
