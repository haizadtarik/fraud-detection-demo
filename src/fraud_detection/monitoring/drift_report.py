import logging
import pandas as pd
import numpy as np 

from fraud_detection.config import load_params, PROJECT_ROOT
from fraud_detection.data.features import FEATURE_COLUMNS, TARGET, SPLIT_KEY
from fraud_detection.monitoring.psi import psi_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("drift")

def binary_psi(ref_y, cur_y, eps=1e-6):
    ref_p = np.array([1 - ref_y.mean(), ref_y.mean()])     # [P(legit), P(fraud)]
    cur_p = np.array([1 - cur_y.mean(), cur_y.mean()])
    ref_p = np.clip(ref_p, eps, None)
    cur_p = np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))    

def main():
    params = load_params()
    df = pd.read_parquet(PROJECT_ROOT / params["paths"]["features"])
    cut = params["split"]["train_max_day"]

    reference = df[df[SPLIT_KEY] <= cut]      # what the model trained on
    current   = df[df[SPLIT_KEY] >  cut]      # newer "production" traffic
    log.info(f"Reference: {len(reference):,} rows (fraud {reference[TARGET].mean():.3%}) | "
             f"Current: {len(current):,} rows (fraud {current[TARGET].mean():.3%})")

    # --- PSI on features ---
    psi = psi_report(reference, current, FEATURE_COLUMNS)
    log.info("PSI by feature:\n" + psi.to_string(index=False))
    flagged = psi[psi.band != "stable"]
    if len(flagged):
        log.warning(f"{len(flagged)} feature(s) drifting: "
                    f"{', '.join(flagged.feature)}")

    # --- Target drift (the prevalence jump) ---
    tgt_psi = binary_psi(reference[TARGET], current[TARGET])
    log.info(f"Target (fraud-rate) PSI: {tgt_psi:.4f} "
             f"[{reference[TARGET].mean():.3%} -> {current[TARGET].mean():.3%}]")

    # --- Evidently report (visual artifact) ---
    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset

        ref_df = reference[FEATURE_COLUMNS + [TARGET]].rename(columns={TARGET: "target"})
        cur_df = current[FEATURE_COLUMNS + [TARGET]].rename(columns={TARGET: "target"})

        schema = DataDefinition(
            numerical_columns=[c for c in FEATURE_COLUMNS
                               if c not in ("is_transfer", "is_night",
                                            "dest_is_merchant", "dest_is_new")],
            categorical_columns=["is_transfer", "is_night", "dest_is_merchant",
                                 "dest_is_new", "target"],
        )
        ref_ds = Dataset.from_pandas(ref_df, data_definition=schema)
        cur_ds = Dataset.from_pandas(cur_df, data_definition=schema)

        report = Report([DataDriftPreset(method="psi")])
        snapshot = report.run(cur_ds, ref_ds)          # positional: current, reference

        out = PROJECT_ROOT / "data" / "05_monitoring" / "drift_report.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        # save_html
        snapshot.save_html(str(out)) 
        log.info(f"Evidently report -> {out}")
    except Exception as e:
        log.warning(f"Evidently report failed ({type(e).__name__}: {e}). "
                    f"Hand-rolled PSI above is the primary signal.")

    # Persist PSI table.
    psi_out = PROJECT_ROOT / "data" / "05_monitoring" / "psi.csv"
    psi_out.parent.mkdir(parents=True, exist_ok=True)
    psi.to_csv(psi_out, index=False)
    log.info(f"PSI table -> {psi_out}")

if __name__ == "__main__":
    main()