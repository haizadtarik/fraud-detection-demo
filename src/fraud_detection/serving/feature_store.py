import logging
import pandas as pd
from fraud_detection.config import load_params, PROJECT_ROOT

log = logging.getLogger("feature_store")

class RecipientFeatureStore:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self._global_mean_inflow: float = 0.0

    def build_from_intermediate(self):
        """Precompute final recipient aggregates from the cleaned transaction history."""
        params = load_params()
        path = PROJECT_ROOT / params["paths"]["intermediate"]
        df = pd.read_parquet(path).sort_values("step")

        # Mirror the TRAINING aggregation, taking each recipient's final running state.
        g = df.groupby("nameDest")
        agg = pd.DataFrame({
            "dest_prior_txn_count": g.size(),
            "dest_prior_amount_sum": g["amount"].sum(),
        })
        agg["dest_prior_amount_mean"] = (
            agg["dest_prior_amount_sum"] / agg["dest_prior_txn_count"].where(agg["dest_prior_txn_count"] > 0)
        ).fillna(0.0)
        self._store = agg.to_dict("index")
        self._global_mean_inflow = float(df["amount"].mean())
        log.info(f"Feature store built: {len(self._store):,} recipients")

    def lookup(self, name_dest: str) -> dict:
        """Return recipient aggregates. Unknown recipient => first-time beneficiary."""
        if name_dest in self._store:
            rec = self._store[name_dest]
            return {
                "dest_prior_txn_count": int(rec["dest_prior_txn_count"]),
                "dest_prior_amount_sum": float(rec["dest_prior_amount_sum"]),
                "dest_prior_amount_mean": float(rec["dest_prior_amount_mean"]),
                "dest_is_new": 0,
            }
        # Cold-start: genuinely new recipient — the high-risk mule signal.
        return {
            "dest_prior_txn_count": 0,
            "dest_prior_amount_sum": 0.0,
            "dest_prior_amount_mean": 0.0,
            "dest_is_new": 1,
        }