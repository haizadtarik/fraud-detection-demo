import time
import logging
import joblib
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager

from fraud_detection.config import load_params, PROJECT_ROOT
from fraud_detection.data.features import FEATURE_COLUMNS
from fraud_detection.serving.schemas import Transaction, ScoreResponse
from fraud_detection.serving.feature_store import RecipientFeatureStore
from fraud_detection.serving.rules import apply_rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("serving")

STATE: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    params = load_params()
    bundle = joblib.load(PROJECT_ROOT / params["paths"]["model"])
    STATE["model"] = bundle["model"]
    STATE["features"] = bundle["features"]
    STATE["bands"] = params["serving"]["decision_bands"]
    fs = RecipientFeatureStore()
    fs.build_from_intermediate()
    STATE["store"] = fs
    log.info("Service ready.")
    yield
    STATE.clear()

app = FastAPI(title="Fraud Detection Service", version="0.1.0", lifespan=lifespan)

def assemble_features(txn: Transaction, store_feats: dict) -> pd.DataFrame:
    hour = txn.step % 24
    row = {
        "amount": txn.amount,
        "is_transfer": int(txn.transaction_type == "TRANSFER"),
        "dest_is_merchant": int(txn.name_dest.startswith("M")),
        "hour": hour,
        "is_night": int(0 <= hour < 6),
        "dest_prior_txn_count": store_feats["dest_prior_txn_count"],
        "dest_prior_amount_sum": store_feats["dest_prior_amount_sum"],
        "dest_prior_amount_mean": store_feats["dest_prior_amount_mean"],
        "dest_is_new": store_feats["dest_is_new"],
        "orig_prior_txn_count": 0,   # cold at serving; degenerate anyway (validated null)
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]

def decide(prob: float, bands: dict) -> tuple[str, str]:
    if prob >= bands["block"]:        return "BLOCK", "model_high_risk"
    if prob >= bands["hold"]:         return "HOLD", "model_elevated_risk"
    if prob >= bands["step_up"]:      return "STEP_UP_AUTH", "model_moderate_risk"
    return "ALLOW", "model_low_risk"

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}

@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    t0 = time.perf_counter()
    store_feats = STATE["store"].lookup(txn.name_dest)

    # 1. Rules first — short-circuit on a hard hit.
    rule_decision, rule_name = apply_rules(txn, store_feats["dest_is_new"])
    if rule_decision is not None:
        return ScoreResponse(
            decision=rule_decision, fraud_probability=-1.0,
            reason=f"rule:{rule_name}", rule_triggered=rule_name,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # 2-3. Assemble + score.
    X = assemble_features(txn, store_feats)
    prob = float(STATE["model"].predict_proba(X)[:, 1][0])

    # 4. Tiered decision.
    decision, reason = decide(prob, STATE["bands"])
    return ScoreResponse(
        decision=decision, fraud_probability=round(prob, 4),
        reason=reason, rule_triggered=None,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )