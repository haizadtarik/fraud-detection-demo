# Real-Time Transaction Fraud Detection Demo

An end-to-end fraud-detection system built using sample dataset from [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — synthetic mobile-money transactions (transfers, cash-outs, payments)

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
   raw transaction  ─────▶│  RULES LAYER  (instant, explainable)     │
                          │  hard amount cap · known-mule list ·     │
                          │  new-payee + large amount (cooling-off)  │
                          └───────────────┬─────────────────────────┘
                                          │ no hard rule fired
                                          ▼
                          ┌─────────────────────────────────────────┐
                          │  FEATURE ASSEMBLY                        │
                          │  stateless ← request                     │
                          │  stateful  ← recipient feature store     │
                          └───────────────┬─────────────────────────┘
                                          ▼
                          ┌─────────────────────────────────────────┐
                          │  LightGBM  →  fraud probability          │
                          └───────────────┬─────────────────────────┘
                                          ▼
                          ┌─────────────────────────────────────────┐
                          │  TIERED DECISION ENGINE                  │
                          │  ALLOW · STEP_UP_AUTH · HOLD · BLOCK     │
                          └─────────────────────────────────────────┘
```

The **rules-first hybrid** is deliberate: static rules give instant, auditable
blocks for known patterns while the model catches the novel fraud rules cannot enumerate. 
A hard rule short-circuits before the model is even called.

---

## Pipeline stages

| Stage | Command | Output |
|-------|---------|--------|
| Feature engineering (sample) | `make data` | `data/03_features/feature_table.parquet` |
| Feature engineering (full) | `make data-full` | full-prevalence feature table |
| Train + evaluate | `make train` | model bundle + MLflow run |
| Serve | `make serve` | FastAPI scoring API on `:8000` |
| Monitor | `make monitor` | drift report (PSI + Evidently) |
| Lint | `make lint` | linting and formating |

---

## Features

| Feature | Type | Notes |
|---------|------|-------|
| `amount` | stateless | Transaction size — the drain leg; strongest signal |
| `is_transfer` | stateless | TRANSFER vs CASH_OUT |
| `dest_is_merchant` | stateless | Merchants are not the mule pattern |
| `hour`, `is_night` | stateless | Temporal risk window |
| `dest_prior_txn_count` | stateful | Recipient fan-in count |
| `dest_prior_amount_sum` | stateful | Cumulative prior inflow to recipient |
| `dest_prior_amount_mean` | stateful | Average prior inflow size |
| `dest_is_new` | stateful | First-ever payment to this beneficiary |
| `orig_prior_txn_count` | stateful | Origin velocity (validated as a null — see below) |

**Stateless** features derive from the request alone. **Stateful** features require
recipient history and, in production, come from an online feature store — which is
the central serving challenge this repo confronts head-on (below).

---

## Results (full dataset, true prevalence)

Temporal split: train on days ≤ 22, test on days ≥ 23.

| Metric | Value | Reading |
|--------|------:|---------|
| Test prevalence | 2.26% | Later days are denser with fraud than the 0.3% global rate |
| ROC-AUC | 0.930 | Reported, but **not** the headline at this imbalance |
| **PR-AUC** | **0.556** | **Headline** — honest, leakage-free, at true imbalance |

**Reading these numbers honestly:**

Top features by gain: `amount`, `hour`, and the recipient-history group
(`dest_prior_amount_sum` + `dest_prior_amount_mean` + `dest_prior_txn_count`) — the
mule-detection signal. The recipient-history features are correlated and read as a
**group**; their individual ranks are not over-interpreted.

---

## The serving challenge: stateful features and train-serve skew

The hard part of serving this model is not latency — it is **state**. Half the
features summarize a recipient's history, which does not exist in a single
transaction request. In production this is a **feature store** (e.g. Feast over a
Redis online store) keyed by recipient, updated asynchronously after each scored
transaction, with the offline aggregation logic *shared with training* to guarantee
no skew.

This repo implements a **precomputed lookup stand-in** for the online store, and the
serving API is structured so swapping in a real feature store is a single-layer
change. The **train-serve consistency contract** — that the online aggregation
exactly reproduces the training `cumsum` logic — is enforced by a CI test
(`tests/test_train_serve_consistency.py`).

---

## Monitoring

Drift is monitored three ways, because each metric has a blind spot:

- **Feature drift (PSI):** computed from scratch (`src/fraud_detection/monitoring/psi.py`)
  for transparency, plus an Evidently report for the visual artifact. Banking-standard
  bands: < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant. Result: only `hour`
  drifts (0.158) — and it happens to be the second-most-important feature, exactly the
  drift worth watching.
- **Target drift:** the fraud rate jumped 10× between periods, yet PSI reads only
  0.047 ("stable"). This is **PSI's rare-event blind spot** — it weights by absolute
  frequency, so a large relative shift in a tiny class barely registers. The takeaway:
  PSI is necessary but not sufficient for fraud; pair it with direct fraud-rate
  tracking.
- **Prediction (score) drift:** PSI on the model's output distribution, which (unlike
  the rare binary label) has spread and so behaves well — the practical leading
  indicator for the prevalence shift PSI under-weights on the target.

Because fraud labels arrive late (chargebacks, scam reports), drift is the *leading*
indicator and performance decay is the *lagging* one.

---

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# Download PaySim to data/01_raw/ then:
make data-full        
make train       
make serve       # FastAPI scoring API at http://localhost:8000/docs
```

---

## Repository layout

```
├── conf/parameters.yml            # all paths + params (config-driven)
├── data/                          # 01_raw → 02_intermediate → 03_features → 04_models
├── notebooks/01_eda.ipynb         # EDA + empirical data-card verification
├── src/fraud_detection/
│   ├── data/                      # load, clean, feature engineering
│   ├── models/                    # train, evaluate
│   ├── pipelines/                 # runnable data + training pipelines
│   ├── serving/                   # FastAPI app, rules, feature-store stand-in
│   └── monitoring/                # PSI (from scratch) + drift report
├── tests/                         # incl. train-serve reconciliation + leakage gate
├── Dockerfile                     # multi-stage, non-root
└── .github/workflows/ci.yml       # lint → test → build, self-contained
```
