"""Population Stability Index — drift metric, computed from scratch.

PSI = Σ (actual%% - expected%%) * ln(actual%% / expected%%)  over bins.
Interpretation:
  < 0.10  stable
  0.10-0.25  moderate shift — investigate
  > 0.25  significant shift — likely retrain
"""

import numpy as np
import pandas as pd


def _psi_for_feature(expected, actual, bins=10, eps=1e-6):
    """PSI for one numeric feature. Bin edges set on the EXPECTED (reference) dist."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant feature — no drift definable
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    exp_pct = np.clip(exp_pct, eps, None)  # avoid div0 / log0 in empty bins
    act_pct = np.clip(act_pct, eps, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, features, bins=10):
    """PSI per feature, with a banding label. Returns a tidy DataFrame."""
    rows = []
    for f in features:
        psi = _psi_for_feature(reference[f], current[f], bins=bins)
        band = "stable" if psi < 0.10 else "moderate_shift" if psi < 0.25 else "significant_shift"
        rows.append({"feature": f, "psi": round(psi, 4), "band": band})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
