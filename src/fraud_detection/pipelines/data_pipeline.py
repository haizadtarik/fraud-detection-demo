import argparse, logging
from fraud_detection.config import load_params, PROJECT_ROOT
from fraud_detection.data.load import load_raw, clean
from fraud_detection.data.features import build_features, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("data_pipeline")

def main(source: str):
    params = load_params()

    df = load_raw(source)
    log.info(f"Loaded {len(df):,} rows (source={source})")

    df = clean(df)
    log.info(f"Cleaned -> {len(df):,} modelled rows, fraud rate {df['isFraud'].mean():.4%}")
    inter = PROJECT_ROOT / params["paths"]["intermediate"]
    inter.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(inter, index=False)
    log.info(f"Wrote intermediate -> {inter}")

    feats = build_features(df, params)
    log.info(f"Built {len(FEATURE_COLUMNS)} model features for {len(feats):,} rows")
    out = PROJECT_ROOT / params["paths"]["features"]
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    log.info(f"Wrote feature table -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sample", "full"], default="sample")
    main(ap.parse_args().source)