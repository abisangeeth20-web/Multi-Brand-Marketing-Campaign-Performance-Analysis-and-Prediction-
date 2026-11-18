"""End-to-end pipeline: collect -> clean -> engineer features -> EDA -> train/evaluate -> save.

Usage:  python run_pipeline.py            (full run)
        python run_pipeline.py --skip-train
"""
import json
import sys
import time

from src import config as C
from src import eda, features, preprocessing, train


def log(msg):
    print(msg, flush=True)


def main(skip_train=False):
    t0 = time.time()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log("1. Data collection")
    raw = preprocessing.load_raw()
    log(f"   loaded {len(raw):,} rows x {raw.shape[1] - 2} columns from {len(C.RAW_FILES)} brand files")

    log("2. Cleaning & preprocessing")
    clean, clog = preprocessing.clean(raw)
    clean.to_csv(C.CLEANED_CSV, index=False, date_format=C.DATE_FORMAT)
    for k, v in clog.items():
        log(f"   {k}: {v}")

    log("3. Feature engineering")
    feat = features.add_profit_flag(features.add_features(clean))
    feat.to_csv(C.FEATURED_CSV, index=False, date_format=C.DATE_FORMAT)
    clean.to_csv(C.CLEANED_GZ, index=False, date_format=C.DATE_FORMAT, compression="gzip")
    feat.to_csv(C.FEATURED_GZ, index=False, date_format=C.DATE_FORMAT, compression="gzip")
    feat[C.DASHBOARD_COLS].round({"Total_Cost": 2, "Revenue_per_Conversion": 2}).to_csv(
        C.DASHBOARD_GZ, index=False, date_format=C.DATE_FORMAT, compression="gzip")
    log(f"   feature-engineered dataset: {feat.shape}  profit share: {feat.Profit_Flag.mean():.3f}")

    log("4. Exploratory data analysis")
    eda.run_eda(feat)
    log(f"   figures -> {C.FIGURES_DIR}")

    summary = {"cleaning_log": clog, "rows_after_cleaning": len(clean),
               "profit_share": float(feat.Profit_Flag.mean())}
    if not skip_train:
        log("5-6. Model building & evaluation")
        res = train.train_all(feat, log)
        summary["regression_best"] = res["reg_best"]
        summary["classification_best"] = res["clf_best"]
        summary["prelaunch_benchmark"] = res["pre"]
    with open(C.REPORTS_DIR / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(skip_train="--skip-train" in sys.argv)
