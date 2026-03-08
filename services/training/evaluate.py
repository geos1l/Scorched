"""
Phase 7 — Model evaluation + predictions.

INPUTS:
  models/xgboost_heat_model.json     Trained XGBoost model (from Julie)
  models/train_test_split.json       Test cell IDs (from Julie) — {"test_cell_ids": [...]}
  data/processed/features.parquet    Full feature set (from Phase 6)

OUTPUTS:
  data/processed/predictions.parquet   cell_id, predicted_heat, severity
  data/processed/feature_importance.png

Severity buckets (from toronto.json):
  extreme: predicted_heat > +5.0
  high:    +2.0 to +5.0
  moderate: 0.0 to +2.0
  low:     < 0.0

USAGE:
  python -m services.training.evaluate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "xgboost_heat_model.json"
SPLIT_PATH = REPO_ROOT / "models" / "train_test_split.json"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"
PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "predictions.parquet"
IMPORTANCE_PATH = REPO_ROOT / "data" / "processed" / "feature_importance.png"
CONFIG_PATH = REPO_ROOT / "data" / "city_configs" / "toronto.json"

FEATURE_COLS = [
    "seg_building_pct", "seg_road_pct", "seg_vegetation_pct",
    "seg_water_pct", "seg_land_pct", "seg_unlabeled_pct",
    "ndvi_mean", "brightness_mean", "nir_mean",
    "gis_building_coverage", "gis_road_coverage", "gis_park_coverage", "water_distance_m",
]
TARGET_COL = "relative_lst_c"


def assign_severity(predicted_heat: pd.Series, thresholds: dict) -> pd.Series:
    extreme = thresholds["extreme"]
    high = thresholds["high"]
    moderate = thresholds["moderate"]

    def bucket(v):
        if v > extreme:
            return "extreme"
        elif v > high:
            return "high"
        elif v > moderate:
            return "moderate"
        else:
            return "low"

    return predicted_heat.apply(bucket)


def plot_feature_importance(model: xgb.XGBRegressor, feature_names: list[str]) -> None:
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx])
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance")
    plt.tight_layout()
    fig.savefig(IMPORTANCE_PATH, dpi=150)
    plt.close(fig)
    log.info("Feature importance plot saved to %s", IMPORTANCE_PATH)


def main() -> None:
    # Load config for severity thresholds
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    thresholds = config["severity_thresholds"]

    # Load model
    log.info("Loading model from %s", MODEL_PATH)
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    # Load features
    log.info("Loading features from %s", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    log.info("Features loaded: %d rows", len(df))

    # Drop rows missing the target (non-AOI cells won't have relative_lst_c)
    df_valid = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)
    log.info("Rows with complete features + target: %d", len(df_valid))

    # Load test split
    if SPLIT_PATH.exists():
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        test_cell_ids = set(split["test_cell_ids"])
        test_mask = df_valid["cell_id"].isin(test_cell_ids)
        df_test = df_valid[test_mask]
        log.info("Test set: %d cells", len(df_test))

        if len(df_test) > 0:
            X_test = df_test[FEATURE_COLS].values
            y_test = df_test[TARGET_COL].values
            y_pred_test = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred_test)
            log.info("=== Evaluation Metrics ===")
            log.info("MAE:  %.4f °C", mae)
            log.info("RMSE: %.4f °C", rmse)
            log.info("R²:   %.4f", r2)
    else:
        log.warning("train_test_split.json not found — skipping evaluation metrics")

    # Predict on all valid cells
    X_all = df_valid[FEATURE_COLS].values
    predictions = model.predict(X_all)

    df_valid = df_valid.copy()
    df_valid["predicted_heat"] = predictions
    df_valid["severity"] = assign_severity(df_valid["predicted_heat"], thresholds)

    log.info("Severity distribution:\n%s", df_valid["severity"].value_counts().to_string())

    # Output predictions.parquet — all cells, NaN predicted_heat for cells with missing features
    out = df[["cell_id"]].copy()
    pred_map = df_valid.set_index("cell_id")[["predicted_heat", "severity"]]
    out = out.join(pred_map, on="cell_id")
    out.to_parquet(PREDICTIONS_PATH, index=False)
    log.info("Written: %s (%d rows)", PREDICTIONS_PATH, len(out))

    # Feature importance plot
    plot_feature_importance(model, FEATURE_COLS)

    # Acceptance criteria check
    assert set(out["severity"].dropna().unique()) <= {"low", "moderate", "high", "extreme"}
    log.info("All severity values valid ✓")


if __name__ == "__main__":
    main()
