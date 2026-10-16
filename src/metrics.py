"""
F1 Race Intelligence — Analytical Metrics and ML Pipeline

Contains:  feature engineering, model training & evaluation,
           pit-strategy simulation, degradation estimation,
           and consistency metrics.

Bug-fixes applied in Phase 1:
  • Removed duplicate imports and duplicate calculate_degradation.
  • CompoundEncoded replaced with proper one-hot encoding.
  • CV now runs on training set only (prevents test-data leakage).
  • Pit-strategy simulation uses dynamic feature columns.
  • evaluate_models uses the same feature set as training.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from config import (
    CV_SPLITS,
    DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    MIN_LAPS_FOR_ANALYSIS,
    MIN_LAPS_FOR_ML,
    TEST_SIZE,
)
from src.data_validator import clean_laps

logger = logging.getLogger(__name__)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def prepare_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML-ready features from cleaned lap data.

    Features
    --------
    LapNumber : sequential lap number
    TireAge   : laps on current tyre set (within stint)
    Compound_*: one-hot encoded tyre compound

    Target
    ------
    LapTimeSec : lap time in seconds

    Returns an empty DataFrame when cleaning yields too few laps.
    """
    cleaned = clean_laps(laps)
    if cleaned.empty:
        return pd.DataFrame()

    data = cleaned.copy()

    # Tyre age within each stint
    if "Stint" in data.columns:
        data["TireAge"] = data.groupby("Stint").cumcount() + 1
    else:
        data["TireAge"] = range(1, len(data) + 1)

    # One-hot encode compound (replaces the old .cat.codes approach)
    result_cols = ["LapNumber", "TireAge", "LapTimeSec"]

    if "Compound" in data.columns:
        dummies = pd.get_dummies(data["Compound"], prefix="Compound", dtype=int)
        data = pd.concat([data, dummies], axis=1)
        result_cols.extend(dummies.columns.tolist())

    available = [c for c in result_cols if c in data.columns]
    return data[available].reset_index(drop=True)


def get_feature_columns(data: pd.DataFrame) -> list:
    """Return feature column names (everything except the target)."""
    return [c for c in data.columns if c != "LapTimeSec"]


# =========================================================
# MODEL TRAINING
# =========================================================

def train_lap_models(laps: pd.DataFrame) -> dict:
    """
    Train Linear Regression and Random Forest regressors on lap data.

    Methodology
    -----------
    * Chronological 80/20 train-test split (no shuffling).
    * TimeSeriesSplit CV on **training set only** to avoid leaking
      test observations into the validation folds.
    * MAE and RMSE computed on held-out test set.

    Returns
    -------
    dict  keyed by model name ('linear', 'random_forest'),
          each containing the fitted model, feature column list,
          and evaluation metrics.

    Raises
    ------
    ValueError  if fewer than ``MIN_LAPS_FOR_ML`` laps remain after cleaning.
    """
    data = prepare_features(laps)

    if data.empty or len(data) < MIN_LAPS_FOR_ML:
        raise ValueError(
            f"Insufficient data for training: {len(data)} laps "
            f"(minimum: {MIN_LAPS_FOR_ML})"
        )

    feature_cols = get_feature_columns(data)
    X = data[feature_cols]
    y = data["LapTimeSec"]

    logger.info(f"Training on {len(X)} laps — features: {feature_cols}")

    # Chronological split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )

    results = {}

    models = [
        ("linear", LinearRegression, {}),
        (
            "random_forest",
            RandomForestRegressor,
            {"n_estimators": 200, "random_state": 42},
        ),
    ]

    for name, cls, kwargs in models:
        # ── Fit final model on training set ──
        model = cls(**kwargs)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # ── CV on training data only ──
        n_cv_splits = min(CV_SPLITS, max(2, len(X_train) // 10))
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)

        cv_scores = []
        for tr_idx, va_idx in tscv.split(X_train):
            cv_model = cls(**kwargs)
            cv_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = cv_model.predict(X_train.iloc[va_idx])
            if len(set(y_train.iloc[va_idx])) > 1:
                cv_scores.append(r2_score(y_train.iloc[va_idx], pred))

        results[name] = {
            "model": model,
            "feature_cols": feature_cols,
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(y_test, test_pred),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "cv_r2_mean": float(np.mean(cv_scores)) if cv_scores else None,
            "cv_r2_std": float(np.std(cv_scores)) if cv_scores else None,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "y_test": y_test.values,
            "y_test_index": y_test.index.tolist(),
            "test_pred": test_pred,
        }

    return results


# =========================================================
# PIT STRATEGY SIMULATION
# =========================================================

def _build_prediction_row(
    lap_number: int,
    tire_age: int,
    feature_cols: list,
    compound_col: str = None,
) -> pd.DataFrame:
    """Construct a single-row DataFrame matching the model's feature set."""
    row = {"LapNumber": lap_number, "TireAge": tire_age}
    for col in feature_cols:
        if col.startswith("Compound_"):
            row[col] = 0
    if compound_col and compound_col in feature_cols:
        row[compound_col] = 1
    return pd.DataFrame([row])[feature_cols]


def simulate_pit_strategy(
    model,
    feature_cols: list,
    current_lap: int,
    current_tire_age: int,
    pit_penalty: float = 22.0,
    future_laps: int = 10,
    compound_col: str = None,
) -> tuple:
    """
    Compare stay-out vs pit-now over *future_laps*.

    Returns (stay_out_total, pit_total) cumulative projected seconds.
    """
    stay_total = 0.0
    pit_total = float(pit_penalty)

    for i in range(1, future_laps + 1):
        lap_num = current_lap + i

        stay_df = _build_prediction_row(
            lap_num, current_tire_age + i, feature_cols, compound_col
        )
        pit_df = _build_prediction_row(
            lap_num, i, feature_cols, compound_col
        )

        stay_total += float(model.predict(stay_df)[0])
        pit_total += float(model.predict(pit_df)[0])

    return stay_total, pit_total


def optimize_pit_window(
    model,
    feature_cols: list,
    start_lap: int,
    end_lap: int,
    current_tire_age: int,
    pit_penalty: float = 22.0,
    evaluation_window: int = 10,
    compound_col: str = None,
) -> tuple:
    """
    Evaluate every candidate pit lap in [start_lap, end_lap].

    For each candidate the function projects *evaluation_window* future laps
    and compares the pit-stop strategy (penalty + fresh tyres) against
    the stay-out strategy (aging tyres, no penalty).

    Returns (best_pit_lap, results_DataFrame).
    """
    results = []

    for pit_lap in range(int(start_lap), int(end_lap) + 1):
        stay_total = 0.0
        pit_total = float(pit_penalty)
        tire_age_at_pit = current_tire_age + (pit_lap - int(start_lap))

        for i in range(1, evaluation_window + 1):
            lap_num = pit_lap + i

            stay_df = _build_prediction_row(
                lap_num, tire_age_at_pit + i, feature_cols, compound_col
            )
            pit_df = _build_prediction_row(
                lap_num, i, feature_cols, compound_col
            )

            stay_total += float(model.predict(stay_df)[0])
            pit_total += float(model.predict(pit_df)[0])

        results.append(
            {
                "PitLap": pit_lap,
                "StayOutTime": round(stay_total, 3),
                "PitTime": round(pit_total, 3),
                "Advantage": round(stay_total - pit_total, 3),
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        return int(start_lap), df

    best_idx = df["Advantage"].idxmax()
    return int(df.loc[best_idx, "PitLap"]), df


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model, feature_cols: list) -> dict:
    """Extract feature importances (tree models) or coefficients (linear)."""
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_cols, model.feature_importances_))
    if hasattr(model, "coef_"):
        return dict(zip(feature_cols, model.coef_))
    return {}


# =========================================================
# DEGRADATION & CONSISTENCY
# =========================================================

def calculate_degradation(laps: pd.DataFrame, fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP) -> tuple:
    """
    Robust tyre degradation estimation using Huber regression.

    Returns (slope, intercept, r², n_laps).
    """
    from src.tyre_analysis import compute_stint_degradation

    res = compute_stint_degradation(laps, fuel_effect=fuel_effect, use_fuel_correction=True)
    if not res["valid"]:
        return 0.0, 0.0, 0.0, 0

    return (
        float(res["slope"]),
        float(res["intercept"]),
        float(res["r2"]),
        int(res["n_laps"]),
    )


def calculate_consistency(laps: pd.DataFrame) -> float:
    """
    Standard deviation of lap times (seconds).
    Lower → more consistent.
    """
    cleaned = clean_laps(laps)
    if len(cleaned) < MIN_LAPS_FOR_ANALYSIS:
        return 0.0
    return float(cleaned["LapTimeSec"].std())


# =========================================================
# CONVENIENCE
# =========================================================

def get_training_data(laps: pd.DataFrame):
    """Return (X, y) ready for external model training / inspection."""
    data = prepare_features(laps)
    if data.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    cols = get_feature_columns(data)
    return data[cols], data["LapTimeSec"]
