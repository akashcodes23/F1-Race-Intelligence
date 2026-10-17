"""
F1 Race Intelligence — Machine Learning Lab Pipeline

Robust lap-time modeling engine adhering to motorsport ML best practices:
- One-hot compound encoding (avoids ordinal bias)
- Strictly chronological train/test split (no lookahead / future leakage)
- Dummy baseline comparator (sample mean)
- Linear Regression & Random Forest regressors
- TimeSeriesSplit cross-validation on training partition only
- Residual analysis, prediction vs actual, and feature importances
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from config import (
    CV_SPLITS,
    MIN_LAPS_FOR_ML,
    TEST_SIZE,
)
from src.data_validator import clean_laps

logger = logging.getLogger(__name__)


def prepare_ml_features(laps: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extracts time-series features from cleaned lap data.
    """
    cleaned = clean_laps(laps)
    if cleaned.empty or len(cleaned) < MIN_LAPS_FOR_ML:
        return pd.DataFrame(), pd.Series(dtype=float), []

    df = cleaned.sort_values("LapNumber").copy()

    # Calculate tyre age within stint
    if "Stint" in df.columns:
        df["TireAge"] = df.groupby("Stint").cumcount() + 1
    else:
        df["TireAge"] = np.arange(1, len(df) + 1)

    features = ["LapNumber", "TireAge"]

    # One-hot encode compound if available
    if "Compound" in df.columns and df["Compound"].nunique() > 1:
        compound_dummies = pd.get_dummies(df["Compound"], prefix="Compound", dtype=float)
        df = pd.concat([df, compound_dummies], axis=1)
        features.extend(compound_dummies.columns.tolist())

    X = df[features].copy()
    y = df["LapTimeSec"].copy()

    return X, y, features


def train_and_evaluate_models(
    laps: pd.DataFrame,
    test_size: float = TEST_SIZE,
    cv_splits: int = CV_SPLITS,
) -> Dict[str, Any]:
    """
    Trains baseline, Linear Regression, and Random Forest models with chronological evaluation.
    """
    X, y, feature_names = prepare_ml_features(laps)

    if X.empty or len(X) < MIN_LAPS_FOR_ML:
        return {
            "success": False,
            "message": f"Insufficient clean laps ({len(X)}) for machine learning. Minimum required: {MIN_LAPS_FOR_ML}.",
        }

    n_samples = len(X)
    split_idx = int(n_samples * (1.0 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 1. Baseline Model (Mean Lap Time)
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_r2 = float(r2_score(y_test, baseline_pred))

    model_candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6),
    }

    results = {
        "success": True,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_names": feature_names,
        "baseline": {
            "name": "Chronological Mean Baseline",
            "mae": baseline_mae,
            "rmse": baseline_rmse,
            "r2": baseline_r2,
        },
        "models": {},
    }

    for name, model in model_candidates.items():
        # Train on train set
        model.fit(X_train, y_train)

        # Predict
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        test_mae = float(mean_absolute_error(y_test, test_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
        test_r2 = float(r2_score(y_test, test_pred))

        # TimeSeriesSplit cross validation on training set only
        actual_cv = min(cv_splits, max(2, len(X_train) // 8))
        tscv = TimeSeriesSplit(n_splits=actual_cv)
        cv_scores = []
        for tr_i, val_i in tscv.split(X_train):
            clone_model = LinearRegression() if name == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
            clone_model.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
            val_p = clone_model.predict(X_train.iloc[val_i])
            cv_scores.append(r2_score(y_train.iloc[val_i], val_p))

        # Feature importances
        importances = {}
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(feature_names, [float(x) for x in model.feature_importances_]))
        elif hasattr(model, "coef_"):
            importances = dict(zip(feature_names, [float(x) for x in model.coef_]))

        residuals = y_test.values - test_pred

        results["models"][name] = {
            "model_obj": model,
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "cv_r2_mean": float(np.mean(cv_scores)) if cv_scores else None,
            "cv_r2_std": float(np.std(cv_scores)) if cv_scores else None,
            "importances": importances,
            "y_test": y_test.values,
            "test_predictions": test_pred,
            "test_laps": X_test["LapNumber"].values,
            "residuals": residuals,
        }

    return results
