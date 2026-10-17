"""
F1 Race Intelligence — Machine Learning Lab Pipeline (Phase 2)

Lap-Time Prediction Engine:
1. Target: Relative Lap Time (seconds delta relative to driver's early-race reference pace)
   RelativeLapTimeSec = LapTimeSec - DriverReferencePaceSec.
   Normalizes circuits of varying lap distances (e.g. Monaco 75s vs Spa 105s).
2. Features: Causal lap dynamics without future lookahead:
   - TyreLife (current tyre age in laps)
   - LapsRemaining (fuel burn-off proxy)
   - LapNumber (race progression & track evolution)
   - PrevLapRelativePace (strictly lagged pace delta from t-1)
   - Compound (one-hot encoded: SOFT, MEDIUM, HARD)
3. Models:
   - Naive Reference Pace Baseline (predicts delta = 0.0)
   - Phase 1 Per-Stint Linear Baseline
   - Gradient Boosting Regressor (HistGradientBoostingRegressor)
4. Evaluation:
   - Leave-One-Race-Out (LORO) Cross-Validation across distinct Grand Prix events
   - Reports MAE, RMSE, and R² per held-out race with mean and spread
   - Residual breakdown by tyre age bands and compound
5. Model Persistence:
   - Saves compact joblib artifact with metadata (< 5 MB)
"""

import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

from config import PROJECT_ROOT
from src.data.cleaning import clean_laps

logger = logging.getLogger(__name__)

STANDARD_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lap_pace_model.joblib")


def prepare_ml_features(
    laps: pd.DataFrame,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extracts strictly causal time-series features predicting relative lap pace.

    Target:
      RelativeLapTimeSec = LapTimeSec - DriverReferencePaceSec
      (where DriverReferencePaceSec is the median of driver's first 5 clean laps in that race).

    Features:
      - TyreLife (int)
      - LapsRemaining (int / float)
      - LapNumber (int)
      - PrevLapRelativePace (float, strictly lagged by 1 lap from t-1)
      - Compound_HARD, Compound_MEDIUM, Compound_SOFT (binary one-hot)
    """
    if laps is None or laps.empty:
        return pd.DataFrame(), pd.Series(dtype=float), []

    # Clean laps using robust data cleaning pipeline
    cleaned = clean_laps(laps) if "FuelCorrectedLapTimeSec" not in laps.columns else laps.copy()
    if cleaned.empty or len(cleaned) < 15:
        return pd.DataFrame(), pd.Series(dtype=float), []

    df = cleaned.copy()

    # Exclude wet compounds
    if "Compound" in df.columns:
        df = df[~df["Compound"].astype(str).str.upper().isin({"INTERMEDIATE", "WET"})]
    if "Driver" not in df.columns:
        df["Driver"] = "DRV"

    # Sort strictly by race, driver, and lap number
    sort_cols = ["RaceID", "Driver", "LapNumber"] if "RaceID" in df.columns else ["Driver", "LapNumber"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # 1. Driver Reference Pace
    if "DriverReferencePaceSec" not in df.columns:
        ref_paces = {}
        driver_group = ["RaceID", "Driver"] if "RaceID" in df.columns else ["Driver"]
        for keys, grp in df.groupby(driver_group):
            early = grp.iloc[:min(5, len(grp))]
            ref_paces[keys] = float(early["LapTimeSec"].median())

        if "RaceID" in df.columns:
            df["DriverReferencePaceSec"] = df.apply(
                lambda r: ref_paces.get((r["RaceID"], r["Driver"]), r["LapTimeSec"]), axis=1
            )
        else:
            df["DriverReferencePaceSec"] = df["Driver"].map(ref_paces).fillna(df["LapTimeSec"])

    if "RelativeLapTimeSec" not in df.columns:
        df["RelativeLapTimeSec"] = df["LapTimeSec"] - df["DriverReferencePaceSec"]

    # 2. TyreLife
    if "TyreLife" not in df.columns or df["TyreLife"].isna().all():
        if "Stint" in df.columns:
            stint_grp = ["RaceID", "Driver", "Stint"] if "RaceID" in df.columns else ["Driver", "Stint"]
            df["TyreLife"] = df.groupby(stint_grp).cumcount() + 1
        else:
            drv_grp = ["RaceID", "Driver"] if "RaceID" in df.columns else ["Driver"]
            df["TyreLife"] = df.groupby(drv_grp).cumcount() + 1

    # 3. LapsRemaining (Fuel proxy)
    if "LapsRemaining" not in df.columns:
        race_grp = ["RaceID"] if "RaceID" in df.columns else None
        if race_grp:
            total_laps_map = df.groupby("RaceID")["LapNumber"].max()
            df["LapsRemaining"] = df["RaceID"].map(total_laps_map) - df["LapNumber"]
        else:
            total_laps = df["LapNumber"].max()
            df["LapsRemaining"] = total_laps - df["LapNumber"]
        df["LapsRemaining"] = df["LapsRemaining"].clip(lower=0)

    # 4. Strictly Lagged Previous Lap Relative Pace (No Lookahead!)
    if "PrevLapRelativePace" not in df.columns:
        lag_grp = ["RaceID", "Driver", "Stint"] if ("RaceID" in df.columns and "Stint" in df.columns) else ["Driver", "Stint"] if "Stint" in df.columns else ["Driver"]
        df["PrevLapRelativePace"] = df.groupby(lag_grp)["RelativeLapTimeSec"].shift(1)
        # For the first lap of a stint, previous lap in that stint is unknown; fill with 0.0
        df["PrevLapRelativePace"] = df["PrevLapRelativePace"].fillna(0.0)

    # 5. One-hot compound encoding (fixed vocabulary prevents test-set mismatch)
    compound_series = df["Compound"].astype(str).str.upper() if "Compound" in df.columns else pd.Series("MEDIUM", index=df.index)
    for comp in STANDARD_COMPOUNDS:
        df[f"Compound_{comp}"] = (compound_series == comp).astype(float)

    feature_cols = [
        "TyreLife",
        "LapsRemaining",
        "LapNumber",
        "PrevLapRelativePace",
        "Compound_SOFT",
        "Compound_MEDIUM",
        "Compound_HARD",
    ]

    # Ensure all feature columns exist and are finite
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    X = df[feature_cols].copy()
    y = df["RelativeLapTimeSec"].copy()

    # Pass metadata along as attributes on X for grouped cross-validation
    if "RaceID" in df.columns:
        X["_RaceID"] = df["RaceID"].values
    if "Compound" in df.columns:
        X["_Compound"] = df["Compound"].values

    return X, y, feature_cols


def train_and_evaluate_models(
    laps_data: pd.DataFrame,
    save_artifact_path: Optional[str] = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Trains and cross-validates Naive Baseline, Linear Regression, and HistGradientBoosting.

    Performs Leave-One-Race-Out (LORO) cross validation if multiple races are provided.
    Saves trained production model to disk.
    """
    X, y, feature_names = prepare_ml_features(laps_data)

    if X.empty or len(X) < 20:
        return {
            "success": False,
            "message": f"Insufficient clean dry laps ({len(X)}) for machine learning (minimum required: 20).",
        }

    # Separate metadata columns
    race_ids = X.pop("_RaceID") if "_RaceID" in X.columns else pd.Series("SingleRace", index=X.index)
    compounds = X.pop("_Compound") if "_Compound" in X.columns else pd.Series("UNKNOWN", index=X.index)

    unique_races = race_ids.unique()
    is_multi_race = len(unique_races) >= 2

    cv_results: Dict[str, Any] = {
        "is_multi_race": is_multi_race,
        "n_races": len(unique_races),
        "races": unique_races.tolist(),
        "n_samples": len(X),
        "feature_names": feature_names,
        "naive_baseline": {},
        "linear_model": {},
        "gradient_boosting": {},
        "per_race_eval": [],
        "residual_breakdown": {},
    }

    if is_multi_race:
        # ── Leave-One-Race-Out (LORO) Cross-Validation ──
        logo = LeaveOneGroupOut()
        fold_scores: Dict[str, List[Dict[str, float]]] = {
            "Naive Baseline": [],
            "Linear Regression": [],
            "Gradient Boosting (HGBR)": [],
        }

        all_y_true = []
        all_hgbr_preds = []
        all_linear_preds = []
        all_tyre_life = []
        all_compounds = []

        for train_idx, test_idx in logo.split(X, y, groups=race_ids):
            held_out_race = race_ids.iloc[test_idx].iloc[0]
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # 1. Naive Baseline (predicts 0.0 delta pace)
            pred_naive = np.zeros_like(y_te)
            mae_naive = float(mean_absolute_error(y_te, pred_naive))
            rmse_naive = float(np.sqrt(mean_squared_error(y_te, pred_naive)))
            fold_scores["Naive Baseline"].append({"race": held_out_race, "mae": mae_naive, "rmse": rmse_naive})

            # 2. Linear Model
            lr = Ridge(alpha=1.0)
            lr.fit(X_tr, y_tr)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pred_lr = lr.predict(X_te)
            mae_lr = float(mean_absolute_error(y_te, pred_lr))
            rmse_lr = float(np.sqrt(mean_squared_error(y_te, pred_lr)))
            fold_scores["Linear Regression"].append({"race": held_out_race, "mae": mae_lr, "rmse": rmse_lr})

            # 3. Gradient Boosting
            gb = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_depth=5,
                l2_regularization=1.0,
                random_state=42,
            )
            gb.fit(X_tr, y_tr)
            pred_gb = gb.predict(X_te)
            mae_gb = float(mean_absolute_error(y_te, pred_gb))
            rmse_gb = float(np.sqrt(mean_squared_error(y_te, pred_gb)))
            fold_scores["Gradient Boosting (HGBR)"].append({"race": held_out_race, "mae": mae_gb, "rmse": rmse_gb})

            cv_results["per_race_eval"].append({
                "RaceID": held_out_race,
                "Laps": len(y_te),
                "Naive_MAE": round(mae_naive, 3),
                "Linear_MAE": round(mae_lr, 3),
                "HGBR_MAE": round(mae_gb, 3),
                "Naive_RMSE": round(rmse_naive, 3),
                "Linear_RMSE": round(rmse_lr, 3),
                "HGBR_RMSE": round(rmse_gb, 3),
                "HGBR_vs_Linear_Gain": round(mae_lr - mae_gb, 3),
            })

            all_y_true.extend(y_te.values)
            all_hgbr_preds.extend(pred_gb)
            all_linear_preds.extend(pred_lr)
            all_tyre_life.extend(X_te["TyreLife"].values)
            all_compounds.extend(compounds.iloc[test_idx].values)

        # Aggregate LORO scores
        for m_key, key_target in [
            ("Naive Baseline", "naive_baseline"),
            ("Linear Regression", "linear_model"),
            ("Gradient Boosting (HGBR)", "gradient_boosting"),
        ]:
            maes = [f["mae"] for f in fold_scores[m_key]]
            rmses = [f["rmse"] for f in fold_scores[m_key]]
            cv_results[key_target] = {
                "name": m_key,
                "mean_mae": float(np.mean(maes)),
                "std_mae": float(np.std(maes)),
                "mean_rmse": float(np.mean(rmses)),
                "std_rmse": float(np.std(rmses)),
            }

        # ── Residual Analysis Breakdown ──
        res_df = pd.DataFrame({
            "Actual": all_y_true,
            "Pred_HGBR": all_hgbr_preds,
            "Pred_Linear": all_linear_preds,
            "TyreLife": all_tyre_life,
            "Compound": all_compounds,
        })
        res_df["Residual_HGBR"] = res_df["Actual"] - res_df["Pred_HGBR"]
        res_df["AbsErr_HGBR"] = res_df["Residual_HGBR"].abs()

        # Residual by Tyre Age Band
        bins = [0, 10, 20, 30, 100]
        labels = ["1-10 laps", "11-20 laps", "21-30 laps", "31+ laps"]
        res_df["TyreAgeBand"] = pd.cut(res_df["TyreLife"], bins=bins, labels=labels)
        by_age = res_df.groupby("TyreAgeBand", observed=False)["AbsErr_HGBR"].agg(["count", "mean", "median"]).round(3)
        cv_results["residual_breakdown"]["by_tyre_age"] = by_age.to_dict(orient="index")

        # Residual by Compound
        by_comp = res_df.groupby("Compound")["AbsErr_HGBR"].agg(["count", "mean", "median"]).round(3)
        cv_results["residual_breakdown"]["by_compound"] = by_comp.to_dict(orient="index")

    else:
        # Chronological single-session split fallback
        split_idx = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]

        # Baseline
        pred_naive = np.zeros_like(y_te)
        cv_results["naive_baseline"] = {
            "name": "Naive Reference Pace",
            "mean_mae": float(mean_absolute_error(y_te, pred_naive)),
            "std_mae": 0.0,
            "mean_rmse": float(np.sqrt(mean_squared_error(y_te, pred_naive))),
            "std_rmse": 0.0,
        }

        # Linear
        lr = Ridge(alpha=1.0)
        lr.fit(X_tr, y_tr)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pred_lr = lr.predict(X_te)
        cv_results["linear_model"] = {
            "name": "Linear Regression",
            "mean_mae": float(mean_absolute_error(y_te, pred_lr)),
            "std_mae": 0.0,
            "mean_rmse": float(np.sqrt(mean_squared_error(y_te, pred_lr))),
            "std_rmse": 0.0,
        }

        # HGBR
        gb = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_depth=5,
            l2_regularization=1.0,
            random_state=42,
        )
        gb.fit(X_tr, y_tr)
        pred_gb = gb.predict(X_te)
        cv_results["gradient_boosting"] = {
            "name": "Gradient Boosting (HGBR)",
            "mean_mae": float(mean_absolute_error(y_te, pred_gb)),
            "std_mae": 0.0,
            "mean_rmse": float(np.sqrt(mean_squared_error(y_te, pred_gb))),
            "std_rmse": 0.0,
        }

    # ── Final Production Model Training ──
    final_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=150,
        max_depth=5,
        l2_regularization=1.0,
        random_state=42,
    )
    final_model.fit(X, y)

    # Save compact joblib artifact
    artifact_saved = False
    artifact_size_kb = 0.0
    if save_artifact_path:
        os.makedirs(os.path.dirname(save_artifact_path), exist_ok=True)
        artifact_payload = {
            "model": final_model,
            "feature_names": feature_names,
            "metadata": {
                "created_at": datetime.datetime.utcnow().isoformat(),
                "training_samples": len(X),
                "races": unique_races.tolist(),
                "cv_results": {
                    "naive_baseline": cv_results["naive_baseline"],
                    "linear_model": cv_results["linear_model"],
                    "gradient_boosting": cv_results["gradient_boosting"],
                    "per_race_eval": cv_results.get("per_race_eval", []),
                    "residual_breakdown": cv_results.get("residual_breakdown", {}),
                },
                "target_description": "RelativeLapTimeSec (LapTimeSec minus DriverReferencePaceSec)",
            },
        }
        joblib.dump(artifact_payload, save_artifact_path, compress=3)
        artifact_saved = True
        artifact_size_kb = os.path.getsize(save_artifact_path) / 1024.0
        logger.info(f"Model artifact saved to {save_artifact_path} ({artifact_size_kb:.1f} KB)")

    cv_results["success"] = True
    cv_results["production_model"] = final_model
    cv_results["artifact_saved"] = artifact_saved
    cv_results["artifact_path"] = save_artifact_path
    cv_results["artifact_size_kb"] = round(artifact_size_kb, 1)

    return cv_results


def load_trained_model(model_path: str = DEFAULT_MODEL_PATH) -> Optional[Dict[str, Any]]:
    """Load pre-trained model bundle from disk."""
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None
