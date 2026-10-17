"""
F1 Race Intelligence — Machine Learning Lab Pipeline (Phase 2.5)

Lap-Time Prediction Engine:
1. Target: Relative Lap Time (seconds delta relative to driver's early-race reference pace)
   RelativeLapTimeSec = LapTimeSec - DriverReferencePaceSec.
   Normalizes circuits of varying lap distances (e.g. Monaco 75s vs Spa 105s).
   Computed strictly from clean laps in laps 2-6 (excluding damage/off-pace excursions).
   Trained and evaluated ONLY on laps >= 7 (zero target leakage).
2. Features: Strictly causal lap dynamics without future lookahead:
   - TyreLife (current tyre age in laps; filled via stint cumcount)
   - LapsRemaining (fuel burn-off proxy)
   - LapNumber (race progression & track evolution)
   - PrevLapRelativePace (strictly lagged pace delta from t-1 within stint)
   - Compound (one-hot encoded: SOFT, MEDIUM, HARD)
3. Models & Baselines:
   - Naive Reference Pace Baseline (predicts delta = 0.0)
   - Persistence Baseline (predicts previous lap's relative pace)
   - Phase 1 Per-Stint Huber Degradation Baseline (fit only on prior laps of stint with fuel correction)
   - Scaled Ridge Regression (StandardScaler + Ridge in a Pipeline)
   - HistGradientBoosting Variant A (1-step autoregressive with PrevLapRelativePace)
   - HistGradientBoosting Variant B (multi-step static origin without PrevLapRelativePace)
4. Evaluation:
   - Leave-One-Race-Out (LORO) Cross-Validation across distinct Grand Prix events
   - Reports MAE, RMSE per held-out race with mean and spread
   - Multi-step forecast horizon evaluation (1, 5, 10, 20 laps ahead)
   - Residual breakdown by tyre age bands and compound with exact sample sizes N
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
from scipy.optimize import lsq_linear
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_FUEL_EFFECT_SEC_PER_LAP, PROJECT_ROOT
from src.data.cleaning import clean_laps

logger = logging.getLogger(__name__)

STANDARD_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lap_pace_model.joblib")


def prepare_ml_features(
    laps: pd.DataFrame,
    min_eval_lap: int = 7,
    include_prev_pace: bool = True,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extracts strictly causal time-series features predicting relative lap pace.

    Target:
      RelativeLapTimeSec = LapTimeSec - DriverReferencePaceSec
      (DriverReferencePaceSec is computed strictly from laps 2-6).

    Anti-Leakage Constraint (Phase 2.5):
      - Evaluated and trained ONLY on laps >= min_eval_lap (default: 7).
      - Laps 2-6 are never evaluated or trained on.
      - PrevLapRelativePace is strictly lagged from t-1 within the current stint.

    Features:
      - TyreLife (int)
      - LapsRemaining (int / float)
      - LapNumber (int)
      - PrevLapRelativePace (float, strictly lagged by 1 lap from t-1; optional)
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

    # 1. Driver Reference Pace: computed strictly from clean laps in laps 2-6
    if "DriverReferencePaceSec" not in df.columns:
        ref_paces = {}
        if "RaceID" in df.columns:
            for race_id, r_grp in df.groupby("RaceID"):
                sess_median = float(r_grp["LapTimeSec"].median())
                for drv, grp in r_grp.groupby("Driver"):
                    # Use laps 2-6; filter damaged/off-pace laps > 1.15 * session median
                    early = grp[(grp["LapNumber"] >= 2) & (grp["LapNumber"] <= 6) & (grp["LapTimeSec"] < sess_median * 1.15)]
                    if len(early) < 2:
                        early = grp[(grp["LapTimeSec"] < sess_median * 1.15)].iloc[:min(5, len(grp))]
                    if early.empty:
                        early = grp.iloc[:min(5, len(grp))]
                    ref_paces[(race_id, drv)] = float(early["LapTimeSec"].median())

            df["DriverReferencePaceSec"] = df.apply(
                lambda r: ref_paces.get((r["RaceID"], r["Driver"]), r["LapTimeSec"]), axis=1
            )
        else:
            sess_median = float(df["LapTimeSec"].median())
            for drv, grp in df.groupby("Driver"):
                early = grp[(grp["LapNumber"] >= 2) & (grp["LapNumber"] <= 6) & (grp["LapTimeSec"] < sess_median * 1.15)]
                if len(early) < 2:
                    early = grp[(grp["LapTimeSec"] < sess_median * 1.15)].iloc[:min(5, len(grp))]
                if early.empty:
                    early = grp.iloc[:min(5, len(grp))]
                ref_paces[drv] = float(early["LapTimeSec"].median())
            df["DriverReferencePaceSec"] = df["Driver"].map(ref_paces).fillna(df["LapTimeSec"])

    if "RelativeLapTimeSec" not in df.columns:
        df["RelativeLapTimeSec"] = df["LapTimeSec"] - df["DriverReferencePaceSec"]

    # 2. TyreLife: fill missing values using cumulative stint lap count
    stint_grp = ["RaceID", "Driver", "Stint"] if ("RaceID" in df.columns and "Stint" in df.columns) else ["Driver", "Stint"] if "Stint" in df.columns else ["Driver"]
    calc_tyre_life = df.groupby(stint_grp).cumcount() + 1
    if "TyreLife" not in df.columns:
        df["TyreLife"] = calc_tyre_life
    else:
        df["TyreLife"] = df["TyreLife"].fillna(calc_tyre_life)

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

    # 4. Strictly Lagged Previous Lap Relative Pace (computed BEFORE filtering laps >= 7)
    if "PrevLapRelativePace" not in df.columns:
        df["PrevLapRelativePace"] = df.groupby(stint_grp)["RelativeLapTimeSec"].shift(1)
        # For the first lap of a stint, previous lap in that stint is unknown; fill with 0.0
        df["PrevLapRelativePace"] = df["PrevLapRelativePace"].fillna(0.0)

    # 5. One-hot compound encoding (fixed vocabulary prevents test-set mismatch)
    compound_series = df["Compound"].astype(str).str.upper() if "Compound" in df.columns else pd.Series("MEDIUM", index=df.index)
    for comp in STANDARD_COMPOUNDS:
        df[f"Compound_{comp}"] = (compound_series == comp).astype(float)

    # ── Phase 2.5 Anti-Leakage Filtering: Train and evaluate strictly on laps >= min_eval_lap ──
    if min_eval_lap > 0 and "LapNumber" in df.columns:
        df = df[df["LapNumber"] >= min_eval_lap].copy().reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), []

    if include_prev_pace:
        feature_cols = [
            "TyreLife",
            "LapsRemaining",
            "LapNumber",
            "PrevLapRelativePace",
            "Compound_SOFT",
            "Compound_MEDIUM",
            "Compound_HARD",
        ]
    else:
        feature_cols = [
            "TyreLife",
            "LapsRemaining",
            "LapNumber",
            "Compound_SOFT",
            "Compound_MEDIUM",
            "Compound_HARD",
        ]

    # Ensure all feature columns exist, are float64, and are finite
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0).astype(np.float64)

    X = df[feature_cols].copy()
    y = df["RelativeLapTimeSec"].astype(np.float64).copy()

    # Pass metadata along as attributes on X for grouped cross-validation
    if "RaceID" in df.columns:
        X["_RaceID"] = df["RaceID"].values
    if "Compound" in df.columns:
        X["_Compound"] = df["Compound"].values
    if "Driver" in df.columns:
        X["_Driver"] = df["Driver"].values
    if "Stint" in df.columns:
        X["_Stint"] = df["Stint"].values
    if "LapTimeSec" in df.columns:
        X["_LapTimeSec"] = df["LapTimeSec"].values
    if "DriverReferencePaceSec" in df.columns:
        X["_DriverReferencePaceSec"] = df["DriverReferencePaceSec"].values

    return X, y, feature_cols


def fit_s1(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> Dict[str, float]:
    """Fit per-compound linear degradation slope on fuel-corrected relative pace."""
    slopes = {}
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        c_mask = (X_tr[f"Compound_{comp}"] == 1.0)
        if c_mask.sum() < 10:
            slopes[comp] = 0.04
            continue
        fuel_term = fuel_effect * (X_tr.loc[c_mask, "LapsRemaining"] - (X_tr.loc[c_mask, "LapsRemaining"] + X_tr.loc[c_mask, "LapNumber"] - 4.0))
        y_adj = y_tr[c_mask] - fuel_term
        tl = X_tr.loc[c_mask, "TyreLife"].values
        deg_slope = float(np.sum(tl * y_adj.values) / np.maximum(np.sum(tl ** 2), 1e-6))
        slopes[comp] = max(0.005, deg_slope)
    return slopes


def predict_s1(
    X_te: pd.DataFrame,
    slopes: Dict[str, float],
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> np.ndarray:
    """Predict relative lap pace using S1 physics linear baseline."""
    preds = np.zeros(len(X_te))
    fuel_term = fuel_effect * (X_te["LapsRemaining"] - (X_te["LapsRemaining"] + X_te["LapNumber"] - 4.0))
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        c_mask = (X_te[f"Compound_{comp}"] == 1.0)
        if c_mask.sum() > 0:
            deg = slopes.get(comp, 0.04)
            preds[c_mask] = fuel_term[c_mask] + deg * X_te.loc[c_mask, "TyreLife"]
    return preds


def fit_s1a(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Fit S1a components on training fold:
    1. Per-compound monotone degradation curves deg(c, age) = a1*age + a2*age^2 (monotone non-decreasing).
    2. Compound offsets relative to MEDIUM.
    """
    compounds = ["SOFT", "MEDIUM", "HARD"]
    deg_params = {}
    offsets = {}

    total_laps_proxy = X_tr["LapsRemaining"] + X_tr["LapNumber"]
    ref_laps_rem = total_laps_proxy - 4.0
    fuel_adj_pace = y_tr + fuel_effect * (X_tr["LapsRemaining"] - ref_laps_rem)

    for c in compounds:
        col = f"Compound_{c}"
        if col in X_tr.columns:
            c_mask = (X_tr[col] == 1.0)
        else:
            c_mask = pd.Series(True, index=X_tr.index)

        if c_mask.sum() >= 20:
            age = X_tr.loc[c_mask, "TyreLife"].values.astype(float)
            y_sub = fuel_adj_pace[c_mask].values.astype(float)

            early_mask = age <= 5
            base = float(np.median(y_sub[early_mask])) if early_mask.sum() >= 5 else float(np.median(y_sub))
            y_deg = np.maximum(0.0, y_sub - base)

            A = np.column_stack([age, age**2])
            res = lsq_linear(A, y_deg, bounds=(0, np.inf))
            deg_params[c] = {"a1": float(res.x[0]), "a2": float(res.x[1]), "base": base}
        else:
            deg_params[c] = {"a1": 0.035, "a2": 0.0005, "base": 0.0}

    med_base = deg_params.get("MEDIUM", {}).get("base", 0.0)
    for c in compounds:
        offsets[c] = deg_params.get(c, {}).get("base", 0.0) - med_base

    return deg_params, offsets


def calc_s1a_deg(comp: str, age: float, deg_params: Dict[str, Dict[str, float]]) -> float:
    """Evaluate monotone quadratic degradation curve deg(c, age) = a1*age + a2*age^2."""
    params = deg_params.get(comp.upper(), deg_params.get("MEDIUM", {"a1": 0.035, "a2": 0.0005}))
    return float(params["a1"] * age + params["a2"] * (age**2))


def predict_s1a_stint(
    anchor: float,
    comp: str,
    ages: np.ndarray,
    laps_rem: np.ndarray,
    ref_laps_rem: float,
    deg_params: Dict[str, Dict[str, float]],
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> np.ndarray:
    """Predict relative pace along a stint timeline anchored to current driver pace."""
    preds = []
    for a, rem in zip(ages, laps_rem):
        deg = calc_s1a_deg(comp, float(a), deg_params)
        fuel_term = -fuel_effect * (float(rem) - ref_laps_rem)
        preds.append(anchor + fuel_term + deg)
    return np.array(preds)


def train_and_evaluate_models(
    laps_data: pd.DataFrame,
    save_artifact_path: Optional[str] = DEFAULT_MODEL_PATH,
    fuel_constant: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> Dict[str, Any]:
    """
    Trains and cross-validates Naive Baseline, Persistence Baseline, Phase 1 Huber Per-Stint Baseline,
    Scaled Ridge, and HistGradientBoosting (Variant A 1-step and Variant B multi-step).

    Performs Leave-One-Race-Out (LORO) cross validation if multiple races are provided.
    Saves trained production models to disk with full CV metrics.
    """
    # Extract features for Variant A (with PrevLapRelativePace)
    X_A, y, feature_names_A = prepare_ml_features(laps_data, min_eval_lap=7, include_prev_pace=True)

    if X_A.empty or len(X_A) < 20:
        return {
            "success": False,
            "message": f"Insufficient clean dry laps ({len(X_A)}) for machine learning (minimum required: 20 on laps >= 7).",
        }

    # Extract features for Variant B (multi-step without PrevLapRelativePace)
    feature_names_B = [c for c in feature_names_A if c != "PrevLapRelativePace"]

    # Separate metadata columns
    race_ids = X_A.pop("_RaceID") if "_RaceID" in X_A.columns else pd.Series("SingleRace", index=X_A.index)
    compounds = X_A.pop("_Compound") if "_Compound" in X_A.columns else pd.Series("UNKNOWN", index=X_A.index)
    drivers = X_A.pop("_Driver") if "_Driver" in X_A.columns else pd.Series("DRV", index=X_A.index)
    stints = X_A.pop("_Stint") if "_Stint" in X_A.columns else pd.Series(1, index=X_A.index)
    lap_times_sec = X_A.pop("_LapTimeSec") if "_LapTimeSec" in X_A.columns else None
    ref_paces_sec = X_A.pop("_DriverReferencePaceSec") if "_DriverReferencePaceSec" in X_A.columns else None

    X_B = X_A[feature_names_B].copy()

    unique_races = race_ids.unique()
    is_multi_race = len(unique_races) >= 2

    cv_results: Dict[str, Any] = {
        "is_multi_race": is_multi_race,
        "n_races": len(unique_races),
        "races": unique_races.tolist(),
        "n_samples": len(X_A),
        "feature_names_A": feature_names_A,
        "feature_names_B": feature_names_B,
        "naive_baseline": {},
        "persistence_baseline": {},
        "huber_per_stint": {},
        "scaled_ridge": {},
        "gradient_boosting_A": {},
        "gradient_boosting_B": {},
        "per_race_eval": [],
        "residual_breakdown": {},
        "horizon_evaluation": {},
    }

    if is_multi_race:
        # ── Leave-One-Race-Out (LORO) Cross-Validation ──
        logo = LeaveOneGroupOut()
        fold_scores: Dict[str, List[Dict[str, float]]] = {
            "Naive Baseline": [],
            "Persistence": [],
            "Huber Per-Stint": [],
            "S1 Physics Linear": [],
            "S2 Scaled Ridge": [],
            "Scaled Ridge": [],
            "HGBR Variant A": [],
            "HGBR Variant B": [],
            "HGBR Variant B Monotonic": [],
        }

        all_y_true = []
        all_hgbr_preds = []
        all_tyre_life = []
        all_compounds = []

        # Multi-origin horizon evaluations (H = 1, 5, 10, 20 laps)
        horizons = [1, 5, 10, 20]
        target_origins = [3, 8, 15]
        horizon_records: Dict[str, Dict[int, Dict[str, List[float]]]] = {
            m: {h: {"abs_err": [], "cum_err": []} for h in horizons}
            for m in [
                "Persistence",
                "Huber (Prior Stint)",
                "S1 Physics Linear",
                "S2 Scaled Ridge",
                "Variant A (Recursive)",
                "Variant B (Unconstrained)",
                "Variant B (Monotonic)",
            ]
        }

        for train_idx, test_idx in logo.split(X_A, y, groups=race_ids):
            held_out_race = race_ids.iloc[test_idx].iloc[0]
            X_A_tr, X_A_te = X_A.iloc[train_idx], X_A.iloc[test_idx]
            X_B_tr, X_B_te = X_B.iloc[train_idx], X_B.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # 1. Naive Baseline (predicts 0.0 delta pace)
            pred_naive = np.zeros_like(y_te)
            mae_naive = float(mean_absolute_error(y_te, pred_naive))
            rmse_naive = float(np.sqrt(mean_squared_error(y_te, pred_naive)))
            fold_scores["Naive Baseline"].append({"race": held_out_race, "mae": mae_naive, "rmse": rmse_naive})

            # 2. Persistence Baseline (predicts previous lap's relative pace)
            pred_pers = X_A_te["PrevLapRelativePace"].values
            mae_pers = float(mean_absolute_error(y_te, pred_pers))
            rmse_pers = float(np.sqrt(mean_squared_error(y_te, pred_pers)))
            fold_scores["Persistence"].append({"race": held_out_race, "mae": mae_pers, "rmse": rmse_pers})

            # 3. Phase 1 Huber Per-Stint Baseline (extrapolating fuel-corrected Huber on prior laps of stint)
            pred_huber = []
            test_df_fold = pd.DataFrame({
                "Driver": drivers.iloc[test_idx].values,
                "Stint": stints.iloc[test_idx].values,
                "LapNumber": X_A_te["LapNumber"].values,
                "TyreLife": X_A_te["TyreLife"].values,
                "LapsRemaining": X_A_te["LapsRemaining"].values,
                "RelativeLapTimeSec": y_te.values,
                "PrevLapRelativePace": X_A_te["PrevLapRelativePace"].values,
            })
            if lap_times_sec is not None and ref_paces_sec is not None:
                test_df_fold["LapTimeSec"] = lap_times_sec.iloc[test_idx].values
                test_df_fold["DriverReferencePaceSec"] = ref_paces_sec.iloc[test_idx].values
                test_df_fold["FuelCorrectedLapTimeSec"] = test_df_fold["LapTimeSec"] - (fuel_constant * test_df_fold["LapsRemaining"])
            else:
                test_df_fold["FuelCorrectedLapTimeSec"] = y_te.values - (fuel_constant * test_df_fold["LapsRemaining"])
                test_df_fold["DriverReferencePaceSec"] = 0.0

            for _, r in test_df_fold.iterrows():
                drv_m, st_m, lap_m, rem_m, ref_m = r["Driver"], r["Stint"], r["LapNumber"], r["LapsRemaining"], r["DriverReferencePaceSec"]
                prior = test_df_fold[
                    (test_df_fold["Driver"] == drv_m)
                    & (test_df_fold["Stint"] == st_m)
                    & (test_df_fold["LapNumber"] < lap_m)
                ]
                if len(prior) >= 4:
                    try:
                        hr = HuberRegressor()
                        hr.fit(prior["LapNumber"].values.reshape(-1, 1), prior["FuelCorrectedLapTimeSec"].values)
                        p_fuel = hr.predict([[lap_m]])[0]
                        p_lap = p_fuel + (fuel_constant * rem_m)
                        pred_huber.append(float(p_lap - ref_m))
                    except Exception:
                        pred_huber.append(float(r["PrevLapRelativePace"]))
                else:
                    pred_huber.append(float(r["PrevLapRelativePace"]))

            pred_huber_arr = np.array(pred_huber)
            mae_huber = float(mean_absolute_error(y_te, pred_huber_arr))
            rmse_huber = float(np.sqrt(mean_squared_error(y_te, pred_huber_arr)))
            fold_scores["Huber Per-Stint"].append({"race": held_out_race, "mae": mae_huber, "rmse": rmse_huber})

            # 4. S1 Physics Linear Model
            s1_slopes = fit_s1(X_B_tr, y_tr, fuel_effect=fuel_constant)
            pred_s1 = predict_s1(X_B_te, s1_slopes, fuel_effect=fuel_constant)
            mae_s1 = float(mean_absolute_error(y_te, pred_s1))
            rmse_s1 = float(np.sqrt(mean_squared_error(y_te, pred_s1)))
            fold_scores["S1 Physics Linear"].append({"race": held_out_race, "mae": mae_s1, "rmse": rmse_s1})

            # 5. S2 Scaled Ridge on Variant B
            pipe_s2 = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            pipe_s2.fit(X_B_tr, y_tr)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pred_s2 = pipe_s2.predict(X_B_te)
            assert np.all(np.isfinite(pred_s2)), "S2 Scaled Ridge predictions contain non-finite values"
            mae_s2 = float(mean_absolute_error(y_te, pred_s2))
            rmse_s2 = float(np.sqrt(mean_squared_error(y_te, pred_s2)))
            fold_scores["S2 Scaled Ridge"].append({"race": held_out_race, "mae": mae_s2, "rmse": rmse_s2})

            # 6. Scaled Ridge on Variant A (features including PrevLapRelativePace)
            pipe_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            pipe_ridge.fit(X_A_tr, y_tr)
            assert np.all(np.isfinite(X_A_te.values)), "Features matrix X_A_te contains non-finite values"
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pred_ridge = pipe_ridge.predict(X_A_te)
            assert np.all(np.isfinite(pred_ridge)), "Scaled Ridge predictions contain non-finite values"
            mae_ridge = float(mean_absolute_error(y_te, pred_ridge))
            rmse_ridge = float(np.sqrt(mean_squared_error(y_te, pred_ridge)))
            fold_scores["Scaled Ridge"].append({"race": held_out_race, "mae": mae_ridge, "rmse": rmse_ridge})

            # 7. HistGradientBoosting Variant A (1-step Autoregressive)
            gb_a = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_depth=5,
                l2_regularization=1.0,
                random_state=42,
            )
            gb_a.fit(X_A_tr, y_tr)
            pred_gb_a = gb_a.predict(X_A_te)
            mae_gb_a = float(mean_absolute_error(y_te, pred_gb_a))
            rmse_gb_a = float(np.sqrt(mean_squared_error(y_te, pred_gb_a)))
            fold_scores["HGBR Variant A"].append({"race": held_out_race, "mae": mae_gb_a, "rmse": rmse_gb_a})

            # 8. HistGradientBoosting Variant B (Unconstrained)
            gb_b = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_depth=5,
                l2_regularization=1.0,
                random_state=42,
            )
            gb_b.fit(X_B_tr, y_tr)
            pred_gb_b = gb_b.predict(X_B_te)
            mae_gb_b = float(mean_absolute_error(y_te, pred_gb_b))
            rmse_gb_b = float(np.sqrt(mean_squared_error(y_te, pred_gb_b)))
            fold_scores["HGBR Variant B"].append({"race": held_out_race, "mae": mae_gb_b, "rmse": rmse_gb_b})

            # 9. HistGradientBoosting Variant B (Monotonic Constraints: TyreLife +1, LapsRemaining +1)
            gb_b_mono = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_depth=5,
                l2_regularization=1.0,
                monotonic_cst=[1, 1, 0, 0, 0, 0],
                random_state=42,
            )
            gb_b_mono.fit(X_B_tr, y_tr)
            pred_gb_b_mono = gb_b_mono.predict(X_B_te)
            mae_gb_b_mono = float(mean_absolute_error(y_te, pred_gb_b_mono))
            rmse_gb_b_mono = float(np.sqrt(mean_squared_error(y_te, pred_gb_b_mono)))
            fold_scores["HGBR Variant B Monotonic"].append({"race": held_out_race, "mae": mae_gb_b_mono, "rmse": rmse_gb_b_mono})

            # Multi-origin Horizon Evaluation across stints (TyreLife ~ 3, 8, 15)
            for (_drv_h, _st_h), sgrp in test_df_fold.groupby(["Driver", "Stint"]):
                sgrp = sgrp.sort_values("TyreLife").reset_index(drop=True)
                if len(sgrp) < 10:
                    continue

                for orig_tl in target_origins:
                    tl_diffs = np.abs(sgrp["TyreLife"] - orig_tl)
                    if tl_diffs.min() > 1:
                        continue
                    orig_idx = int(tl_diffs.argmin())
                    orig_row = sgrp.iloc[orig_idx]
                    orig_pace = orig_row["RelativeLapTimeSec"]
                    prior_laps = sgrp.iloc[: orig_idx + 1]

                    for h in horizons:
                        end_idx = orig_idx + h
                        if end_idx >= len(sgrp):
                            continue

                        target_pace = sgrp.iloc[end_idx]["RelativeLapTimeSec"]
                        horizon_slice = sgrp.iloc[orig_idx + 1 : end_idx + 1]
                        actual_laps_y = horizon_slice["RelativeLapTimeSec"].values

                        # Horizon slice features
                        h_indices = horizon_slice.index
                        sub_X_B = X_B_te.loc[X_B_te.index[h_indices]]
                        sub_X_A = X_A_te.loc[X_A_te.index[h_indices]]

                        # Persistence
                        p_pers = np.full(h, orig_pace)
                        horizon_records["Persistence"][h]["abs_err"].append(float(abs(p_pers[-1] - target_pace)))
                        horizon_records["Persistence"][h]["cum_err"].append(float(np.sum(p_pers - actual_laps_y)))

                        # Huber
                        if len(prior_laps) >= 4:
                            try:
                                hr_m = HuberRegressor().fit(prior_laps["TyreLife"].values.reshape(-1, 1), prior_laps["RelativeLapTimeSec"].values)
                                p_hub = hr_m.predict(horizon_slice["TyreLife"].values.reshape(-1, 1))
                            except Exception:
                                p_hub = np.full(h, orig_pace)
                        else:
                            p_hub = np.full(h, orig_pace)
                        horizon_records["Huber (Prior Stint)"][h]["abs_err"].append(float(abs(p_hub[-1] - target_pace)))
                        horizon_records["Huber (Prior Stint)"][h]["cum_err"].append(float(np.sum(p_hub - actual_laps_y)))

                        # S1
                        p_s1 = predict_s1(sub_X_B, s1_slopes, fuel_effect=fuel_constant)
                        horizon_records["S1 Physics Linear"][h]["abs_err"].append(float(abs(p_s1[-1] - target_pace)))
                        horizon_records["S1 Physics Linear"][h]["cum_err"].append(float(np.sum(p_s1 - actual_laps_y)))

                        # S2
                        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                            p_s2 = pipe_s2.predict(sub_X_B)
                        horizon_records["S2 Scaled Ridge"][h]["abs_err"].append(float(abs(p_s2[-1] - target_pace)))
                        horizon_records["S2 Scaled Ridge"][h]["cum_err"].append(float(np.sum(p_s2 - actual_laps_y)))

                        # Variant B
                        p_b = gb_b.predict(sub_X_B)
                        horizon_records["Variant B (Unconstrained)"][h]["abs_err"].append(float(abs(p_b[-1] - target_pace)))
                        horizon_records["Variant B (Unconstrained)"][h]["cum_err"].append(float(np.sum(p_b - actual_laps_y)))

                        # Variant B Monotonic
                        p_bm = gb_b_mono.predict(sub_X_B)
                        horizon_records["Variant B (Monotonic)"][h]["abs_err"].append(float(abs(p_bm[-1] - target_pace)))
                        horizon_records["Variant B (Monotonic)"][h]["cum_err"].append(float(np.sum(p_bm - actual_laps_y)))

                        # Variant A (recursive)
                        curr_pace = orig_pace
                        p_a_seq = []
                        for step_idx in range(h):
                            s_row = sub_X_A.iloc[[step_idx]].copy()
                            s_row["PrevLapRelativePace"] = curr_pace
                            pred_step = float(gb_a.predict(s_row)[0])
                            p_a_seq.append(pred_step)
                            curr_pace = pred_step
                        p_a_seq = np.array(p_a_seq)
                        horizon_records["Variant A (Recursive)"][h]["abs_err"].append(float(abs(p_a_seq[-1] - target_pace)))
                        horizon_records["Variant A (Recursive)"][h]["cum_err"].append(float(np.sum(p_a_seq - actual_laps_y)))

            cv_results["per_race_eval"].append({
                "RaceID": held_out_race,
                "Laps": len(y_te),
                "Naive_MAE": round(mae_naive, 3),
                "Persistence_MAE": round(mae_pers, 3),
                "Huber_MAE": round(mae_huber, 3),
                "S1_MAE": round(mae_s1, 3),
                "S2_MAE": round(mae_s2, 3),
                "Ridge_MAE": round(mae_ridge, 3),
                "HGBR_A_MAE": round(mae_gb_a, 3),
                "HGBR_B_MAE": round(mae_gb_b, 3),
                "HGBR_BMono_MAE": round(mae_gb_b_mono, 3),
                "Naive_RMSE": round(rmse_naive, 3),
                "Persistence_RMSE": round(rmse_pers, 3),
                "Huber_RMSE": round(rmse_huber, 3),
                "S1_RMSE": round(rmse_s1, 3),
                "S2_RMSE": round(rmse_s2, 3),
                "Ridge_RMSE": round(rmse_ridge, 3),
                "HGBR_A_RMSE": round(rmse_gb_a, 3),
                "HGBR_B_RMSE": round(rmse_gb_b, 3),
                "HGBR_BMono_RMSE": round(rmse_gb_b_mono, 3),
                "HGBR_vs_Persistence_Gain": round(mae_pers - mae_gb_a, 3),
                "HGBR_vs_Ridge_Gain": round(mae_ridge - mae_gb_a, 3),
            })

            all_y_true.extend(y_te.values)
            all_hgbr_preds.extend(pred_gb_a)
            all_tyre_life.extend(X_A_te["TyreLife"].values)
            all_compounds.extend(compounds.iloc[test_idx].values)

        # Aggregate LORO scores
        for m_key, key_target in [
            ("Naive Baseline", "naive_baseline"),
            ("Persistence", "persistence_baseline"),
            ("Huber Per-Stint", "huber_per_stint"),
            ("S1 Physics Linear", "s1_physics_linear"),
            ("S2 Scaled Ridge", "s2_scaled_ridge"),
            ("Scaled Ridge", "scaled_ridge"),
            ("HGBR Variant A", "gradient_boosting_A"),
            ("HGBR Variant B", "gradient_boosting_B"),
            ("HGBR Variant B Monotonic", "gradient_boosting_B_monotonic"),
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

        # Multi-origin Horizon summary
        for h in horizons:
            cv_results["horizon_evaluation"][f"H_{h}"] = {
                "horizon_laps": h,
                "models": {
                    m: {
                        "N": len(horizon_records[m][h]["abs_err"]),
                        "mae_at_h": float(np.mean(horizon_records[m][h]["abs_err"])) if horizon_records[m][h]["abs_err"] else 0.0,
                        "signed_cum_err": float(np.mean(horizon_records[m][h]["cum_err"])) if horizon_records[m][h]["cum_err"] else 0.0,
                        "abs_cum_err": float(np.mean(np.abs(horizon_records[m][h]["cum_err"]))) if horizon_records[m][h]["cum_err"] else 0.0,
                    }
                    for m in horizon_records
                },
                "variant_A_recursive_mae": float(np.mean(horizon_records["Variant A (Recursive)"][h]["abs_err"])) if horizon_records["Variant A (Recursive)"][h]["abs_err"] else None,
                "variant_B_static_mae": float(np.mean(horizon_records["Variant B (Unconstrained)"][h]["abs_err"])) if horizon_records["Variant B (Unconstrained)"][h]["abs_err"] else None,
            }

        # ── Residual Analysis Breakdown with Sample Sizes N ──
        res_df = pd.DataFrame({
            "Actual": all_y_true,
            "Pred_HGBR": all_hgbr_preds,
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
        split_idx = int(len(X_A) * 0.8)
        X_A_tr, X_A_te = X_A.iloc[:split_idx], X_A.iloc[split_idx:]
        X_B_tr, X_B_te = X_B.iloc[:split_idx], X_B.iloc[split_idx:]
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

        # Persistence
        pred_pers = X_A_te["PrevLapRelativePace"].values
        cv_results["persistence_baseline"] = {
            "name": "Persistence",
            "mean_mae": float(mean_absolute_error(y_te, pred_pers)),
            "std_mae": 0.0,
            "mean_rmse": float(np.sqrt(mean_squared_error(y_te, pred_pers))),
            "std_rmse": 0.0,
        }

        # Scaled Ridge
        pipe_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        pipe_ridge.fit(X_A_tr, y_tr)
        assert np.all(np.isfinite(X_A_te.values)), "Features matrix X_A_te contains non-finite values"
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pred_ridge = pipe_ridge.predict(X_A_te)
        assert np.all(np.isfinite(pred_ridge)), "Scaled Ridge predictions contain non-finite values"
        cv_results["scaled_ridge"] = {
            "name": "Scaled Ridge",
            "mean_mae": float(mean_absolute_error(y_te, pred_ridge)),
            "std_mae": 0.0,
            "mean_rmse": float(np.sqrt(mean_squared_error(y_te, pred_ridge))),
            "std_rmse": 0.0,
        }

        # HGBR Variant A
        gb_a = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_depth=5,
            l2_regularization=1.0,
            random_state=42,
        )
        gb_a.fit(X_A_tr, y_tr)
        pred_gb_a = gb_a.predict(X_A_te)
        cv_results["gradient_boosting_A"] = {
            "name": "HGBR Variant A",
            "mean_mae": float(mean_absolute_error(y_te, pred_gb_a)),
            "std_mae": 0.0,
            "mean_rmse": float(np.sqrt(mean_squared_error(y_te, pred_gb_a))),
            "std_rmse": 0.0,
        }

        # HGBR Variant B
        gb_b = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_depth=5,
            l2_regularization=1.0,
            random_state=42,
        )
        gb_b.fit(X_B_tr, y_tr)
        pred_gb_b = gb_b.predict(X_B_te)
        cv_results["gradient_boosting_B"] = {
            "name": "HGBR Variant B",
            "mean_mae": float(mean_absolute_error(y_te, pred_gb_b)),
            "std_mae": 0.0,
            "mean_rmse": float(np.sqrt(mean_squared_error(y_te, pred_gb_b))),
            "std_rmse": 0.0,
        }

    # ── Final Production Models Training (on 100% of clean data) ──
    final_model_a = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=150,
        max_depth=5,
        l2_regularization=1.0,
        random_state=42,
    )
    final_model_a.fit(X_A, y)

    final_model_b = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=150,
        max_depth=5,
        l2_regularization=1.0,
        random_state=42,
    )
    final_model_b.fit(X_B, y)

    final_model_b_mono = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=150,
        max_depth=5,
        l2_regularization=1.0,
        monotonic_cst=[1, 1, 0, 0, 0, 0],
        random_state=42,
    )
    final_model_b_mono.fit(X_B, y)

    final_s1_slopes = fit_s1(X_B, y, fuel_effect=fuel_constant)
    final_pipe_s2 = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(X_B, y)

    # Save compact joblib artifact
    artifact_saved = False
    artifact_size_kb = 0.0
    if save_artifact_path:
        os.makedirs(os.path.dirname(save_artifact_path), exist_ok=True)
        artifact_payload = {
            "model": final_model_a,
            "model_variant_a": final_model_a,
            "model_variant_b": final_model_b,
            "model_variant_b_mono": final_model_b_mono,
            "model_s1": {"slopes": final_s1_slopes, "fuel_effect": fuel_constant},
            "model_s2": final_pipe_s2,
            "feature_names_A": feature_names_A,
            "feature_names_B": feature_names_B,
            "feature_names": feature_names_A,
            "metadata": {
                "created_at": datetime.datetime.utcnow().isoformat(),
                "training_samples": len(X_A),
                "races": unique_races.tolist(),
                "cv_results": cv_results,
                "hyperparameters": {
                    "learning_rate": 0.05,
                    "max_iter": 150,
                    "max_depth": 5,
                    "l2_regularization": 1.0,
                    "selection_method": "Fixed physics-informed regularization defaults (never tuned on held-out test races)",
                },
                "target_description": "RelativeLapTimeSec (LapTimeSec minus DriverReferencePaceSec, evaluated strictly on laps >= 7)",
            },
        }
        joblib.dump(artifact_payload, save_artifact_path, compress=3)
        artifact_saved = True
        artifact_size_kb = os.path.getsize(save_artifact_path) / 1024.0
        logger.info(f"Model artifact saved to {save_artifact_path} ({artifact_size_kb:.1f} KB)")

    cv_results["success"] = True
    cv_results["production_model_A"] = final_model_a
    cv_results["production_model_B"] = final_model_b
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
