"""
F1 Race Intelligence — Tyre Intelligence & Degradation Analytics (Phase 1)

Replaces naive OLS regressions with:
1. Fuel-burn pace correction (~0.03 s/lap) isolating pure tyre decay.
2. Robust Huber regression on TyreLife vs fuel-corrected lap times.
3. 95% Confidence Intervals for degradation slopes and predicted pace bands.
4. Two-stage change-point tyre cliff detector (identifying accelerated decay).
5. Driver Consistency Index (robust MAD) and Stint Pace Rankings.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from config import (
    CLIFF_SLOPE_MULTIPLIER,
    DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    MIN_LAPS_FOR_ANALYSIS,
    MIN_LAPS_FOR_CLIFF_DETECTION,
    MIN_LAPS_FOR_REGRESSION,
)
from src.data.cleaning import clean_laps
from src.features.fuel import apply_fuel_correction

logger = logging.getLogger(__name__)


def detect_tyre_cliff(
    x: np.ndarray,
    y: np.ndarray,
    min_cliff_lap: int = MIN_LAPS_FOR_CLIFF_DETECTION,
    slope_multiplier: float = CLIFF_SLOPE_MULTIPLIER,
) -> Dict[str, Any]:
    """
    Two-stage segmented change-point detector for tyre performance cliffs.

    Compares a single linear degradation slope against a two-phase segmented model.
    A tyre cliff is flagged if a candidate knot k satisfies:
      1. Post-cliff degradation slope > slope_multiplier * pre-cliff slope
      2. Sum of squared residual error reduces by at least 15% vs single line
    """
    no_cliff = {
        "cliff_detected": False,
        "cliff_lap": None,
        "pre_cliff_slope": None,
        "post_cliff_slope": None,
        "confidence": 0.0,
    }

    n = len(x)
    if n < min_cliff_lap + 4:
        return no_cliff

    single_fit = np.polyfit(x, y, 1)
    single_err = np.sum((y - np.poly1d(single_fit)(x)) ** 2)
    if single_err <= 1e-6:
        return no_cliff

    best_knot = None
    best_err_drop = 0.0
    best_p1 = single_fit
    best_p2 = single_fit

    # Evaluate candidate knots
    for k in range(min_cliff_lap, n - 3):
        mask1 = x <= k
        mask2 = x > k
        if mask1.sum() < 4 or mask2.sum() < 3:
            continue

        p1 = np.polyfit(x[mask1], y[mask1], 1)
        p2 = np.polyfit(x[mask2], y[mask2], 1)

        two_err = np.sum((y[mask1] - np.poly1d(p1)(x[mask1])) ** 2) + np.sum(
            (y[mask2] - np.poly1d(p2)(x[mask2])) ** 2
        )
        err_drop = (single_err - two_err) / single_err

        # Check for meaningful slope acceleration and error drop
        if p2[0] > slope_multiplier * max(p1[0], 0.01) and err_drop > 0.15:
            if err_drop > best_err_drop:
                best_err_drop = err_drop
                best_knot = int(x[mask1][-1])
                best_p1 = p1
                best_p2 = p2

    if best_knot is not None:
        return {
            "cliff_detected": True,
            "cliff_lap": best_knot,
            "pre_cliff_slope": round(float(best_p1[0]), 4),
            "post_cliff_slope": round(float(best_p2[0]), 4),
            "confidence": round(float(best_err_drop), 3),
        }

    return no_cliff


def compute_stint_degradation(
    stint_laps: pd.DataFrame,
    min_laps: int = MIN_LAPS_FOR_REGRESSION,
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    use_fuel_correction: bool = True,
) -> Dict[str, Any]:
    """
    Robust tyre degradation estimation using Huber regression on fuel-corrected lap times.

    Parameters:
      stint_laps: Laps DataFrame for a single driver stint.
      min_laps: Minimum clean observations required.
      fuel_effect: Per-lap fuel mass effect (s/lap).
      use_fuel_correction: Whether to apply fuel burn-off correction.

    Returns:
      dict containing robust slope, 95% Confidence Intervals, R², base pace,
      tyre cliff diagnostics, and CI curve coordinates.
    """
    empty_result = {
        "slope": 0.0,
        "slope_ci_lower": 0.0,
        "slope_ci_upper": 0.0,
        "intercept": 0.0,
        "r2": 0.0,
        "n_laps": 0,
        "compound": "UNKNOWN",
        "stint": None,
        "driver": None,
        "fuel_effect_applied": 0.0,
        "cliff_detected": False,
        "cliff_lap": None,
        "pre_cliff_slope": None,
        "post_cliff_slope": None,
        "valid": False,
        "fit_x": np.array([]),
        "fit_y": np.array([]),
        "ci_lower": np.array([]),
        "ci_upper": np.array([]),
        "cleaned_data": pd.DataFrame(),
    }

    if stint_laps is None or stint_laps.empty:
        return empty_result

    cleaned = clean_laps(stint_laps, min_laps=min_laps)
    if len(cleaned) < min_laps:
        return empty_result

    # 1. Apply Fuel Correction
    if use_fuel_correction and fuel_effect > 0.0:
        cleaned = apply_fuel_correction(cleaned, fuel_effect_sec_per_lap=fuel_effect)
        target_col = "FuelCorrectedLapTimeSec"
    else:
        target_col = "LapTimeSec"

    # 2. Tyre Life independent variable
    if "TyreLife" not in cleaned.columns:
        if "TireAge" in cleaned.columns:
            cleaned["TyreLife"] = cleaned["TireAge"]
        else:
            cleaned["TyreLife"] = np.arange(1, len(cleaned) + 1)

    X = cleaned[["TyreLife"]].values.astype(float)
    y = cleaned[target_col].values.astype(float)
    n = len(y)

    # 3. Robust Huber Regression
    huber = HuberRegressor(epsilon=1.35, max_iter=200)
    huber.fit(X, y)

    slope = float(huber.coef_[0])
    intercept = float(huber.intercept_)

    # 4. Residual standard error & 95% Confidence Intervals
    preds = huber.predict(X)
    residuals = y - preds
    # Robust standard deviation via Median Absolute Deviation (MAD)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    sigma_robust = max(1.4826 * mad, 0.01)

    x_flat = X.ravel()
    x_mean = float(np.mean(x_flat))
    ss_x = float(np.sum((x_flat - x_mean) ** 2))

    if ss_x > 0:
        se_slope = sigma_robust / np.sqrt(ss_x)
        slope_ci_lower = slope - 1.96 * se_slope
        slope_ci_upper = slope + 1.96 * se_slope

        # Prediction CI band across x
        x_grid = np.linspace(x_flat.min(), x_flat.max(), 50)
        fit_y = huber.predict(x_grid.reshape(-1, 1))
        se_pred = sigma_robust * np.sqrt((1.0 / n) + ((x_grid - x_mean) ** 2) / ss_x)
        ci_lower = fit_y - 1.96 * se_pred
        ci_upper = fit_y + 1.96 * se_pred
    else:
        slope_ci_lower = slope
        slope_ci_upper = slope
        x_grid = x_flat
        fit_y = preds
        ci_lower = preds
        ci_upper = preds

    # 5. R² Score
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r2 = float(max(0.0, 1.0 - (ss_res / ss_tot))) if ss_tot > 0 else 0.0

    # 6. Tyre Cliff Detection
    cliff_res = detect_tyre_cliff(x_flat, y)

    compound = str(cleaned["Compound"].iloc[0]) if "Compound" in cleaned.columns else "UNKNOWN"
    driver = str(cleaned["Driver"].iloc[0]) if "Driver" in cleaned.columns else "UNKNOWN"
    stint_num = int(cleaned["Stint"].iloc[0]) if "Stint" in cleaned.columns and pd.notna(cleaned["Stint"].iloc[0]) else 1

    return {
        "slope": slope,
        "slope_ci_lower": slope_ci_lower,
        "slope_ci_upper": slope_ci_upper,
        "intercept": intercept,
        "r2": r2,
        "n_laps": n,
        "compound": compound,
        "stint": stint_num,
        "driver": driver,
        "fuel_effect_applied": float(fuel_effect) if use_fuel_correction else 0.0,
        "cliff_detected": cliff_res["cliff_detected"],
        "cliff_lap": cliff_res["cliff_lap"],
        "pre_cliff_slope": cliff_res["pre_cliff_slope"],
        "post_cliff_slope": cliff_res["post_cliff_slope"],
        "valid": True,
        "fit_x": x_grid,
        "fit_y": fit_y,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cleaned_data": cleaned,
    }


def analyze_all_stints(
    laps: pd.DataFrame,
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    use_fuel_correction: bool = True,
) -> pd.DataFrame:
    """
    Computes robust Huber tyre degradation regression for every driver and stint.

    Returns:
      pd.DataFrame with robust slopes, confidence intervals, R², and cliff markers.
    """
    if laps is None or laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    results = []
    group_cols = ["Driver", "Stint"] if "Stint" in laps.columns else ["Driver"]

    for _keys, group in laps.groupby(group_cols):
        res = compute_stint_degradation(
            group,
            fuel_effect=fuel_effect,
            use_fuel_correction=use_fuel_correction,
        )
        if res["valid"]:
            cliff_txt = f"Lap {res['cliff_lap']} (+{res['post_cliff_slope']:.3f} s/l)" if res["cliff_detected"] else "None Detected"
            results.append({
                "Driver": res["driver"],
                "Stint": res["stint"],
                "Compound": res["compound"],
                "DegradationRate": round(res["slope"], 4),
                "DegradationFormatted": f"{res['slope']:+.3f} s/lap",
                "SlopeCI": f"[{res['slope_ci_lower']:+.3f}, {res['slope_ci_upper']:+.3f}]",
                "R2": round(res["r2"], 3),
                "LapsAnalyzed": res["n_laps"],
                "BasePace": round(res["intercept"], 3),
                "TyreCliff": cliff_txt,
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(by=["Driver", "Stint"])


def get_compound_summary(
    laps: pd.DataFrame,
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> pd.DataFrame:
    """
    Aggregates robust compound performance across drivers:
    Median degradation rate, base pace, confidence intervals, and cliff frequency.
    """
    if laps is None or laps.empty or "Compound" not in laps.columns:
        return pd.DataFrame()

    cleaned = clean_laps(laps)
    if cleaned.empty:
        return pd.DataFrame()

    rows = []
    for compound, group in cleaned.groupby("Compound"):
        if pd.isna(compound) or str(compound).strip() == "":
            continue

        if len(group) < MIN_LAPS_FOR_ANALYSIS:
            continue

        deg_res = compute_stint_degradation(group, min_laps=5, fuel_effect=fuel_effect)
        if deg_res["valid"]:
            cliff_count = 0
            total_stints = 0
            if "Stint" in group.columns:
                for _, stint_grp in group.groupby(["Driver", "Stint"]):
                    s_res = compute_stint_degradation(stint_grp, min_laps=5, fuel_effect=fuel_effect)
                    if s_res["valid"]:
                        total_stints += 1
                        if s_res["cliff_detected"]:
                            cliff_count += 1

            cliff_freq_txt = f"{cliff_count}/{total_stints} stints" if total_stints > 0 else "N/A"

            rows.append({
                "Compound": str(compound).upper(),
                "TotalLaps": len(group),
                "MedianBasePace": round(deg_res["intercept"], 3),
                "DegradationRate": round(deg_res["slope"], 4),
                "DegradationFormatted": f"{deg_res['slope']:+.3f} s/lap",
                "Slope95CI": f"[{deg_res['slope_ci_lower']:+.3f}, {deg_res['slope_ci_upper']:+.3f}]",
                "CliffFrequency": cliff_freq_txt,
                "DriversUsed": group["Driver"].nunique() if "Driver" in group.columns else 1,
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(by="MedianBasePace")


def compute_driver_consistency_index(
    laps: pd.DataFrame,
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> pd.DataFrame:
    """
    Computes Driver Consistency Index & Race Pace Rankings.

    Uses robust Median Absolute Deviation (MAD * 1.4826) on fuel-corrected clean laps
    to eliminate traffic and pit stop artifacts.
    """
    if laps is None or laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    cleaned = clean_laps(laps)
    if cleaned.empty:
        return pd.DataFrame()

    corrected = apply_fuel_correction(cleaned, fuel_effect_sec_per_lap=fuel_effect)

    records = []
    for driver, grp in corrected.groupby("Driver"):
        if len(grp) < MIN_LAPS_FOR_ANALYSIS:
            continue

        times = grp["FuelCorrectedLapTimeSec"].dropna()
        if len(times) < MIN_LAPS_FOR_ANALYSIS:
            continue

        median_pace = float(times.median())
        mad_dispersion = float(np.median(np.abs(times - median_pace)) * 1.4826)
        std_dispersion = float(times.std())
        q25 = float(times.quantile(0.25))
        q75 = float(times.quantile(0.75))
        iqr = q75 - q25

        records.append({
            "Driver": str(driver),
            "CleanLaps": len(times),
            "MedianRacePaceSec": round(median_pace, 3),
            "ConsistencyIndex_MAD": round(mad_dispersion, 3),  # Lower is more consistent
            "StandardDeviation": round(std_dispersion, 3),
            "IQR": round(iqr, 3),
            "BestLapSec": round(float(times.min()), 3),
        })

    if not records:
        return pd.DataFrame()

    df_res = pd.DataFrame(records)
    # Add rankings
    df_res["PaceRank"] = df_res["MedianRacePaceSec"].rank().astype(int)
    df_res["ConsistencyRank"] = df_res["ConsistencyIndex_MAD"].rank().astype(int)

    return df_res.sort_values(by="PaceRank").reset_index(drop=True)
