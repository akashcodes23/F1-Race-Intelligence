"""
F1 Race Intelligence — Tyre Intelligence & Degradation Analytics (Phase 1.5)

Analytical Engine:
1. Fuel-burn pace correction isolating pure tyre degradation.
2. Robust Huber regression on TyreLife vs fuel-corrected lap times.
3. Both i.i.d. and HAC (Heteroskedasticity and Autocorrelation Consistent) Newey-West
   Confidence Intervals to account for lap-to-lap sequential autocorrelation.
4. Two-stage change-point tyre cliff detector with minimum tyre age constraints
   (default >= 10 laps) and minimum sample sizes on both sides.
5. Exclusion of wet compounds (INTERMEDIATE, WET) reported as "not comparable".
6. Overlapping-CI warnings for compound comparisons without unjustified compound superiority claims.
"""

import logging
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from config import (
    CLIFF_MIN_LAPS_POST,
    CLIFF_MIN_LAPS_PRE,
    CLIFF_MIN_TYRE_AGE,
    CLIFF_SLOPE_MULTIPLIER,
    DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    MIN_LAPS_FOR_ANALYSIS,
    MIN_LAPS_FOR_REGRESSION,
)
from src.data.cleaning import clean_laps
from src.features.fuel import apply_fuel_correction

logger = logging.getLogger(__name__)

WET_COMPOUNDS: Set[str] = {"INTERMEDIATE", "WET"}


def _fit_line(lx: np.ndarray, ly: np.ndarray) -> Optional[np.ndarray]:
    """
    Closed-form ordinary least squares linear regression returning np.array([slope, intercept]).
    Returns None if fewer than 2 points or if x-variance is zero.
    """
    if lx is None or ly is None or len(lx) < 2 or len(ly) < 2:
        return None
    lx = np.asarray(lx, dtype=float)
    ly = np.asarray(ly, dtype=float)
    xm, ym = float(np.mean(lx)), float(np.mean(ly))
    ss_x = float(np.sum((lx - xm) ** 2))
    if ss_x <= 1e-12:
        return None
    slp = float(np.sum((lx - xm) * (ly - ym)) / ss_x)
    icept = ym - slp * xm
    return np.array([slp, icept])


def detect_tyre_cliff(
    x: np.ndarray,
    y: np.ndarray,
    min_cliff_lap: int = CLIFF_MIN_TYRE_AGE,
    slope_multiplier: float = CLIFF_SLOPE_MULTIPLIER,
    min_laps_pre: int = CLIFF_MIN_LAPS_PRE,
    min_laps_post: int = CLIFF_MIN_LAPS_POST,
    min_slope_delta: float = 0.04,
    min_err_drop: float = 0.15,
    f_stat_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Two-stage segmented change-point detector for tyre performance cliffs.

    Phase 1.5 Constraints:
      - Requires tyre age >= min_cliff_lap before a candidate knot is allowed (default 10).
      - Requires at least min_laps_pre laps before knot (default 5).
      - Requires at least min_laps_post laps after knot (default 4).
      - Post-cliff slope must exceed slope_multiplier * pre_cliff_slope AND (post - pre) >= min_slope_delta.
      - Sum of squared error must drop by >= min_err_drop (15%).
      - Optional formal F-test vs single linear fit: F = ((SS_single - SS_two)/2) / (SS_two/(n-4)) >= f_stat_threshold.
    """
    no_cliff = {
        "cliff_detected": False,
        "cliff_lap": None,
        "pre_cliff_slope": None,
        "post_cliff_slope": None,
        "confidence": 0.0,
    }

    if x is None or y is None:
        return no_cliff

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    # Need enough data to evaluate knot satisfying both sides
    if n < min_laps_pre + min_laps_post:
        return no_cliff

    single_fit = _fit_line(x, y)
    if single_fit is None:
        return no_cliff
    single_err = float(np.sum((y - (single_fit[0] * x + single_fit[1])) ** 2))
    if single_err <= 1e-6:
        return no_cliff

    best_knot = None
    best_err_drop = 0.0
    best_p1 = single_fit
    best_p2 = single_fit

    # Evaluate candidate knots satisfying minimum tyre age and minimum lap counts on both sides
    for k in np.unique(x):
        if k < min_cliff_lap:
            continue

        mask1 = x <= k
        mask2 = x > k
        if mask1.sum() < min_laps_pre or mask2.sum() < min_laps_post:
            continue

        p1 = _fit_line(x[mask1], y[mask1])
        p2 = _fit_line(x[mask2], y[mask2])
        if p1 is None or p2 is None:
            continue

        two_err = float(
            np.sum((y[mask1] - (p1[0] * x[mask1] + p1[1])) ** 2)
            + np.sum((y[mask2] - (p2[0] * x[mask2] + p2[1])) ** 2)
        )
        err_drop = (single_err - two_err) / single_err

        # Check for meaningful slope acceleration and error drop
        slope_accel = p2[0] > (slope_multiplier * max(p1[0], 0.01))
        meaningful_delta = (p2[0] - p1[0]) >= min_slope_delta

        if f_stat_threshold is not None:
            df2 = n - 4
            if df2 <= 0 or two_err <= 1e-12:
                continue
            f_stat = ((single_err - two_err) / 2.0) / (two_err / df2)
            if f_stat < f_stat_threshold:
                continue

        if slope_accel and meaningful_delta and err_drop > min_err_drop:
            if err_drop > best_err_drop:
                best_err_drop = err_drop
                best_knot = int(k)
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

    Excludes wet compounds (INTERMEDIATE, WET) and reports them as 'not comparable'.
    Computes both i.i.d. and HAC (Newey-West) standard errors to account for autocorrelation.

    Returns:
      dict containing robust slope, i.i.d. & HAC Confidence Intervals, R², base pace,
      tyre cliff diagnostics, and CI curve coordinates.
    """
    empty_result: Dict[str, Any] = {
        "slope": 0.0,
        "slope_ci_lower": 0.0,
        "slope_ci_upper": 0.0,
        "slope_ci_lower_iid": 0.0,
        "slope_ci_upper_iid": 0.0,
        "slope_ci_lower_hac": 0.0,
        "slope_ci_upper_hac": 0.0,
        "se_slope_iid": 0.0,
        "se_slope_hac": 0.0,
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
        "is_comparable": False,
        "status": "insufficient_laps",
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

    compound = str(cleaned["Compound"].iloc[0]).upper() if "Compound" in cleaned.columns else "UNKNOWN"
    driver = str(cleaned["Driver"].iloc[0]) if "Driver" in cleaned.columns else "UNKNOWN"
    stint_num = int(cleaned["Stint"].iloc[0]) if "Stint" in cleaned.columns and pd.notna(cleaned["Stint"].iloc[0]) else 1

    # Exclude wet compounds (INTERMEDIATE, WET) from slope estimation
    if compound in WET_COMPOUNDS:
        return {
            "slope": None,
            "slope_ci_lower": None,
            "slope_ci_upper": None,
            "slope_ci_lower_iid": None,
            "slope_ci_upper_iid": None,
            "slope_ci_lower_hac": None,
            "slope_ci_upper_hac": None,
            "se_slope_iid": None,
            "se_slope_hac": None,
            "intercept": round(float(cleaned["LapTimeSec"].median()), 3) if "LapTimeSec" in cleaned.columns else None,
            "r2": None,
            "n_laps": len(cleaned),
            "compound": compound,
            "stint": stint_num,
            "driver": driver,
            "fuel_effect_applied": 0.0,
            "cliff_detected": False,
            "cliff_lap": None,
            "pre_cliff_slope": None,
            "post_cliff_slope": None,
            "valid": True,
            "is_comparable": False,
            "status": "not comparable (wet compound)",
            "degradation_formatted": "Not comparable (wet compound)",
            "fit_x": np.array([]),
            "fit_y": np.array([]),
            "ci_lower": np.array([]),
            "ci_upper": np.array([]),
            "cleaned_data": cleaned,
        }

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

    # 4. Residuals & Confidence Intervals
    preds = huber.predict(X)
    residuals = y - preds

    # Robust i.i.d. standard error via MAD
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    sigma_robust = max(1.4826 * mad, 0.01)

    x_flat = X.ravel()
    x_mean = float(np.mean(x_flat))
    z = x_flat - x_mean
    ss_x = float(np.sum(z ** 2))

    if ss_x > 0:
        # i.i.d. Standard Error
        se_slope_iid = float(sigma_robust / np.sqrt(ss_x))
        slope_ci_lower_iid = slope - 1.96 * se_slope_iid
        slope_ci_upper_iid = slope + 1.96 * se_slope_iid

        # HAC (Newey-West) Standard Error with lag L=1
        gamma_0 = float(np.sum((z ** 2) * (residuals ** 2)))
        # 1-lag autocovariance
        if n > 2:
            gamma_1 = float(np.sum(z[1:] * z[:-1] * residuals[1:] * residuals[:-1]))
            # Bartlett kernel weight: w_1 = 1 - 1/2 = 0.5
            v_hac = gamma_0 + 2.0 * 0.5 * gamma_1
            # Small sample df correction
            df_corr = float(n / max(n - 2, 1))
            var_hac = (v_hac / (ss_x ** 2)) * df_corr
            se_slope_hac = float(np.sqrt(max(var_hac, (se_slope_iid ** 2) * 0.5)))
        else:
            se_slope_hac = se_slope_iid

        slope_ci_lower_hac = slope - 1.96 * se_slope_hac
        slope_ci_upper_hac = slope + 1.96 * se_slope_hac

        # Default prediction CI band across x (using HAC-adjusted standard error)
        x_grid = np.linspace(x_flat.min(), x_flat.max(), 50)
        fit_y = huber.predict(x_grid.reshape(-1, 1))
        se_pred = max(sigma_robust, se_slope_hac * np.sqrt(ss_x)) * np.sqrt((1.0 / n) + ((x_grid - x_mean) ** 2) / ss_x)
        ci_lower = fit_y - 1.96 * se_pred
        ci_upper = fit_y + 1.96 * se_pred
    else:
        se_slope_iid = 0.0
        se_slope_hac = 0.0
        slope_ci_lower_iid = slope
        slope_ci_upper_iid = slope
        slope_ci_lower_hac = slope
        slope_ci_upper_hac = slope
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

    return {
        "slope": slope,
        "slope_ci_lower": slope_ci_lower_hac,  # Default to HAC
        "slope_ci_upper": slope_ci_upper_hac,
        "slope_ci_lower_iid": slope_ci_lower_iid,
        "slope_ci_upper_iid": slope_ci_upper_iid,
        "slope_ci_lower_hac": slope_ci_lower_hac,
        "slope_ci_upper_hac": slope_ci_upper_hac,
        "se_slope_iid": se_slope_iid,
        "se_slope_hac": se_slope_hac,
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
        "is_comparable": True,
        "status": "valid",
        "degradation_formatted": f"{slope:+.3f} s/lap",
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

    Reports i.i.d. and HAC Confidence Intervals side by side.
    Excludes wet compounds, labeling them as 'Not comparable (wet)'.
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
        if not res["valid"]:
            continue

        if not res.get("is_comparable", True) or res["slope"] is None:
            results.append({
                "Driver": res["driver"],
                "Stint": res["stint"],
                "Compound": res["compound"],
                "DegradationRate": None,
                "DegradationFormatted": "Not comparable (wet)",
                "SlopeCI": "N/A",
                "SlopeCI_HAC": "N/A",
                "R2": "N/A",
                "LapsAnalyzed": res["n_laps"],
                "BasePace": round(res["intercept"], 3) if res["intercept"] is not None else "N/A",
                "TyreCliff": "N/A",
                "IsComparable": False,
            })
        else:
            cliff_txt = f"Lap {res['cliff_lap']} (+{res['post_cliff_slope']:.3f} s/l)" if res["cliff_detected"] else "None Detected"
            results.append({
                "Driver": res["driver"],
                "Stint": res["stint"],
                "Compound": res["compound"],
                "DegradationRate": round(res["slope"], 4),
                "DegradationFormatted": f"{res['slope']:+.3f} s/lap",
                "SlopeCI": f"[{res['slope_ci_lower_iid']:+.3f}, {res['slope_ci_upper_iid']:+.3f}]",
                "SlopeCI_HAC": f"[{res['slope_ci_lower_hac']:+.3f}, {res['slope_ci_upper_hac']:+.3f}]",
                "R2": round(res["r2"], 3),
                "LapsAnalyzed": res["n_laps"],
                "BasePace": round(res["intercept"], 3),
                "TyreCliff": cliff_txt,
                "IsComparable": True,
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(by=["Driver", "Stint"])


def get_compound_summary(
    laps: pd.DataFrame,
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> pd.DataFrame:
    """
    Aggregates robust compound performance across drivers.

    Displays both i.i.d. and HAC intervals, excludes wet compounds as 'not comparable',
    and annotates overlapping-CI warnings without unwarranted compound superiority claims.
    """
    if laps is None or laps.empty or "Compound" not in laps.columns:
        return pd.DataFrame()

    cleaned = clean_laps(laps)
    if cleaned.empty:
        return pd.DataFrame()

    rows = []
    dry_intervals = []

    for compound, group in cleaned.groupby("Compound"):
        comp_str = str(compound).upper().strip()
        if not comp_str:
            continue

        if len(group) < MIN_LAPS_FOR_ANALYSIS:
            continue

        deg_res = compute_stint_degradation(group, min_laps=5, fuel_effect=fuel_effect)
        if not deg_res["valid"]:
            continue

        if comp_str in WET_COMPOUNDS or not deg_res.get("is_comparable", True):
            rows.append({
                "Compound": comp_str,
                "TotalLaps": len(group),
                "MedianBasePace": round(float(group["LapTimeSec"].median()), 3),
                "DegradationRate": None,
                "DegradationFormatted": "Not comparable (wet compound)",
                "Slope95CI": "N/A",
                "Slope95CI_HAC": "N/A",
                "CliffFrequency": "N/A",
                "DriversUsed": group["Driver"].nunique() if "Driver" in group.columns else 1,
                "CIWarning": "Wet compound — pace governed by track drying/water dispersal rather than rubber degradation.",
            })
        else:
            cliff_count = 0
            total_stints = 0
            if "Stint" in group.columns:
                for _, stint_grp in group.groupby(["Driver", "Stint"]):
                    s_res = compute_stint_degradation(stint_grp, min_laps=5, fuel_effect=fuel_effect)
                    if s_res["valid"] and s_res.get("is_comparable", True):
                        total_stints += 1
                        if s_res["cliff_detected"]:
                            cliff_count += 1

            cliff_freq_txt = f"{cliff_count}/{total_stints} stints" if total_stints > 0 else "N/A"

            row_data = {
                "Compound": comp_str,
                "TotalLaps": len(group),
                "MedianBasePace": round(deg_res["intercept"], 3),
                "DegradationRate": round(deg_res["slope"], 4),
                "DegradationFormatted": f"{deg_res['slope']:+.3f} s/lap",
                "Slope95CI": f"[{deg_res['slope_ci_lower_iid']:+.3f}, {deg_res['slope_ci_upper_iid']:+.3f}]",
                "Slope95CI_HAC": f"[{deg_res['slope_ci_lower_hac']:+.3f}, {deg_res['slope_ci_upper_hac']:+.3f}]",
                "CliffFrequency": cliff_freq_txt,
                "DriversUsed": group["Driver"].nunique() if "Driver" in group.columns else 1,
                "CIWarning": "Single compound",
                "_ci_low": deg_res["slope_ci_lower_hac"],
                "_ci_high": deg_res["slope_ci_upper_hac"],
            }
            dry_intervals.append(row_data)
            rows.append(row_data)

    if not rows:
        return pd.DataFrame()

    # Check for overlapping 95% CIs among dry compounds
    if len(dry_intervals) >= 2:
        for i in range(len(dry_intervals)):
            overlaps = []
            for j in range(len(dry_intervals)):
                if i != j:
                    c1 = dry_intervals[i]
                    c2 = dry_intervals[j]
                    # Check overlap: max(low1, low2) <= min(high1, high2)
                    if max(c1["_ci_low"], c2["_ci_low"]) <= min(c1["_ci_high"], c2["_ci_high"]):
                        overlaps.append(c2["Compound"])
            if overlaps:
                dry_intervals[i]["CIWarning"] = (
                    f"Overlapping 95% CI with {', '.join(overlaps)} (sample size: {dry_intervals[i]['TotalLaps']} laps). "
                    "Observed differences are statistically indistinguishable under race sample size."
                )
            else:
                dry_intervals[i]["CIWarning"] = f"Distinct 95% CI (sample size: {dry_intervals[i]['TotalLaps']} laps)."

    # Remove internal temporary keys
    for r in rows:
        r.pop("_ci_low", None)
        r.pop("_ci_high", None)

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
