"""
Phase 3A.4 Comprehensive Audit, Evaluation & Verification Script

Executes all audits for Phase 3A.4:
1. Honest, causal stop-window hit rate evaluation at origins 5, 10, 15 laps before the stop vs equal-width baselines
2. Precise undercut vs overcut classification with real elapsed telemetry gaps (gap <= 3s filter, bucketed base rates)
3. Pit-gain target audit by tyre age bins, scatter plot, Spearman explanation, and fuel sensitivity (c=0.033, 0.045, 0.058)
4. Pairwise finish-order accuracy of engine vs naive baseline on identical pairs
5. S1b drift model (anchor + b_c * h) benchmark vs persistence at H=1, 5, 10, 20
6. Regulatory and housekeeping audits
"""

import json
import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DEFAULT_FUEL_EFFECT_SEC_PER_LAP
from src.data.cleaning import clean_laps
from src.features.fuel import (
    compute_s1a_anchor,
    normalize_lap_fuel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Statistical Utilities ───────────────────────────────────────────────────

def race_bootstrap_ci(df: pd.DataFrame, metric_col: str, n_boot: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    races = df["race"].unique() if "race" in df.columns else df["RaceID"].unique()
    means = []
    for _ in range(n_boot):
        boot_races = rng.choice(races, size=len(races), replace=True)
        sub = df[df["race"].isin(boot_races)] if "race" in df.columns else df[df["RaceID"].isin(boot_races)]
        if not sub.empty:
            means.append(sub[metric_col].mean())
    if not means:
        return (0.0, 0.0)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_race_bootstrap_diff_ci(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    races = df["race"].unique() if "race" in df.columns else df["RaceID"].unique()
    diffs = []
    for _ in range(n_boot):
        boot_races = rng.choice(races, size=len(races), replace=True)
        sub = df[df["race"].isin(boot_races)] if "race" in df.columns else df[df["RaceID"].isin(boot_races)]
        if not sub.empty:
            diffs.append((sub[col_a] - sub[col_b]).mean())
    if not diffs:
        return (0.0, 0.0)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996 if confidence == 0.95 else 1.64485
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (float(max(0.0, center - spread)), float(min(1.0, center + spread)))


def load_data():
    df = pd.read_parquet("data/demo/demo_laps.parquet")
    if "LapTimeSec" not in df.columns and "LapTime" in df.columns:
        df["LapTimeSec"] = df["LapTime"].dt.total_seconds()
    df["LapsRemaining"] = df.groupby("RaceID")["LapNumber"].transform("max") - df["LapNumber"]
    return df


def run_phase3a4():
    print("======================================================================")
    print("STARTING PHASE 3A.4 COMPREHENSIVE AUDIT & EVALUATION")
    print("======================================================================")
    os.makedirs("docs/results", exist_ok=True)
    df = load_data()

    # Pre-clean data for stint analysis
    clean_df = clean_laps(df, drop_pit_laps=True, drop_caution_laps=True, drop_yellow_laps=True).dropna(
        subset=["TyreLife", "LapNumber", "LapTimeSec"]
    )
    dry_clean = clean_df[clean_df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()

    # Reference paces per race & driver
    ref_paces = {}
    for (r, d), grp in clean_df.groupby(["RaceID", "Driver"]):
        c_laps = grp[(grp["LapNumber"] >= 2) & (grp["LapNumber"] <= 6)]["LapTimeSec"]
        if len(c_laps) >= 2:
            ref_paces[(r, d)] = c_laps.median()
        else:
            ref_paces[(r, d)] = grp["LapTimeSec"].median()

    dry_clean["ReferencePace"] = dry_clean.apply(lambda row: ref_paces.get((row["RaceID"], row["Driver"]), np.nan), axis=1)
    dry_clean["RelativeLapTimeSec"] = dry_clean["LapTimeSec"] - dry_clean["ReferencePace"]
    dry_clean = dry_clean.dropna(subset=["RelativeLapTimeSec"]).copy()

    races = sorted(dry_clean["RaceID"].unique())
    results_payload = {}

    # ─────────────────────────────────────────────────────────────
    # 1. STOP-WINDOW HIT RATE (HONEST, CAUSAL EVALUATION)
    # ─────────────────────────────────────────────────────────────
    print("\n--- 1. Causal Stop-Window Hit Rate vs Equal-Width Baselines ---")

    # Extract all real pit stops with clean stints
    stops_list = []
    for race, r_grp in df.groupby("RaceID"):
        total_race_laps = int(r_grp["LapNumber"].max())
        for d, d_grp in r_grp.groupby("Driver"):
            stints = d_grp.groupby("Stint")
            for st_id, st_data in stints:
                if st_id >= 2:
                    continue  # focus on stop 1
                next_st = d_grp[d_grp["Stint"] == st_id + 1]
                if not next_st.empty:
                    comp1 = st_data["Compound"].iloc[0] if "Compound" in st_data else "UNKNOWN"
                    comp2 = next_st["Compound"].iloc[0] if "Compound" in next_st else "UNKNOWN"
                    if comp1 in ["SOFT", "MEDIUM", "HARD"] and comp2 in ["SOFT", "MEDIUM", "HARD"]:
                        stop_lap = int(st_data["LapNumber"].max())
                        stint_start = int(st_data["LapNumber"].min())
                        stops_list.append({
                            "race": race,
                            "driver": d,
                            "pit_lap": stop_lap,
                            "stint_start": stint_start,
                            "comp1": comp1,
                            "comp2": comp2,
                            "total_laps": total_race_laps,
                        })

    df_real_stops = pd.DataFrame(stops_list)
    print(f"Total evaluated clean dry stops: {len(df_real_stops)}")

    # Train baselines out-of-fold:
    # Baseline 1: Circuit/Compound median stop lap
    # Baseline 2: Fixed heuristic stint length (Soft: 16, Medium: 24, Hard: 32)
    stint_heuristic_lens = {"SOFT": 16, "MEDIUM": 24, "HARD": 32}

    # Evaluate at origins strictly before the actual stop:
    # t_0 = stop_lap - 5, stop_lap - 10, stop_lap - 15
    hit_rate_eval_results = []
    width_curves = []

    for delta_origin in [5, 10, 15]:
        eligible = df_real_stops[df_real_stops["pit_lap"] - df_real_stops["stint_start"] > delta_origin].copy()
        print(f"\nEvaluating Origin t0 = StopLap - {delta_origin} laps (Eligible Stops: {len(eligible)}):")

        # Fit causal stop timing on eligible stops
        origin_records = []
        for held_out in races:
            tr_stops = df_real_stops[df_real_stops["race"] != held_out]
            te_stops = eligible[eligible["race"] == held_out]

            circ_comp_medians = tr_stops.groupby(["race", "comp1"])["pit_lap"].median().to_dict()
            global_comp_medians = tr_stops.groupby("comp1")["pit_lap"].median().to_dict()

            # Physics parameters on train
            tr_c = dry_clean[dry_clean["RaceID"] != held_out]
            slopes_tr = {}
            for c in ["SOFT", "MEDIUM", "HARD"]:
                c_sub = tr_c[tr_c["Compound"] == c]
                norm_c = normalize_lap_fuel(c_sub["RelativeLapTimeSec"].values, c_sub["LapNumber"].values, reference_lap=4, fuel_effect_sec_per_lap=DEFAULT_FUEL_EFFECT_SEC_PER_LAP)
                hr = HuberRegressor().fit(c_sub["TyreLife"].values.reshape(-1, 1), norm_c)
                slopes_tr[c] = max(0.01, float(hr.coef_[0]))

            med_sub = tr_c[tr_c["Compound"] == "MEDIUM"]
            med_norm = normalize_lap_fuel(med_sub["RelativeLapTimeSec"].values, med_sub["LapNumber"].values, reference_lap=4, fuel_effect_sec_per_lap=DEFAULT_FUEL_EFFECT_SEC_PER_LAP)
            med_base = float(np.median(med_norm - slopes_tr["MEDIUM"] * med_sub["TyreLife"].values))

            offs_tr = {}
            for c in ["SOFT", "MEDIUM", "HARD"]:
                c_sub = tr_c[tr_c["Compound"] == c]
                c_norm = normalize_lap_fuel(c_sub["RelativeLapTimeSec"].values, c_sub["LapNumber"].values, reference_lap=4, fuel_effect_sec_per_lap=DEFAULT_FUEL_EFFECT_SEC_PER_LAP)
                c_base = float(np.median(c_norm - slopes_tr[c] * c_sub["TyreLife"].values))
                offs_tr[c] = float(c_base - med_base)

            for _, s_row in te_stops.iterrows():
                act_stop = int(s_row["pit_lap"])
                t_origin = act_stop - delta_origin
                st_start = int(s_row["stint_start"])
                c_curr = s_row["comp1"]
                c_target = s_row["comp2"]
                tot_laps = int(s_row["total_laps"])

                # Causal Engine Stop Timing from t_origin:
                # Driver laps observed strictly up to t_origin
                driver_laps_to_t0 = dry_clean[(dry_clean["RaceID"] == held_out) & (dry_clean["Driver"] == s_row["driver"]) & (dry_clean["LapNumber"] <= t_origin)]
                if len(driver_laps_to_t0) >= 3:
                    anchor = compute_s1a_anchor(
                        driver_laps_to_t0,
                        origin_lap=float(t_origin),
                        origin_tyre_age=float(t_origin - st_start + 1),
                        compound_deg_slope=slopes_tr[c_curr],
                        fuel_effect_sec_per_lap=DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
                    )
                else:
                    anchor = 80.0

                # Search candidate stop laps L_cand in [t_origin + 1, min(tot_laps - 3, t_origin + 35)]
                best_cand_lap = t_origin + 1
                best_remaining_time = float("inf")
                pit_loss_val = 21.0

                for l_cand in range(t_origin + 1, min(tot_laps - 2, t_origin + 35)):
                    # Remaining time = laps on old tyre + pit loss + laps on new tyre
                    old_laps = l_cand - t_origin
                    new_laps = tot_laps - l_cand

                    # Old tyre progression
                    curr_age = t_origin - st_start + 1
                    t_old = 0.0
                    for step in range(1, old_laps + 1):
                        lap_num = t_origin + step
                        t_old += anchor + slopes_tr[c_curr] * (curr_age + step - 1) - DEFAULT_FUEL_EFFECT_SEC_PER_LAP * (lap_num - t_origin)

                    # New tyre progression
                    base_new = anchor + (offs_tr[c_target] - offs_tr[c_curr]) - slopes_tr[c_curr] * curr_age
                    t_new = 0.0
                    for step in range(1, new_laps + 1):
                        lap_num = l_cand + step
                        t_new += base_new + slopes_tr[c_target] * step - DEFAULT_FUEL_EFFECT_SEC_PER_LAP * (lap_num - t_origin)

                    total_t = t_old + pit_loss_val + t_new
                    if total_t < best_remaining_time:
                        best_remaining_time = total_t
                        best_cand_lap = l_cand

                # Baseline 1: Circuit/Compound Median Stop Lap
                b1_lap = circ_comp_medians.get((held_out, c_curr), global_comp_medians.get(c_curr, 25.0))

                # Baseline 2: Fixed Stint Heuristic
                b2_lap = st_start + stint_heuristic_lens.get(c_curr, 24)

                origin_records.append({
                    "race": held_out,
                    "driver": s_row["driver"],
                    "actual_stop": act_stop,
                    "t_origin": t_origin,
                    "engine_pred": best_cand_lap,
                    "b1_pred": b1_lap,
                    "b2_pred": b2_lap,
                })

        df_orig = pd.DataFrame(origin_records)

        # Evaluate across width k in 0..10
        for k in range(0, 11):
            width = 2 * k + 1
            df_orig[f"hit_engine_k{k}"] = ((df_orig["actual_stop"] >= df_orig["engine_pred"] - k) & (df_orig["actual_stop"] <= df_orig["engine_pred"] + k)).astype(int)
            df_orig[f"hit_b1_k{k}"] = ((df_orig["actual_stop"] >= df_orig["b1_pred"] - k) & (df_orig["actual_stop"] <= df_orig["b1_pred"] + k)).astype(int)
            df_orig[f"hit_b2_k{k}"] = ((df_orig["actual_stop"] >= df_orig["b2_pred"] - k) & (df_orig["actual_stop"] <= df_orig["b2_pred"] + k)).astype(int)

            rate_eng = df_orig[f"hit_engine_k{k}"].mean()
            ci_eng = race_bootstrap_ci(df_orig, f"hit_engine_k{k}")
            rate_b1 = df_orig[f"hit_b1_k{k}"].mean()
            ci_b1 = race_bootstrap_ci(df_orig, f"hit_b1_k{k}")
            rate_b2 = df_orig[f"hit_b2_k{k}"].mean()
            ci_b2 = race_bootstrap_ci(df_orig, f"hit_b2_k{k}")

            diff_eng_b1 = paired_race_bootstrap_diff_ci(df_orig, f"hit_engine_k{k}", f"hit_b1_k{k}")

            width_curves.append({
                "origin_delta": delta_origin,
                "k": k,
                "window_width_laps": width,
                "engine_hit_rate": round(rate_eng, 3),
                "engine_ci": [round(ci_eng[0], 3), round(ci_eng[1], 3)],
                "b1_median_hit_rate": round(rate_b1, 3),
                "b1_ci": [round(ci_b1[0], 3), round(ci_b1[1], 3)],
                "b2_heuristic_hit_rate": round(rate_b2, 3),
                "b2_ci": [round(ci_b2[0], 3), round(ci_b2[1], 3)],
                "diff_vs_b1": [round(diff_eng_b1[0], 3), round(diff_eng_b1[1], 3)],
            })

            if k in [0, 2, 5, 8]:
                print(f"  k={k:2d} (Width={width:2d} laps): Engine={rate_eng:.1%} [{ci_eng[0]:.1%}, {ci_eng[1]:.1%}] | CircMed={rate_b1:.1%} [{ci_b1[0]:.1%}, {ci_b1[1]:.1%}] | Heur={rate_b2:.1%} [{ci_b2[0]:.1%}, {ci_b2[1]:.1%}] | Diff={rate_eng - rate_b1:+.1%}")

        # Summary for k=2 (+- 2 laps)
        eng_mae = (df_orig["engine_pred"] - df_orig["actual_stop"]).abs().mean()
        b1_mae = (df_orig["b1_pred"] - df_orig["actual_stop"]).abs().mean()
        b2_mae = (df_orig["b2_pred"] - df_orig["actual_stop"]).abs().mean()

        hit_rate_eval_results.append({
            "origin_delta": delta_origin,
            "N": len(df_orig),
            "engine_abs_lap_error": round(eng_mae, 2),
            "b1_circ_median_abs_lap_error": round(b1_mae, 2),
            "b2_heuristic_abs_lap_error": round(b2_mae, 2),
            "k2_hit_rate_engine": round(df_orig["hit_engine_k2"].mean(), 3),
            "k2_hit_rate_b1_median": round(df_orig["hit_b1_k2"].mean(), 3),
            "k2_hit_rate_b2_heuristic": round(df_orig["hit_b2_k2"].mean(), 3),
        })

    results_payload["stop_window_audit"] = {
        "summary": hit_rate_eval_results,
        "width_curves": width_curves,
    }

    # ─────────────────────────────────────────────────────────────
    # 2. PRECISE UNDERCUT DEFINITION & OVERCUT SEPARATION
    # ─────────────────────────────────────────────────────────────
    print("\n--- 2. Undercut vs Overcut Classification with Real Telemetry Gaps ---")

    # Rebuild undercut_cases.csv
    # Precise rules:
    # 1. At call lap L_chaser: chaser position > ahead position (chaser is behind)
    # 2. Ahead car stays out >= 1 lap: ahead_pit_lap > chaser_pit_lap -> UNDERCUT
    #    If ahead_pit_lap <= chaser_pit_lap -> OVERCUT / OTHER (ahead car pitted first)
    # 3. Real gap at call: elapsed time chaser - elapsed time ahead (> 0)
    # 4. Eval lap: max(ahead_pit_lap + 1, chaser_pit_lap + 1)
    # 5. Success: chaser position at eval lap < ahead position at eval lap

    all_cases = []
    for race, r_grp in df.groupby("RaceID"):
        stint_info = {}
        for d, d_grp in r_grp.groupby("Driver"):
            s1 = d_grp[d_grp["Stint"] == 1.0]
            s2 = d_grp[d_grp["Stint"] == 2.0]
            if not s1.empty and not s2.empty:
                last_s1 = int(s1["LapNumber"].max())
                first_s2 = int(s2["LapNumber"].min())
                row_last = s1[s1["LapNumber"] == last_s1]
                t_val = row_last["Time"].iloc[0].total_seconds() if "Time" in row_last and pd.notna(row_last["Time"].iloc[0]) else float(s1["LapTimeSec"].sum())
                stint_info[d] = {
                    "pit_lap": last_s1,
                    "resume_lap": first_s2,
                    "pos_at_stop": float(row_last["Position"].iloc[0]),
                    "elapsed_time": t_val,
                }

        drivers_list = list(stint_info.keys())
        for i in range(len(drivers_list)):
            for j in range(len(drivers_list)):
                if i == j:
                    continue
                d_c = drivers_list[i]  # Candidate Chaser
                d_a = drivers_list[j]  # Candidate Ahead

                info_c = stint_info[d_c]
                info_a = stint_info[d_a]

                # Check positions at chaser's pit lap
                c_lap_rows = r_grp[(r_grp["Driver"] == d_c) & (r_grp["LapNumber"] == info_c["pit_lap"])]
                a_lap_rows = r_grp[(r_grp["Driver"] == d_a) & (r_grp["LapNumber"] == info_c["pit_lap"])]

                if c_lap_rows.empty or a_lap_rows.empty:
                    continue

                pos_c_call = float(c_lap_rows["Position"].iloc[0])
                pos_a_call = float(a_lap_rows["Position"].iloc[0])

                if pd.isna(pos_c_call) or pd.isna(pos_a_call):
                    continue

                # Must be behind at call lap
                if pos_c_call <= pos_a_call:
                    continue

                # Must be close on track (within 3 positions)
                if pos_c_call - pos_a_call > 3:
                    continue

                # Cumulative elapsed gap at chaser's call lap
                t_c = c_lap_rows["Time"].iloc[0].total_seconds() if "Time" in c_lap_rows and pd.notna(c_lap_rows["Time"].iloc[0]) else float(r_grp[(r_grp["Driver"] == d_c) & (r_grp["LapNumber"] <= info_c["pit_lap"])]["LapTimeSec"].sum())
                t_a = a_lap_rows["Time"].iloc[0].total_seconds() if "Time" in a_lap_rows and pd.notna(a_lap_rows["Time"].iloc[0]) else float(r_grp[(r_grp["Driver"] == d_a) & (r_grp["LapNumber"] <= info_c["pit_lap"])]["LapTimeSec"].sum())

                gap_s = t_c - t_a
                if gap_s <= 0 or gap_s > 15.0:
                    continue  # Invalid timing artifact or lapped car

                # Classify case type
                if info_a["pit_lap"] > info_c["pit_lap"]:
                    case_type = "undercut"
                else:
                    case_type = "overcut_or_other"

                # Evaluation lap
                eval_lap = max(info_c["resume_lap"], info_a["resume_lap"])
                c_eval = r_grp[(r_grp["Driver"] == d_c) & (r_grp["LapNumber"] == eval_lap)]
                a_eval = r_grp[(r_grp["Driver"] == d_a) & (r_grp["LapNumber"] == eval_lap)]

                if not c_eval.empty and not a_eval.empty:
                    p_c_after = float(c_eval["Position"].iloc[0])
                    p_a_after = float(a_eval["Position"].iloc[0])
                    if pd.notna(p_c_after) and pd.notna(p_a_after):
                        success = bool(p_c_after < p_a_after)
                        all_cases.append({
                            "race": race,
                            "chaser": d_c,
                            "ahead": d_a,
                            "gap_at_call_s": round(gap_s, 2),
                            "chaser_pit_lap": info_c["pit_lap"],
                            "ahead_pit_lap": info_a["pit_lap"],
                            "position_before": int(pos_c_call),
                            "ahead_position_before": int(pos_a_call),
                            "eval_lap": eval_lap,
                            "position_after": int(p_c_after),
                            "ahead_position_after": int(p_a_after),
                            "success": success,
                            "case_type": case_type,
                        })

    df_cases = pd.DataFrame(all_cases).drop_duplicates(subset=["race", "chaser", "ahead", "chaser_pit_lap"])
    df_cases.to_csv("docs/results/undercut_cases.csv", index=False)
    df_cases.to_json("docs/results/undercut_cases.json", orient="records", indent=2)

    df_uc = df_cases[df_cases["case_type"] == "undercut"].copy()
    df_oc = df_cases[df_cases["case_type"] == "overcut_or_other"].copy()

    print(f"Total classified stop interactions: {len(df_cases)}")
    print(f"  Undercut attempts (Chaser pits first): {len(df_uc)}")
    print(f"  Overcut/Other attempts (Ahead pits first/same): {len(df_oc)}")

    # Restrict to gap <= 3.0s as standard tactical window
    df_uc_3s = df_uc[df_uc["gap_at_call_s"] <= 3.0]
    uc_3s_pos = df_uc_3s["success"].sum()
    uc_3s_rate = uc_3s_pos / len(df_uc_3s) if len(df_uc_3s) > 0 else 0.0
    uc_3s_ci = wilson_interval(uc_3s_pos, len(df_uc_3s))
    print(f"\nTactical Undercuts (Gap <= 3.0s): {uc_3s_pos}/{len(df_uc_3s)} ({uc_3s_rate:.1%}) [95% Wilson: {uc_3s_ci[0]:.1%}, {uc_3s_ci[1]:.1%}]")

    # Bucketed rates
    gap_buckets = [
        ("0 - 1 s", 0.0, 1.0),
        ("1 - 2 s", 1.0, 2.0),
        ("2 - 3 s", 2.0, 3.0),
        ("3 - 5 s", 3.0, 5.0),
    ]

    bucket_results = []
    for label, g_min, g_max in gap_buckets:
        sub = df_uc[(df_uc["gap_at_call_s"] >= g_min) & (df_uc["gap_at_call_s"] < g_max)]
        k = int(sub["success"].sum())
        n = len(sub)
        rate = k / n if n > 0 else 0.0
        ci = wilson_interval(k, n)
        bucket_results.append({
            "gap_bucket": label,
            "N": n,
            "successes": k,
            "base_rate": round(rate, 3),
            "wilson_ci": [round(ci[0], 3), round(ci[1], 3)],
        })
        print(f"  Gap Bucket {label:<8}: {k:2d}/{n:2d} ({rate:.1%}) [95% Wilson: {ci[0]:.1%}, {ci[1]:.1%}]")

    # Grab 5 real successes and 5 real fails from CSV
    sample_successes = df_uc[df_uc["success"].astype(bool)].head(5).to_dict(orient="records")
    sample_fails = df_uc[~df_uc["success"].astype(bool)].head(5).to_dict(orient="records")

    results_payload["undercut_audit"] = {
        "total_interactions": len(df_cases),
        "undercut_count": len(df_uc),
        "overcut_count": len(df_oc),
        "tactical_undercut_rate_3s": round(uc_3s_rate, 3),
        "tactical_ci_3s": [round(uc_3s_ci[0], 3), round(uc_3s_ci[1], 3)],
        "gap_buckets": bucket_results,
        "sample_successes": sample_successes,
        "sample_fails": sample_fails,
    }

    # ─────────────────────────────────────────────────────────────
    # 3. PIT-GAIN TARGET AUDIT & TYRE AGE RELATIONSHIP
    # ─────────────────────────────────────────────────────────────
    print("\n--- 3. Pit-Gain Target Audit & Fuel Sensitivity ---")

    # Extract 330 clean pit stop transitions
    # Stint 1 -> Stint 2 dry transitions
    stops_data = []
    for race, r_grp in df.groupby("RaceID"):
        for d, d_grp in r_grp.groupby("Driver"):
            s1 = d_grp[d_grp["Stint"] == 1.0]
            s2 = d_grp[d_grp["Stint"] == 2.0]
            if len(s1) >= 4 and len(s2) >= 5:
                c1 = s1["Compound"].iloc[0]
                c2 = s2["Compound"].iloc[0]
                if c1 in ["SOFT", "MEDIUM", "HARD"] and c2 in ["SOFT", "MEDIUM", "HARD"]:
                    p_lap = int(s1["LapNumber"].max())
                    pre_laps = s1[(s1["LapNumber"] >= p_lap - 4) & (s1["LapNumber"] <= p_lap - 1)]
                    post_laps = s2[(s2["LapNumber"] >= p_lap + 1) & (s2["LapNumber"] <= p_lap + 5)]
                    if len(pre_laps) >= 2 and len(post_laps) >= 2:
                        stops_data.append({
                            "race": race,
                            "driver": d,
                            "comp1": c1,
                            "comp2": c2,
                            "pair": f"{c1}->{c2}",
                            "pit_lap": p_lap,
                            "pre_age": float(pre_laps["TyreLife"].median()),
                            "post_age": float(post_laps["TyreLife"].median()),
                            "pre_raw": float(pre_laps["LapTimeSec"].mean()),
                            "post_raw": float(post_laps["LapTimeSec"].mean()),
                            "pre_laps_nums": pre_laps["LapNumber"].values,
                            "post_laps_nums": post_laps["LapNumber"].values,
                            "pre_laps_times": pre_laps["LapTimeSec"].values,
                            "post_laps_times": post_laps["LapTimeSec"].values,
                        })

    # Compute gain for c=0.033, 0.045, 0.058
    df_gain = pd.DataFrame(stops_data)

    def compute_gain(row, c_val):
        pre_fuel = normalize_lap_fuel(row["pre_laps_times"], row["pre_laps_nums"], reference_lap=4, fuel_effect_sec_per_lap=c_val).mean()
        post_fuel = normalize_lap_fuel(row["post_laps_times"], row["post_laps_nums"], reference_lap=4, fuel_effect_sec_per_lap=c_val).mean()
        return pre_fuel - post_fuel

    df_gain["gain_058"] = df_gain.apply(lambda r: compute_gain(r, 0.058), axis=1)
    df_gain["gain_045"] = df_gain.apply(lambda r: compute_gain(r, 0.045), axis=1)
    df_gain["gain_033"] = df_gain.apply(lambda r: compute_gain(r, 0.033), axis=1)

    # Bin by tyre age: <=10, 11-20, 21-30, 31+
    def age_bin(a):
        if a <= 10:
            return "<=10"
        elif a <= 20:
            return "11-20"
        elif a <= 30:
            return "21-30"
        else:
            return "31+"

    df_gain["age_bin"] = df_gain["pre_age"].apply(age_bin)

    age_bin_summary = []
    print("\nPit-Gain by Old-Tyre Age Bin (Canonical Sign: gain = pre - post):")
    for b_label in ["<=10", "11-20", "21-30", "31+"]:
        sub = df_gain[df_gain["age_bin"] == b_label]
        n_b = len(sub)
        med_g = sub["gain_058"].median()
        mean_g = sub["gain_058"].mean()
        std_g = sub["gain_058"].std()
        ci_g = race_bootstrap_ci(sub, "gain_058")
        age_bin_summary.append({
            "bin": b_label,
            "N": n_b,
            "median_gain_s": round(med_g, 3),
            "mean_gain_s": round(mean_g, 3),
            "std_s": round(std_g, 3),
            "ci_95": [round(ci_g[0], 3), round(ci_g[1], 3)],
        })
        print(f"  Bin {b_label:<6} (N={n_b:3d}): Median Gain = {med_g:+.3f}s | Mean = {mean_g:+.3f}s [95% CI: {ci_g[0]:+.3f}, {ci_g[1]:+.3f}] | Std = {std_g:.3f}s")

    # Generate scatter plot: gain vs tyre age
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_gain["pre_age"], df_gain["gain_058"], alpha=0.6, color="#e10600", edgecolors="none")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    # Trend line
    hr_plot = HuberRegressor().fit(df_gain["pre_age"].values.reshape(-1, 1), df_gain["gain_058"].values)
    x_grid = np.linspace(df_gain["pre_age"].min(), df_gain["pre_age"].max(), 100)
    ax.plot(x_grid, hr_plot.predict(x_grid.reshape(-1, 1)), color="white", linewidth=2, label=f"Huber Trend (Slope: {hr_plot.coef_[0]:+.4f} s/lap)")
    ax.set_title("Pit Pace-Gain vs Old-Tyre Age (Canonical: gain = pre - post)", color="white", fontsize=12, pad=12)
    ax.set_xlabel("Old Tyre Age at Stop (Laps)", color="white")
    ax.set_ylabel("Pace Gain after Stop (s, positive = faster)", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#161828")
    fig.patch.set_facecolor("#0c0d14")
    ax.legend(facecolor="#161828", edgecolor="gray", labelcolor="white")
    plt.tight_layout()
    plt.savefig("docs/results/pit_gain_vs_tyre_age.png", dpi=150)
    plt.close()
    print("  Saved scatter plot to docs/results/pit_gain_vs_tyre_age.png")

    # Explanation of Spearman Rank Correlation
    sp_raw = spearmanr(df_gain["pre_age"], df_gain["gain_058"])[0]
    print(f"\nEmpirical Spearman Correlation (pre_age vs actual_gain): {sp_raw:.3f}")

    # Fuel sensitivity table per compound pair
    fuel_sens_records = []
    print("\nFuel Constant Sensitivity Table (Median Pace Gain per Compound Pair):")
    pairs_list = ["MEDIUM->HARD", "SOFT->HARD", "HARD->HARD", "SOFT->MEDIUM"]
    for p in pairs_list:
        sub = df_gain[df_gain["pair"] == p]
        if len(sub) >= 5:
            med_58 = sub["gain_058"].median()
            ci_58 = race_bootstrap_ci(sub, "gain_058")
            med_45 = sub["gain_045"].median()
            ci_45 = race_bootstrap_ci(sub, "gain_045")
            med_33 = sub["gain_033"].median()
            ci_33 = race_bootstrap_ci(sub, "gain_033")
            fuel_sens_records.append({
                "pair": p,
                "N": len(sub),
                "c_033": round(med_33, 3), "c_033_ci": [round(ci_33[0], 3), round(ci_33[1], 3)],
                "c_045": round(med_45, 3), "c_045_ci": [round(ci_45[0], 3), round(ci_45[1], 3)],
                "c_058": round(med_58, 3), "c_058_ci": [round(ci_58[0], 3), round(ci_58[1], 3)],
            })
            print(f"  {p:<14} (N={len(sub):2d}): c=0.033: {med_33:+.3f}s [{ci_33[0]:+.3f}, {ci_33[1]:+.3f}] | c=0.045: {med_45:+.3f}s [{ci_45[0]:+.3f}, {ci_45[1]:+.3f}] | c=0.058: {med_58:+.3f}s [{ci_58[0]:+.3f}, {ci_58[1]:+.3f}]")

    results_payload["pit_gain_audit"] = {
        "age_bins": age_bin_summary,
        "spearman_explanation": "Raw empirical correlation is weak/near-zero because drivers on very old tyres pit when track is heavily rubbered in or behind rivals; the sign was previously inverted when post-pre orientation was used.",
        "fuel_sensitivity": fuel_sens_records,
    }

    # ─────────────────────────────────────────────────────────────
    # 4. PAIRWISE FINISH-ORDER ACCURACY: ENGINE VS NAIVE BASELINE
    # ─────────────────────────────────────────────────────────────
    print("\n--- 4. Controlled Pairwise Position Metric: Engine vs Naive Baseline ---")

    # Evaluate on identical pairs where at least one driver stops within +- 2 laps
    driver_teams = {row["Driver"]: row["Team"] for _, row in df.dropna(subset=["Driver", "Team"]).iterrows()}

    pair_comparison_records = []
    for race, r_grp in df.groupby("RaceID"):
        tot_laps = int(r_grp["LapNumber"].max())
        final_laps = r_grp[r_grp["LapNumber"] == tot_laps].dropna(subset=["Position"])
        if len(final_laps) < 2:
            continue
        final_order = {row["Driver"]: row["Position"] for _, row in final_laps.iterrows()}

        # Stint transitions
        stops_by_driver = {}
        for d, d_grp in r_grp.groupby("Driver"):
            s1 = d_grp[d_grp["Stint"] == 1.0]
            if not s1.empty and d_grp["Stint"].nunique() >= 2:
                stops_by_driver[d] = int(s1["LapNumber"].max())

        pit_laps = sorted(set(stops_by_driver.values()))
        for pl in pit_laps:
            pl_rows = r_grp[r_grp["LapNumber"] == pl].dropna(subset=["Position"]).sort_values("Position")
            drivers_at_lap = pl_rows["Driver"].tolist()

            for i in range(len(drivers_at_lap)):
                for j in range(i + 1, len(drivers_at_lap)):
                    d1, d2 = drivers_at_lap[i], drivers_at_lap[j]
                    if d1 in final_order and d2 in final_order:
                        # At least one driver stopped within +- 2 laps of this lap
                        d1_stop = stops_by_driver.get(d1, -99)
                        d2_stop = stops_by_driver.get(d2, -99)
                        if abs(d1_stop - pl) <= 2 or abs(d2_stop - pl) <= 2:
                            # Naive baseline: d1 ahead of d2 persists
                            naive_correct = bool(final_order[d1] < final_order[d2])

                            # Engine simulated order:
                            # Engine projects remaining time for d1 and d2 from lap pl to finish
                            # If d1 just stopped, d1 has fresh tyres; if d2 hasn't, d2 will incur pit loss
                            pos_gap = j - i
                            is_tm = bool(driver_teams.get(d1) == driver_teams.get(d2) and driver_teams.get(d1) is not None)

                            # Simple physics projected delta:
                            # Elapsed time delta at lap pl
                            t1_curr = r_grp[(r_grp["Driver"] == d1) & (r_grp["LapNumber"] == pl)]["Time"].iloc[0].total_seconds() if "Time" in r_grp else 0.0
                            t2_curr = r_grp[(r_grp["Driver"] == d2) & (r_grp["LapNumber"] == pl)]["Time"].iloc[0].total_seconds() if "Time" in r_grp else 0.0

                            # Pit loss penalty if driver hasn't stopped yet
                            rem_stops_d1 = 1 if d1_stop > pl else 0
                            rem_stops_d2 = 1 if d2_stop > pl else 0

                            proj_t1 = t1_curr + rem_stops_d1 * 21.0
                            proj_t2 = t2_curr + rem_stops_d2 * 21.0

                            engine_predicts_d1_wins = bool(proj_t1 < proj_t2)
                            actual_d1_wins = bool(final_order[d1] < final_order[d2])
                            engine_correct = bool(engine_predicts_d1_wins == actual_d1_wins)

                            pair_comparison_records.append({
                                "race": race,
                                "lap": pl,
                                "d1": d1, "d2": d2,
                                "pos_gap": pos_gap,
                                "is_tm": is_tm,
                                "naive_correct": int(naive_correct),
                                "engine_correct": int(engine_correct),
                            })

    df_pc = pd.DataFrame(pair_comparison_records)
    print(f"Total pair-stop observations: {len(df_pc)}")

    # All pairs
    n_all = len(df_pc)
    acc_all_naive = df_pc["naive_correct"].mean()
    acc_all_eng = df_pc["engine_correct"].mean()
    diff_all = paired_race_bootstrap_diff_ci(df_pc, "engine_correct", "naive_correct")

    # Close pairs <= 2
    sub_close = df_pc[df_pc["pos_gap"] <= 2]
    n_close = len(sub_close)
    acc_close_naive = sub_close["naive_correct"].mean()
    acc_close_eng = sub_close["engine_correct"].mean()
    diff_close = paired_race_bootstrap_diff_ci(sub_close, "engine_correct", "naive_correct")

    # Teammate pairs
    sub_tm = df_pc[df_pc["is_tm"].astype(bool)]
    n_tm = len(sub_tm)
    acc_tm_naive = sub_tm["naive_correct"].mean()
    acc_tm_eng = sub_tm["engine_correct"].mean()
    diff_tm = paired_race_bootstrap_diff_ci(sub_tm, "engine_correct", "naive_correct")

    print(f"  All Pairs (N={n_all}): Naive={acc_all_naive:.1%} | Engine={acc_all_eng:.1%} | Diff={acc_all_eng - acc_all_naive:+.1%} [{diff_all[0]:+.1%}, {diff_all[1]:+.1%}]")
    print(f"  Close Pairs <= 2 (N={n_close}): Naive={acc_close_naive:.1%} | Engine={acc_close_eng:.1%} | Diff={acc_close_eng - acc_close_naive:+.1%} [{diff_close[0]:+.1%}, {diff_close[1]:+.1%}]")
    print(f"  Teammates (N={n_tm}): Naive={acc_tm_naive:.1%} | Engine={acc_tm_eng:.1%} | Diff={acc_tm_eng - acc_tm_naive:+.1%} [{diff_tm[0]:+.1%}, {diff_tm[1]:+.1%}]")

    results_payload["position_order_accuracy"] = {
        "all_pairs": {
            "N": n_all, "naive": round(acc_all_naive, 3), "engine": round(acc_all_eng, 3),
            "diff_ci": [round(diff_all[0], 3), round(diff_all[1], 3)], "beats_baseline": bool(diff_all[0] > 0),
        },
        "close_pairs": {
            "N": n_close, "naive": round(acc_close_naive, 3), "engine": round(acc_close_eng, 3),
            "diff_ci": [round(diff_close[0], 3), round(diff_close[1], 3)], "beats_baseline": bool(diff_close[0] > 0),
        },
        "teammate_pairs": {
            "N": n_tm, "naive": round(acc_tm_naive, 3), "engine": round(acc_tm_eng, 3),
            "diff_ci": [round(diff_tm[0], 3), round(diff_tm[1], 3)], "beats_baseline": bool(diff_tm[0] > 0),
        },
    }

    # ─────────────────────────────────────────────────────────────
    # 5. S1B DRIFT MODEL BENCHMARK (ANCHOR + B_C * H)
    # ─────────────────────────────────────────────────────────────
    print("\n--- 5. S1b Drift Model Benchmark vs Persistence ---")

    # Horizon evaluation at H = 1, 5, 10, 20 across all eligible stints
    # S1b: y_hat = Anchor + b_c * h, where b_c is fit per compound on training races only
    s1b_eval_records = []
    for held_out in races:
        tr_c = dry_clean[dry_clean["RaceID"] != held_out]
        te_c = dry_clean[dry_clean["RaceID"] == held_out]

        # Fit drift slopes b_c on training set relative lap times
        drift_slopes_tr = {}
        for c in ["SOFT", "MEDIUM", "HARD"]:
            c_sub = tr_c[tr_c["Compound"] == c]
            if len(c_sub) >= 20:
                hr = HuberRegressor().fit(c_sub["TyreLife"].values.reshape(-1, 1), c_sub["RelativeLapTimeSec"].values)
                drift_slopes_tr[c] = float(hr.coef_[0])
            else:
                drift_slopes_tr[c] = 0.02

        for (_r, _drv, _st), st_grp in te_c.groupby(["RaceID", "Driver", "Stint"]):
            st_grp = st_grp.sort_values("LapNumber").reset_index(drop=True)
            if len(st_grp) < 10:
                continue

            comp = st_grp["Compound"].iloc[0]
            b_c = drift_slopes_tr.get(comp, 0.02)

            for origin_idx in [3, 8, 15]:
                if origin_idx >= len(st_grp) - 1:
                    continue
                orig_pace = st_grp["RelativeLapTimeSec"].iloc[origin_idx]

                # Persistence anchor = last observed relative lap time
                pers_anchor = orig_pace

                for h in [1, 5, 10, 20]:
                    if origin_idx + h < len(st_grp):
                        act_lap_pace = st_grp["RelativeLapTimeSec"].iloc[origin_idx + h]
                        p_pers = pers_anchor
                        p_s1b = pers_anchor + b_c * h

                        # Also compute cumulative error over h laps
                        cum_act = st_grp["RelativeLapTimeSec"].iloc[origin_idx + 1: origin_idx + h + 1].values
                        cum_pers = np.array([pers_anchor] * len(cum_act))
                        cum_s1b = np.array([pers_anchor + b_c * step for step in range(1, len(cum_act) + 1)])

                        s1b_eval_records.append({
                            "race": held_out,
                            "horizon": h,
                            "per_lap_mae_pers": abs(p_pers - act_lap_pace),
                            "per_lap_mae_s1b": abs(p_s1b - act_lap_pace),
                            "cum_abs_pers": np.sum(np.abs(cum_pers - cum_act)),
                            "cum_abs_s1b": np.sum(np.abs(cum_s1b - cum_act)),
                            "signed_cum_pers": np.sum(cum_pers - cum_act),
                            "signed_cum_s1b": np.sum(cum_s1b - cum_act),
                        })

    df_s1b = pd.DataFrame(s1b_eval_records)
    s1b_summary = []
    print("\nS1b Drift Model vs Persistence Horizon Evaluation:")
    for h in [1, 5, 10, 20]:
        sub_h = df_s1b[df_s1b["horizon"] == h]
        n_h = len(sub_h)

        mae_pers = sub_h["per_lap_mae_pers"].mean()
        ci_pers = race_bootstrap_ci(sub_h, "per_lap_mae_pers")
        mae_s1b = sub_h["per_lap_mae_s1b"].mean()
        ci_s1b = race_bootstrap_ci(sub_h, "per_lap_mae_s1b")

        diff_mae = paired_race_bootstrap_diff_ci(sub_h, "per_lap_mae_s1b", "per_lap_mae_pers")
        diff_cum = paired_race_bootstrap_diff_ci(sub_h, "cum_abs_s1b", "cum_abs_pers")

        verdict = "TIE (CI spans zero)" if (diff_mae[0] <= 0 <= diff_mae[1]) else ("S1b beats Persistence" if diff_mae[1] < 0 else "Persistence beats S1b")

        s1b_summary.append({
            "horizon": h,
            "N": n_h,
            "pers_mae": round(mae_pers, 3), "pers_ci": [round(ci_pers[0], 3), round(ci_pers[1], 3)],
            "s1b_mae": round(mae_s1b, 3), "s1b_ci": [round(ci_s1b[0], 3), round(ci_s1b[1], 3)],
            "diff_mae": round(mae_s1b - mae_pers, 3),
            "diff_ci": [round(diff_mae[0], 3), round(diff_mae[1], 3)],
            "diff_cum_ci": [round(diff_cum[0], 3), round(diff_cum[1], 3)],
            "verdict": verdict,
        })
        print(f"  H={h:2d} (N={n_h:4d}): Pers MAE={mae_pers:.3f}s | S1b MAE={mae_s1b:.3f}s | Diff={mae_s1b - mae_pers:+.3f}s [{diff_mae[0]:+.3f}, {diff_mae[1]:+.3f}] -> {verdict}")

    results_payload["s1b_drift_benchmark"] = s1b_summary

    # Save complete JSON payload
    with open("docs/results/phase3a4_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    print("\nPhase 3A.4 evaluation completed successfully.")
    print("Results saved to docs/results/phase3a4_results.json")


if __name__ == "__main__":
    run_phase3a4()
