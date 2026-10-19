import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor

from src.ml_pipeline import prepare_ml_features


def fit_s1a_components(train_df: pd.DataFrame, fuel_effect: float = 0.058):
    """
    Fit S1a components on training races only:
    1. Per-compound monotone degradation curves deg(c, age) = a1*age + a2*age^2 (monotone non-decreasing).
    2. Compound offsets relative to MEDIUM.
    """
    compounds = ["SOFT", "MEDIUM", "HARD"]
    deg_params = {}
    offsets = {}

    ref_laps_rem = train_df.groupby("RaceID")["LapsRemaining"].transform("max") - 4
    fuel_adj_pace = train_df["RelativeLapTimeSec"] + fuel_effect * (train_df["LapsRemaining"] - ref_laps_rem)

    train_df = train_df.copy()
    train_df["_FuelAdjPace"] = fuel_adj_pace

    # Fit per-compound monotone degradation using bounded linear regression
    for c in compounds:
        c_sub = train_df[train_df["Compound"].str.upper() == c]
        if len(c_sub) >= 20:
            age = c_sub["TyreLife"].values.astype(float)
            y = c_sub["_FuelAdjPace"].values.astype(float)

            early = c_sub[c_sub["TyreLife"] <= 5]
            base = float(early["_FuelAdjPace"].median()) if len(early) >= 5 else float(np.median(y))

            y_deg = np.maximum(0.0, y - base)
            # Scale age to prevent condition number issues
            A = np.column_stack([age, age**2])
            res = lsq_linear(A, y_deg, bounds=(0, np.inf))
            deg_params[c] = {"a1": float(res.x[0]), "a2": float(res.x[1]), "base": base}
        else:
            deg_params[c] = {"a1": 0.035, "a2": 0.0005, "base": 0.0}

    # Normalize offsets relative to MEDIUM
    med_base = deg_params.get("MEDIUM", {}).get("base", 0.0)
    for c in compounds:
        offsets[c] = deg_params.get(c, {}).get("base", 0.0) - med_base

    return deg_params, offsets


def calc_deg(c: str, age: float, deg_params: dict) -> float:
    params = deg_params.get(c.upper(), deg_params.get("MEDIUM", {"a1": 0.035, "a2": 0.0005}))
    return float(params["a1"] * age + params["a2"] * (age**2))


def run_phase27_benchmark():
    df_raw = pd.read_parquet("data/demo/demo_laps.parquet")
    print(f"Loaded demo dataset: {len(df_raw)} laps across {df_raw['RaceID'].nunique()} races.")

    X_A, y, _ = prepare_ml_features(df_raw, min_eval_lap=7, include_prev_pace=True)
    X_B, _, _ = prepare_ml_features(df_raw, min_eval_lap=7, include_prev_pace=False)

    df_eval = X_A.copy()
    df_eval["RelativeLapTimeSec"] = y.values
    df_eval["LapTimeSec"] = df_eval["_LapTimeSec"].values
    df_eval["DriverReferencePaceSec"] = df_eval["_DriverReferencePaceSec"].values
    df_eval["Compound"] = df_eval["_Compound"].values
    df_eval["Stint"] = df_eval["_Stint"].values
    df_eval["Driver"] = df_eval["_Driver"].values
    df_eval["RaceID"] = df_eval["_RaceID"].values

    races = sorted(df_eval["RaceID"].unique())
    fuel_effect = 0.058

    origins_config = [
        {"name": "Origin 2-4 (Early)", "min_age": 2, "max_age": 4},
        {"name": "Origin 7-9 (Mid)", "min_age": 7, "max_age": 9},
        {"name": "Origin 14-16 (Late)", "min_age": 14, "max_age": 16},
    ]
    horizons = [1, 5, 10, 20]
    models = ["Persistence", "Huber", "S1a", "Variant B", "Variant A"]

    horizon_results = {
        orig["name"]: {m: {h: [] for h in horizons} for m in models}
        for orig in origins_config
    }
    horizon_results["All Origins"] = {m: {h: [] for h in horizons} for m in models}

    pit_records = []

    feat_A = [c for c in X_A.columns if not c.startswith("_")]
    feat_B = [c for c in X_B.columns if not c.startswith("_")]

    print(f"\nStarting Leave-One-Race-Out (LORO) Evaluation across {len(races)} Grand Prix events...")
    for held_out in races:
        print(f"  Evaluating held-out race: {held_out}...")
        train_mask = df_eval["RaceID"] != held_out
        test_mask = df_eval["RaceID"] == held_out

        train_data = df_eval[train_mask]
        test_data = df_eval[test_mask]

        # 1. Fit S1a on train_data
        deg_params, compound_offsets = fit_s1a_components(train_data, fuel_effect=fuel_effect)

        # 2. Fit Variant A and Variant B on train_data
        gb_a = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=150, max_depth=5, l2_regularization=1.0, random_state=42)
        gb_a.fit(train_data[feat_A], train_data["RelativeLapTimeSec"])

        gb_b = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=150, max_depth=5, l2_regularization=1.0, random_state=42)
        gb_b.fit(train_data[feat_B], train_data["RelativeLapTimeSec"])

        # Pre-compute Variant B predictions for entire test fold in one vectorized call
        test_pred_B = gb_b.predict(test_data[feat_B])
        test_data = test_data.copy()
        test_data["_Pred_B"] = test_pred_B

        # 3. Horizon Evaluation on test_data
        for (_drv, _st), s_grp in test_data.groupby(["Driver", "Stint"]):
            s_grp = s_grp.sort_values("LapNumber").reset_index(drop=True)
            if len(s_grp) < 5:
                continue

            ref_laps_rem = s_grp["LapsRemaining"].max() - 4

            for orig_cfg in origins_config:
                o_name = orig_cfg["name"]
                cands = s_grp[(s_grp["TyreLife"] >= orig_cfg["min_age"]) & (s_grp["TyreLife"] <= orig_cfg["max_age"])]
                if cands.empty:
                    continue
                orig_idx = cands.index[0]
                orig_row = s_grp.iloc[orig_idx]
                orig_pace = float(orig_row["RelativeLapTimeSec"])
                orig_comp = str(orig_row["Compound"])

                # Prior clean laps in stint up to origin lap
                prior_stint = s_grp.iloc[: orig_idx + 1]
                last_5 = prior_stint.tail(5)
                anchor_vals = []
                for _, r_k in last_5.iterrows():
                    d_k = calc_deg(str(r_k["Compound"]), float(r_k["TyreLife"]), deg_params)
                    f_k = -fuel_effect * (r_k["LapsRemaining"] - ref_laps_rem)
                    anchor_vals.append(r_k["RelativeLapTimeSec"] - f_k - d_k)
                anchor = float(np.median(anchor_vals)) if anchor_vals else orig_pace

                # Prior laps for Huber: only origins with >= 6 prior laps in the stint
                huber_model = None
                if len(prior_stint) >= 6:
                    try:
                        hr = HuberRegressor()
                        hr_X = prior_stint["TyreLife"].values.reshape(-1, 1)
                        hr_y = prior_stint["RelativeLapTimeSec"].values + fuel_effect * (prior_stint["LapsRemaining"].values - ref_laps_rem)
                        hr.fit(hr_X, hr_y)
                        huber_model = hr
                    except Exception:
                        huber_model = None

                for h in horizons:
                    if orig_idx + h >= len(s_grp):
                        continue
                    fut_window = s_grp.iloc[orig_idx + 1 : orig_idx + h + 1]
                    if len(fut_window) < h:
                        continue

                    actuals = fut_window["RelativeLapTimeSec"].values
                    target_pace = actuals[-1]

                    # Persistence
                    p_pers = np.full(h, orig_pace)
                    signed_err_pers = p_pers[-1] - target_pace
                    stint_err_pers = np.sum(p_pers - actuals)
                    rec_pers = {"race": held_out, "abs_err": abs(signed_err_pers), "bias": signed_err_pers, "stint_err": abs(stint_err_pers), "signed_stint_err": stint_err_pers}
                    horizon_results[o_name]["Persistence"][h].append(rec_pers)
                    horizon_results["All Origins"]["Persistence"][h].append(rec_pers)

                    # Huber (only if >= 6 prior laps in stint)
                    if huber_model is not None:
                        fut_ages = fut_window["TyreLife"].values.reshape(-1, 1)
                        fut_rem = fut_window["LapsRemaining"].values
                        p_hub_fuel = huber_model.predict(fut_ages)
                        p_hub = p_hub_fuel - fuel_effect * (fut_rem - ref_laps_rem)
                        signed_err_hub = p_hub[-1] - target_pace
                        stint_err_hub = np.sum(p_hub - actuals)
                        rec_hub = {"race": held_out, "abs_err": abs(signed_err_hub), "bias": signed_err_hub, "stint_err": abs(stint_err_hub), "signed_stint_err": stint_err_hub}
                        horizon_results[o_name]["Huber"][h].append(rec_hub)
                        horizon_results["All Origins"]["Huber"][h].append(rec_hub)

                    # S1a
                    p_s1a = []
                    for _, fut_r in fut_window.iterrows():
                        fut_age = float(fut_r["TyreLife"])
                        fut_rem = float(fut_r["LapsRemaining"])
                        deg_term = calc_deg(orig_comp, fut_age, deg_params)
                        fuel_term = -fuel_effect * (fut_rem - ref_laps_rem)
                        p_s1a.append(anchor + fuel_term + deg_term)
                    p_s1a = np.array(p_s1a)
                    signed_err_s1a = p_s1a[-1] - target_pace
                    stint_err_s1a = np.sum(p_s1a - actuals)
                    rec_s1a = {"race": held_out, "abs_err": abs(signed_err_s1a), "bias": signed_err_s1a, "stint_err": abs(stint_err_s1a), "signed_stint_err": stint_err_s1a}
                    horizon_results[o_name]["S1a"][h].append(rec_s1a)
                    horizon_results["All Origins"]["S1a"][h].append(rec_s1a)

                    # Variant B
                    p_b = fut_window["_Pred_B"].values
                    signed_err_b = p_b[-1] - target_pace
                    stint_err_b = np.sum(p_b - actuals)
                    rec_b = {"race": held_out, "abs_err": abs(signed_err_b), "bias": signed_err_b, "stint_err": abs(stint_err_b), "signed_stint_err": stint_err_b}
                    horizon_results[o_name]["Variant B"][h].append(rec_b)
                    horizon_results["All Origins"]["Variant B"][h].append(rec_b)

                    # Variant A (recursive)
                    curr_p = orig_pace
                    p_a_seq = []
                    for step_idx in range(h):
                        s_row = fut_window[feat_A].iloc[[step_idx]].copy()
                        s_row["PrevLapRelativePace"] = curr_p
                        pred_step = float(gb_a.predict(s_row)[0])
                        p_a_seq.append(pred_step)
                        curr_p = pred_step
                    p_a_seq = np.array(p_a_seq)
                    signed_err_a = p_a_seq[-1] - target_pace
                    stint_err_a = np.sum(p_a_seq - actuals)
                    rec_a = {"race": held_out, "abs_err": abs(signed_err_a), "bias": signed_err_a, "stint_err": abs(stint_err_a), "signed_stint_err": stint_err_a}
                    horizon_results[o_name]["Variant A"][h].append(rec_a)
                    horizon_results["All Origins"]["Variant A"][h].append(rec_a)

        # 4. Pit Pace-Gain Accuracy on held-out race
        for drv, d_laps in test_data.groupby("Driver"):
            d_laps = d_laps.sort_values("LapNumber").reset_index(drop=True)
            stints = sorted(d_laps["Stint"].unique())
            for idx_s in range(len(stints) - 1):
                s1_id = stints[idx_s]
                s2_id = stints[idx_s + 1]

                stint1 = d_laps[d_laps["Stint"] == s1_id]
                stint2 = d_laps[d_laps["Stint"] == s2_id]

                if len(stint1) < 5 or len(stint2) < 6:
                    continue

                pit_lap_num = stint1["LapNumber"].max()
                caution_check = test_data[
                    (test_data["Driver"] == drv)
                    & (test_data["LapNumber"] >= pit_lap_num - 3)
                    & (test_data["LapNumber"] <= pit_lap_num + 3)
                ]
                if len(caution_check) < 4:
                    continue

                comp1 = str(stint1["Compound"].iloc[-1]).upper()
                comp2 = str(stint2["Compound"].iloc[0]).upper()

                old_laps = stint1.tail(5)
                new_laps = stint2.iloc[1:6]
                if len(old_laps) < 4 or len(new_laps) < 4:
                    continue

                mean_old_rel = float(old_laps["RelativeLapTimeSec"].mean())
                mean_new_rel = float(new_laps["RelativeLapTimeSec"].mean())

                mean_old_rem = float(old_laps["LapsRemaining"].mean())
                mean_new_rem = float(new_laps["LapsRemaining"].mean())

                fuel_diff = -fuel_effect * (mean_new_rem - mean_old_rem)
                actual_delta = (mean_new_rel - mean_old_rel) - fuel_diff

                pred_naive = 0.0

                comp_off = compound_offsets.get(comp2, 0.0) - compound_offsets.get(comp1, 0.0)
                deg_new = float(np.mean([calc_deg(comp2, float(a), deg_params) for a in new_laps["TyreLife"]]))
                deg_old = float(np.mean([calc_deg(comp1, float(a), deg_params) for a in old_laps["TyreLife"]]))
                pred_s1a = comp_off + (deg_new - deg_old)

                if len(stint1) >= 6:
                    try:
                        hr_pit = HuberRegressor()
                        hr_pit.fit(stint1["TyreLife"].values.reshape(-1, 1), stint1["RelativeLapTimeSec"].values)
                        extrap_old = float(hr_pit.predict([[old_laps["TyreLife"].max() + 1]])[0])
                        extrap_new = float(hr_pit.predict([[2]])[0])
                        pred_huber = extrap_new - extrap_old
                    except Exception:
                        pred_huber = 0.0
                else:
                    pred_huber = 0.0

                old_age_val = float(old_laps["TyreLife"].iloc[-1])
                pit_records.append({
                    "race": held_out,
                    "driver": drv,
                    "comp_pair": f"{comp1}->{comp2}",
                    "old_age": old_age_val,
                    "age_group": "<15" if old_age_val < 15 else "15-25" if old_age_val <= 25 else ">25",
                    "actual_delta": actual_delta,
                    "pred_naive": pred_naive,
                    "pred_s1a": pred_s1a,
                    "pred_huber": pred_huber,
                })

    print("LORO Evaluation Complete. Analyzing results...")

    full_deg, full_offsets = fit_s1a_components(df_eval, fuel_effect=fuel_effect)
    print("\n================== S1a FITTED PARAMETERS (Full Dataset) ==================")
    print("Compound Offsets (Relative to MEDIUM):")
    for c, off in full_offsets.items():
        print(f"  {c:7s}: {off:+.4f} s")
    print("Degradation Curves deg(c, age) = a1*age + a2*age^2:")
    for c, p in full_deg.items():
        print(f"  {c:7s}: a1 = {p['a1']:.5f}, a2 = {p['a2']:.6f} (Pace at age 10: {calc_deg(c, 10, full_deg):.3f}s, age 20: {calc_deg(c, 20, full_deg):.3f}s)")

    df_pit = pd.DataFrame(pit_records)
    print(f"\n================== PIT PACE-GAIN ACCURACY (N = {len(df_pit)} stops) ==================")
    mae_naive_pit = np.mean(np.abs(df_pit["actual_delta"] - df_pit["pred_naive"]))
    mae_s1a_pit = np.mean(np.abs(df_pit["actual_delta"] - df_pit["pred_s1a"]))
    mae_hub_pit = np.mean(np.abs(df_pit["actual_delta"] - df_pit["pred_huber"]))

    bias_s1a_pit = np.mean(df_pit["pred_s1a"] - df_pit["actual_delta"])
    corr_s1a_pit, _ = spearmanr(df_pit["pred_s1a"], df_pit["actual_delta"])
    corr_hub_pit, _ = spearmanr(df_pit["pred_huber"], df_pit["actual_delta"])

    print(f"Naive Delta=0 MAE   : {mae_naive_pit:.3f} s")
    print(f"Huber Baseline MAE  : {mae_hub_pit:.3f} s (Rank Corr: {corr_hub_pit:.3f})")
    print(f"S1a Model MAE       : {mae_s1a_pit:.3f} s (Rank Corr: {corr_s1a_pit:.3f}, Bias: {bias_s1a_pit:+.3f} s)")
    print(f"Gate Check: S1a beats Naive MAE? {'YES' if mae_s1a_pit < mae_naive_pit else 'NO'} ({mae_s1a_pit:.3f} vs {mae_naive_pit:.3f})")

    print("\nPit Pace-Gain Breakdown by Compound Pair:")
    pair_summary = df_pit.groupby("comp_pair").apply(
        lambda g: pd.Series({
            "N": len(g),
            "Actual_Delta": g["actual_delta"].mean(),
            "S1a_MAE": np.mean(np.abs(g["actual_delta"] - g["pred_s1a"])),
            "Naive_MAE": np.mean(np.abs(g["actual_delta"] - g["pred_naive"])),
            "S1a_Bias": np.mean(g["pred_s1a"] - g["actual_delta"]),
        })
    ).reset_index()
    print(pair_summary.to_string(index=False))

    print("\nPit Pace-Gain Breakdown by Old Tyre Age:")
    age_summary = df_pit.groupby("age_group").apply(
        lambda g: pd.Series({
            "N": len(g),
            "Actual_Delta": g["actual_delta"].mean(),
            "S1a_MAE": np.mean(np.abs(g["actual_delta"] - g["pred_s1a"])),
            "Naive_MAE": np.mean(np.abs(g["actual_delta"] - g["pred_naive"])),
            "S1a_Bias": np.mean(g["pred_s1a"] - g["actual_delta"]),
        })
    ).reset_index()
    print(age_summary.to_string(index=False))

    print("\n================== HORIZON EVALUATION WITH BOOTSTRAP CIs (1,000 Draws) ==================")
    all_recs = horizon_results["All Origins"]
    np.random.seed(42)

    summary_rows = []
    for h in horizons:
        for m in models:
            m_list = all_recs[m][h]
            if not m_list:
                continue
            N_eval = len(m_list)
            df_m = pd.DataFrame(m_list)
            race_groups = dict(list(df_m.groupby("race")))
            race_keys = list(race_groups.keys())

            point_mae = np.mean(df_m["abs_err"])
            point_bias = np.mean(df_m["bias"])
            point_stint = np.mean(df_m["stint_err"])

            boot_maes, boot_biases, boot_stints = [], [], []
            for _ in range(1000):
                samp_races = np.random.choice(race_keys, size=len(race_keys), replace=True)
                samp_df = pd.concat([race_groups[r] for r in samp_races], ignore_index=True)
                boot_maes.append(np.mean(samp_df["abs_err"]))
                boot_biases.append(np.mean(samp_df["bias"]))
                boot_stints.append(np.mean(samp_df["stint_err"]))

            ci_mae = (np.percentile(boot_maes, 2.5), np.percentile(boot_maes, 97.5))
            ci_bias = (np.percentile(boot_biases, 2.5), np.percentile(boot_biases, 97.5))
            ci_stint = (np.percentile(boot_stints, 2.5), np.percentile(boot_stints, 97.5))

            summary_rows.append({
                "Origin": "All Origins",
                "Horizon": h,
                "Model": m,
                "N": N_eval,
                "MAE": point_mae,
                "MAE_95_CI": f"[{ci_mae[0]:.3f}, {ci_mae[1]:.3f}]",
                "Signed_Bias": point_bias,
                "Bias_95_CI": f"[{ci_bias[0]:+.3f}, {ci_bias[1]:+.3f}]",
                "Stint_Total_Err": point_stint,
                "Stint_95_CI": f"[{ci_stint[0]:.3f}, {ci_stint[1]:.3f}]",
            })

    df_horizon_summary = pd.DataFrame(summary_rows)
    print(df_horizon_summary[["Horizon", "Model", "N", "MAE", "MAE_95_CI", "Signed_Bias", "Stint_Total_Err", "Stint_95_CI"]].to_string(index=False))

    print("\nPer-Origin Stint-Total Error Breakdown:")
    per_orig_rows = []
    for orig_cfg in origins_config:
        o_name = orig_cfg["name"]
        for h in [10, 20]:
            for m in models:
                lst = horizon_results[o_name][m][h]
                if lst:
                    st_err = np.mean([x["stint_err"] for x in lst])
                    per_orig_rows.append({"Origin": o_name, "Horizon": h, "Model": m, "N": len(lst), "Stint_Total_Err": round(st_err, 3)})
    print(pd.DataFrame(per_orig_rows).to_string(index=False))

    print("\n================== S1a vs PERSISTENCE GATE CHECK ==================")
    for h in [10, 20]:
        df_p = pd.DataFrame(all_recs["Persistence"][h])
        df_s1a = pd.DataFrame(all_recs["S1a"][h])
        diff_stint = np.mean(df_s1a["stint_err"]) - np.mean(df_p["stint_err"])

        race_keys = sorted(df_p["race"].unique())
        p_groups = dict(list(df_p.groupby("race")))
        s_groups = dict(list(df_s1a.groupby("race")))

        boot_diffs = []
        for _ in range(1000):
            samp_races = np.random.choice(race_keys, size=len(race_keys), replace=True)
            samp_p = pd.concat([p_groups[r] for r in samp_races], ignore_index=True)
            samp_s = pd.concat([s_groups[r] for r in samp_races], ignore_index=True)
            boot_diffs.append(np.mean(samp_s["stint_err"]) - np.mean(samp_p["stint_err"]))

        ci_diff = (np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5))
        includes_zero = ci_diff[0] <= 0 <= ci_diff[1]
        print(f"H = {h} Laps Ahead:")
        print(f"  Persistence Stint-Total Error: {np.mean(df_p['stint_err']):.3f} s")
        print(f"  S1a Stint-Total Error        : {np.mean(df_s1a['stint_err']):.3f} s")
        print(f"  Delta (S1a - Persistence)    : {diff_stint:+.3f} s (95% CI: [{ci_diff[0]:+.3f}, {ci_diff[1]:+.3f}])")
        print(f"  Does 95% CI include zero?    : {'YES (Not statistically significant)' if includes_zero else 'NO'}")
        print(f"  S1a beats Persistence?       : {'YES' if diff_stint < 0 else 'NO'}")


if __name__ == "__main__":
    run_phase27_benchmark()
