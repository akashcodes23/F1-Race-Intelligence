import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_FUEL_EFFECT_SEC_PER_LAP
from src.ml_pipeline import prepare_ml_features

warnings.filterwarnings('ignore')

# Load dataset
df_raw = pd.read_parquet('data/race_laps_dataset.parquet')

X_A, y_A, cols_A = prepare_ml_features(df_raw, min_eval_lap=7, include_prev_pace=True)
X_B, y_B, cols_B = prepare_ml_features(df_raw, min_eval_lap=7, include_prev_pace=False)

print(f"Dataset prepared: N={len(X_A)} laps >= 7 across {len(X_A['_RaceID'].unique())} races.")

# Fit S1 on training set
def fit_s1(X_tr, y_tr, fuel_effect=DEFAULT_FUEL_EFFECT_SEC_PER_LAP):
    # Delta t = fuel * (LapsRemaining - RefLapsRemaining) + deg_compound * TyreLife
    # We find RefLapsRemaining from race metadata or approximate as MaxLaps - 4
    # For training data, fuel effect is known (0.058 s/lap). We compute per-compound slopes:
    slopes = {}
    for comp in ['SOFT', 'MEDIUM', 'HARD']:
        c_mask = (X_tr[f'Compound_{comp}'] == 1.0)
        if c_mask.sum() < 10:
            slopes[comp] = 0.04
            continue
        # Fuel term
        # (LapsRemaining_t - RefLapsRemaining) = - (LapNumber_t - 4)
        fuel_term = fuel_effect * (X_tr.loc[c_mask, 'LapsRemaining'] - (X_tr.loc[c_mask, 'LapsRemaining'] + X_tr.loc[c_mask, 'LapNumber'] - 4.0))
        y_adj = y_tr[c_mask] - fuel_term
        # Fit deg * TyreLife through origin (or small intercept)
        tl = X_tr.loc[c_mask, 'TyreLife'].values
        # robust or least squares slope through origin
        deg_slope = np.sum(tl * y_adj.values) / np.maximum(np.sum(tl ** 2), 1e-6)
        slopes[comp] = float(deg_slope)
    return slopes

def predict_s1(X_te, slopes, fuel_effect=DEFAULT_FUEL_EFFECT_SEC_PER_LAP):
    preds = np.zeros(len(X_te))
    fuel_term = fuel_effect * (X_te['LapsRemaining'] - (X_te['LapsRemaining'] + X_te['LapNumber'] - 4.0))
    for comp in ['SOFT', 'MEDIUM', 'HARD']:
        c_mask = (X_te[f'Compound_{comp}'] == 1.0)
        if c_mask.sum() > 0:
            deg = slopes.get(comp, 0.03)
            preds[c_mask] = fuel_term[c_mask] + deg * X_te.loc[c_mask, 'TyreLife']
    return preds

# ── 1. Full LORO Cross-Validation for S1, S2, B, Constrained B, and A ──
race_ids = np.unique(X_A['_RaceID'])
loro_scores = {
    'S1 (Physics Linear)': [],
    'S2 (Scaled Ridge on B)': [],
    'Variant B (Unconstrained)': [],
    'Variant B (Monotonic)': [],
    'Variant A (Champion)': [],
}

models_B = {}
models_B_mono = {}

for r in race_ids:
    tr_mask = (X_A['_RaceID'] != r)
    te_mask = (X_A['_RaceID'] == r)

    # 1. S1
    s1_slopes = fit_s1(X_B[tr_mask], y_B[tr_mask])
    s1_preds = predict_s1(X_B[te_mask], s1_slopes)
    loro_scores['S1 (Physics Linear)'].append({
        'race': r, 'mae': mean_absolute_error(y_B[te_mask], s1_preds),
        'rmse': np.sqrt(mean_squared_error(y_B[te_mask], s1_preds))
    })

    # 2. S2: Scaled Ridge on Variant B
    s2_pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
    feat_cols_b = ['TyreLife', 'LapsRemaining', 'LapNumber', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']
    s2_pipe.fit(X_B.loc[tr_mask, feat_cols_b], y_B[tr_mask])
    s2_preds = s2_pipe.predict(X_B.loc[te_mask, feat_cols_b])
    loro_scores['S2 (Scaled Ridge on B)'].append({
        'race': r, 'mae': mean_absolute_error(y_B[te_mask], s2_preds),
        'rmse': np.sqrt(mean_squared_error(y_B[te_mask], s2_preds))
    })

    # 3. Variant B (Unconstrained)
    hgbr_b = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=150, max_depth=5, l2_regularization=1.0, min_samples_leaf=20, random_state=42
    )
    hgbr_b.fit(X_B.loc[tr_mask, feat_cols_b], y_B[tr_mask])
    b_preds = hgbr_b.predict(X_B.loc[te_mask, feat_cols_b])
    loro_scores['Variant B (Unconstrained)'].append({
        'race': r, 'mae': mean_absolute_error(y_B[te_mask], b_preds),
        'rmse': np.sqrt(mean_squared_error(y_B[te_mask], b_preds))
    })
    models_B[r] = hgbr_b

    # 4. Variant B (Monotonic Constraints: TyreLife +1, LapsRemaining +1)
    hgbr_b_mono = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=150, max_depth=5, l2_regularization=1.0, min_samples_leaf=20,
        monotonic_cst=[1, 1, 0, 0, 0, 0], random_state=42
    )
    hgbr_b_mono.fit(X_B.loc[tr_mask, feat_cols_b], y_B[tr_mask])
    b_mono_preds = hgbr_b_mono.predict(X_B.loc[te_mask, feat_cols_b])
    loro_scores['Variant B (Monotonic)'].append({
        'race': r, 'mae': mean_absolute_error(y_B[te_mask], b_mono_preds),
        'rmse': np.sqrt(mean_squared_error(y_B[te_mask], b_mono_preds))
    })
    models_B_mono[r] = hgbr_b_mono

    # 5. Variant A (with lagged pace)
    feat_cols_a = ['TyreLife', 'LapsRemaining', 'LapNumber', 'PrevLapRelativePace', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']
    hgbr_a = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=150, max_depth=5, l2_regularization=1.0, min_samples_leaf=20, random_state=42
    )
    hgbr_a.fit(X_A.loc[tr_mask, feat_cols_a], y_A[tr_mask])
    a_preds = hgbr_a.predict(X_A.loc[te_mask, feat_cols_a])
    loro_scores['Variant A (Champion)'].append({
        'race': r, 'mae': mean_absolute_error(y_A[te_mask], a_preds),
        'rmse': np.sqrt(mean_squared_error(y_A[te_mask], a_preds))
    })

print("\n=== LORO CROSS-VALIDATION SUMMARY (Laps >= 7) ===")
summary_rows = []
for model_name, folds in loro_scores.items():
    maes = [f['mae'] for f in folds]
    rmses = [f['rmse'] for f in folds]
    summary_rows.append({
        'Model': model_name,
        'Mean MAE (s)': np.mean(maes),
        'Std MAE (s)': np.std(maes),
        'Mean RMSE (s)': np.mean(rmses),
        'Std RMSE (s)': np.std(rmses),
    })
sum_df = pd.DataFrame(summary_rows)
print(sum_df.to_string(index=False))

# ── 2. Multi-Origin Horizon Evaluation at H = 1, 5, 10, 20 ──
# Origins at TyreLife ~ 3, 8, 15 across ALL eligible stints in held-out folds
horizons = [1, 5, 10, 20]
target_origins = [3, 8, 15]

# Data structures: model -> horizon -> list of (abs_err, signed_cum_err, stint_id)
horizon_data = {
    m: {h: {'abs_errs': [], 'cum_errs': [], 'N': 0} for h in horizons}
    for m in ['Persistence', 'Huber (Prior Stint)', 'S1', 'S2', 'Variant A (Recursive)', 'Variant B', 'Variant B (Monotonic)']
}

# Attach metadata to X_A for grouping
X_full = X_A.copy()
X_full['RelativeLapTimeSec'] = y_A.values

for r in race_ids:
    te_mask = (X_full['_RaceID'] == r)
    te_df = X_full[te_mask].copy()

    # Get models trained without race r
    tr_mask = (X_A['_RaceID'] != r)
    s1_sl = fit_s1(X_B[tr_mask], y_B[tr_mask])

    feat_b = ['TyreLife', 'LapsRemaining', 'LapNumber', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']
    s2_m = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]).fit(X_B.loc[tr_mask, feat_b], y_B[tr_mask])

    feat_a = ['TyreLife', 'LapsRemaining', 'LapNumber', 'PrevLapRelativePace', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']
    ha_m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=150, max_depth=5, l2_regularization=1.0, min_samples_leaf=20, random_state=42).fit(X_A.loc[tr_mask, feat_a], y_A[tr_mask])

    hb_m = models_B[r]
    hbm_m = models_B_mono[r]

    for (_drv, _stint), sgrp in te_df.groupby(['_Driver', '_Stint']):
        sgrp = sgrp.sort_values('TyreLife').reset_index(drop=True)
        if len(sgrp) < 10:
            continue

        for orig_tl in target_origins:
            # Find closest lap with TyreLife near orig_tl (+-1 lap)
            tl_diffs = np.abs(sgrp['TyreLife'] - orig_tl)
            if tl_diffs.min() > 1:
                continue
            orig_idx = int(tl_diffs.argmin())
            orig_row = sgrp.iloc[orig_idx]
            orig_pace = orig_row['RelativeLapTimeSec']

            # Prior laps for Huber
            prior_laps = sgrp.iloc[:orig_idx + 1]

            for h in horizons:
                end_idx = orig_idx + h
                if end_idx >= len(sgrp):
                    continue  # Not enough laps in this stint for horizon h

                target_row = sgrp.iloc[end_idx]
                target_pace = target_row['RelativeLapTimeSec']

                # Laps from orig_idx + 1 to end_idx (inclusive)
                horizon_slice = sgrp.iloc[orig_idx + 1 : end_idx + 1]
                actual_laps_y = horizon_slice['RelativeLapTimeSec'].values

                # 1. Persistence: predict orig_pace for all laps in horizon
                pred_pers = np.full(h, orig_pace)
                abs_e_pers = np.abs(pred_pers[-1] - target_pace)
                cum_e_pers = np.sum(pred_pers - actual_laps_y)
                horizon_data['Persistence'][h]['abs_errs'].append(abs_e_pers)
                horizon_data['Persistence'][h]['cum_errs'].append(cum_e_pers)
                horizon_data['Persistence'][h]['N'] += 1

                # 2. Huber extrapolation
                if len(prior_laps) >= 4:
                    huber = HuberRegressor().fit(prior_laps['TyreLife'].values.reshape(-1, 1), prior_laps['RelativeLapTimeSec'].values)
                    pred_huber = huber.predict(horizon_slice['TyreLife'].values.reshape(-1, 1))
                else:
                    pred_huber = np.full(h, orig_pace)
                abs_e_hub = np.abs(pred_huber[-1] - target_pace)
                cum_e_hub = np.sum(pred_huber - actual_laps_y)
                horizon_data['Huber (Prior Stint)'][h]['abs_errs'].append(abs_e_hub)
                horizon_data['Huber (Prior Stint)'][h]['cum_errs'].append(cum_e_hub)
                horizon_data['Huber (Prior Stint)'][h]['N'] += 1

                # 3. S1 (Static Baseline 1)
                pred_s1 = predict_s1(horizon_slice, s1_sl)
                abs_e_s1 = np.abs(pred_s1[-1] - target_pace)
                cum_e_s1 = np.sum(pred_s1 - actual_laps_y)
                horizon_data['S1'][h]['abs_errs'].append(abs_e_s1)
                horizon_data['S1'][h]['cum_errs'].append(cum_e_s1)
                horizon_data['S1'][h]['N'] += 1

                # 4. S2 (Scaled Ridge on B)
                pred_s2 = s2_m.predict(horizon_slice[feat_b])
                abs_e_s2 = np.abs(pred_s2[-1] - target_pace)
                cum_e_s2 = np.sum(pred_s2 - actual_laps_y)
                horizon_data['S2'][h]['abs_errs'].append(abs_e_s2)
                horizon_data['S2'][h]['cum_errs'].append(cum_e_s2)
                horizon_data['S2'][h]['N'] += 1

                # 5. Variant A (Recursive)
                curr_pace = orig_pace
                pred_a_seq = []
                for step_idx in range(h):
                    step_row = horizon_slice.iloc[step_idx].copy()
                    step_feat = step_row[feat_a].to_frame().T
                    step_feat['PrevLapRelativePace'] = curr_pace
                    p_step = float(ha_m.predict(step_feat)[0])
                    pred_a_seq.append(p_step)
                    curr_pace = p_step
                pred_a_seq = np.array(pred_a_seq)
                abs_e_a = np.abs(pred_a_seq[-1] - target_pace)
                cum_e_a = np.sum(pred_a_seq - actual_laps_y)
                horizon_data['Variant A (Recursive)'][h]['abs_errs'].append(abs_e_a)
                horizon_data['Variant A (Recursive)'][h]['cum_errs'].append(cum_e_a)
                horizon_data['Variant A (Recursive)'][h]['N'] += 1

                # 6. Variant B (Unconstrained)
                pred_b = hb_m.predict(horizon_slice[feat_b])
                abs_e_b = np.abs(pred_b[-1] - target_pace)
                cum_e_b = np.sum(pred_b - actual_laps_y)
                horizon_data['Variant B'][h]['abs_errs'].append(abs_e_b)
                horizon_data['Variant B'][h]['cum_errs'].append(cum_e_b)
                horizon_data['Variant B'][h]['N'] += 1

                # 7. Variant B (Monotonic)
                pred_bm = hbm_m.predict(horizon_slice[feat_b])
                abs_e_bm = np.abs(pred_bm[-1] - target_pace)
                cum_e_bm = np.sum(pred_bm - actual_laps_y)
                horizon_data['Variant B (Monotonic)'][h]['abs_errs'].append(abs_e_bm)
                horizon_data['Variant B (Monotonic)'][h]['cum_errs'].append(cum_e_bm)
                horizon_data['Variant B (Monotonic)'][h]['N'] += 1

print("\n=== MULTI-ORIGIN HORIZON BENCHMARK (N per cell, MAE at Horizon, Signed Cumulative Stint Error) ===")
h_rows = []
for model_name in horizon_data:
    for h in horizons:
        n_eval = horizon_data[model_name][h]['N']
        mae_h = np.mean(horizon_data[model_name][h]['abs_errs'])
        cum_err = np.mean(horizon_data[model_name][h]['cum_errs'])
        abs_cum_err = np.mean(np.abs(horizon_data[model_name][h]['cum_errs']))
        h_rows.append({
            'Model': model_name,
            'Horizon (H)': h,
            'N': n_eval,
            'MAE at H (s)': round(mae_h, 4),
            'Signed Cum Err (s)': round(cum_err, 4),
            'Abs Cum Err (s)': round(abs_cum_err, 4),
        })

h_df = pd.DataFrame(h_rows)
print(h_df.to_string(index=False))

# ── 3. Partial Dependence & Implied Slopes for Variant B ──
full_hgbr_b = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=150, max_depth=5, l2_regularization=1.0, min_samples_leaf=20, random_state=42)
full_hgbr_b.fit(X_B[feat_b], y_B)

full_hgbr_b_mono = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=150, max_depth=5, l2_regularization=1.0, min_samples_leaf=20, monotonic_cst=[1, 1, 0, 0, 0, 0], random_state=42)
full_hgbr_b_mono.fit(X_B[feat_b], y_B)

print("\n=== PARTIAL DEPENDENCE: Predicted Relative Pace vs TyreLife (at fixed LapsRemaining=30, LapNumber=25) ===")
tyre_lives = [1, 5, 10, 15, 20, 25, 30]
pd_tyre = []
for comp in ['SOFT', 'MEDIUM', 'HARD']:
    for tl in tyre_lives:
        row_dict = {
            'TyreLife': [tl], 'LapsRemaining': [30.0], 'LapNumber': [25.0],
            'Compound_SOFT': [1.0 if comp == 'SOFT' else 0.0],
            'Compound_MEDIUM': [1.0 if comp == 'MEDIUM' else 0.0],
            'Compound_HARD': [1.0 if comp == 'HARD' else 0.0]
        }
        df_row = pd.DataFrame(row_dict)
        pred_uncon = full_hgbr_b.predict(df_row)[0]
        pred_mono = full_hgbr_b_mono.predict(df_row)[0]
        pd_tyre.append({'Compound': comp, 'TyreLife': tl, 'Unconstrained Pace (s)': round(pred_uncon, 4), 'Monotonic Pace (s)': round(pred_mono, 4)})

pd_t_df = pd.DataFrame(pd_tyre)
print(pd_t_df.to_string(index=False))

# Implied tyre degradation slope (between TyreLife 5 and 25)
print("\n=== IMPLIED TYRE DEGRADATION SLOPES (TyreLife 5 -> 25) vs Phase 1 ===")
for comp in ['SOFT', 'MEDIUM', 'HARD']:
    sub = pd_t_df[pd_t_df['Compound'] == comp]
    p5_u = sub[sub['TyreLife'] == 5]['Unconstrained Pace (s)'].iloc[0]
    p25_u = sub[sub['TyreLife'] == 25]['Unconstrained Pace (s)'].iloc[0]
    slope_u = (p25_u - p5_u) / 20.0

    p5_m = sub[sub['TyreLife'] == 5]['Monotonic Pace (s)'].iloc[0]
    p25_m = sub[sub['TyreLife'] == 25]['Monotonic Pace (s)'].iloc[0]
    slope_m = (p25_m - p5_m) / 20.0

    print(f"{comp}: Unconstrained Slope = {slope_u:+.4f} s/lap | Monotonic Slope = {slope_m:+.4f} s/lap")

print("\n=== PARTIAL DEPENDENCE: Predicted Relative Pace vs LapsRemaining (at fixed TyreLife=10, LapNumber=25, MEDIUM) ===")
laps_rem_vals = [50, 40, 30, 20, 10, 0]
pd_fuel = []
for lr in laps_rem_vals:
    row_dict = {
        'TyreLife': [10.0], 'LapsRemaining': [lr], 'LapNumber': [25.0],
        'Compound_SOFT': [0.0], 'Compound_MEDIUM': [1.0], 'Compound_HARD': [0.0]
    }
    df_row = pd.DataFrame(row_dict)
    pred_u = full_hgbr_b.predict(df_row)[0]
    pred_m = full_hgbr_b_mono.predict(df_row)[0]
    pd_fuel.append({'LapsRemaining': lr, 'Unconstrained Pace (s)': round(pred_u, 4), 'Monotonic Pace (s)': round(pred_m, 4)})

pd_f_df = pd.DataFrame(pd_fuel)
print(pd_f_df.to_string(index=False))

# Implied fuel effect: as LapsRemaining decreases from 50 to 10 (40 laps burned)
p50_u = pd_f_df[pd_f_df['LapsRemaining'] == 50]['Unconstrained Pace (s)'].iloc[0]
p10_u = pd_f_df[pd_f_df['LapsRemaining'] == 10]['Unconstrained Pace (s)'].iloc[0]
implied_fuel_u = (p50_u - p10_u) / 40.0

p50_m = pd_f_df[pd_f_df['LapsRemaining'] == 50]['Monotonic Pace (s)'].iloc[0]
p10_m = pd_f_df[pd_f_df['LapsRemaining'] == 10]['Monotonic Pace (s)'].iloc[0]
implied_fuel_m = (p50_m - p10_m) / 40.0

print(f"Implied Fuel Burn Effect: Unconstrained = {implied_fuel_u:+.4f} s/lap | Monotonic = {implied_fuel_m:+.4f} s/lap | Config Baseline = {DEFAULT_FUEL_EFFECT_SEC_PER_LAP:.4f} s/lap")
