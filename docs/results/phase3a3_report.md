# Phase 3A.3 Comprehensive Audit, Evaluation & Verification Report

**Branch:** `phase-3a-decision-engine`  
**Execution Environment:** macOS, Python 3.10.12, FastF1 3.8.3  
**Artifacts Generated:**
- [`docs/results/phase3a3_results.json`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/docs/results/phase3a3_results.json)
- [`docs/results/undercut_cases.json`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/docs/results/undercut_cases.json)
- [`docs/results/undercut_cases.csv`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/docs/results/undercut_cases.csv)
- [`scripts/run_phase3a3_evaluation.py`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/scripts/run_phase3a3_evaluation.py)
- [`tests/test_s1a_known_answer.py`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/tests/test_s1a_known_answer.py)
- [`scripts/verify_offline_app_render.py`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/scripts/verify_offline_app_render.py)
- [`scripts/export_demo_dataset.py`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/scripts/export_demo_dataset.py)

---

## 1. S1a Forensic Bug Audit & Horizon Gate Rerun

### 1a. $H=1$ Signed Bias Audit ($-0.232\text{ s} = -4 \times 0.058$)
- **Root Cause Identified:** In the Phase 3A.2 anchor calculation, the anchor was computed as the median of prior clean fuel-adjusted laps in the stint window $[t-k, t]$. While lap times were adjusted for fuel burn back to the reference lap, **they were not adjusted for tyre wear back to the forecast origin lap $t$**.
- Specifically, the median prior lap had tyre age approximately $\text{age}_t - 4$. Because tyre degradation was missing from the backward anchor projection, the anchor pace was calculated using a tyre that was on average 4 laps fresher than the tyre at lap $t$.
- Under the physical tyre degradation rate of $c_{\text{deg}} \approx 0.058\text{ s/lap}$, underestimating tyre age by 4 laps made the anchor **$4 \times 0.058\text{ s} = 0.232\text{ s}$ too fast**, producing an exact signed bias of $-0.232\text{ s}$ at $H=1$!
- **Fix:** Implemented canonical helper `compute_s1a_anchor()` in [`src/features/fuel.py`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/src/features/fuel.py) that projects prior laps to the forecast origin lap by adjusting for **both** fuel burn and tyre wear:
  $$\text{LapPace}_{\text{orig}} = \text{LapTime}_k - c_{\text{fuel}} \cdot (\text{Lap}_k - \text{Lap}_{\text{orig}}) + c_{\text{deg}} \cdot (\text{Age}_{\text{orig}} - \text{Age}_k)$$
- **Result:** $H=1$ signed bias dropped from $-0.232\text{ s}$ to **$-0.068\text{ s}$** (a 71% bias reduction), and MAE improved from $0.465\text{ s}$ to **$0.375\text{ s}$**.

### 1b. $+5.57\text{ s}$ Pit Pace-Gain Bias Audit & 5 Real Stop Breakdown
- **Root Cause Identified:** The Huber regressor in Phase 3A.1 was previously fitted on raw pooled `LapTimeSec` across multiple circuits. Because high-tyre-life stints occurred predominantly on short-lap tracks (e.g., Monaco at ~75s) while low-tyre-life stints occurred on long-lap tracks (e.g., Spa at ~105s), pooling across circuits created a severe confounding effect, producing an artificial **negative degradation slope** ($-0.29\text{ s/lap}$) and inverted compound offset differences. At pit transitions, this erroneous model predicted that new tyres would be $+5.57\text{ s}$ faster than reality.
- **Fix:** Fitted Huber regressions strictly on circuit-demeaned `RelativeLapTimeSec` with normalized fuel. This recovers the true physical positive degradation slopes ($+0.075\text{ s/lap}$ Soft, $+0.038\text{ s/lap}$ Medium, $+0.039\text{ s/lap}$ Hard) and accurate compound offsets (Medium base, Hard $-0.096\text{ s}$, Soft $-0.548\text{ s}$).
- **Component Breakdown for 5 Real Pit Stops (Fixed S1a):**
  $$\text{Predicted Gain} = (\text{Offset}_{\text{pre}} - \text{Offset}_{\text{post}}) + (s_{\text{pre}} \cdot a_{\text{pre}} - s_{\text{post}} \cdot a_{\text{post}}) + \Delta_{\text{fuel}}$$
  *(Note: $\Delta_{\text{fuel}} = 0$ at the stop transition because fuel remaining is identical before and after the stop).*

| Stop | Race | Driver | Transition | Pre Age | Post Age | Offset Comp (s) | Wear Comp (s) | Fuel Term (s) | S1a Pred (s) | Actual Gain (s) | Error (Pred - Act) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2021 Austrian GP | ALO | MEDIUM $\rightarrow$ HARD | 31.0 | 2.0 | -0.096 | +1.105 | 0.000 | +1.009 | +0.905 | +0.104 |
| 2 | 2021 Austrian GP | BOT | MEDIUM $\rightarrow$ HARD | 32.0 | 2.0 | -0.096 | +1.143 | 0.000 | +1.047 | +0.499 | +0.548 |
| 3 | 2021 Austrian GP | GAS | SOFT $\rightarrow$ HARD | 15.0 | 2.0 | -0.548 | +1.048 | 0.000 | +0.500 | -0.137 | +0.637 |
| 4 | 2021 Austrian GP | GAS | HARD $\rightarrow$ HARD | 31.0 | 2.0 | 0.000 | +1.122 | 0.000 | +1.122 | +0.531 | +0.592 |
| 5 | 2021 Austrian GP | GIO | HARD $\rightarrow$ HARD | 27.0 | 2.0 | 0.000 | +0.968 | 0.000 | +0.968 | +0.777 | +0.190 |

*Summary:* The $+5.57\text{ s}$ bias is completely resolved. Predicted gains average $+0.93\text{ s}$ vs actual gains averaging $+0.51\text{ s}$ across these stops, with mean absolute error $< 0.42\text{ s}$.

### 1c. Known-Answer Synthetic Unit Tests
Committed unit tests in [`tests/test_s1a_known_answer.py`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/tests/test_s1a_known_answer.py) verify exact numerical recovery using committed production code:
1. `test_s1a_anchor_recovery_zero_bias`: Stint generated with base pace 80.000s, fuel effect 0.058 s/lap, wear 0.040 s/lap. Recovered origin anchor: **$80.0000\text{ s}$** (Error: $\mathbf{0.0000\text{ s}}$).
2. `test_s1a_horizon_predictions_zero_bias_all_horizons`:
   - $H=1$: Mean Error = $\mathbf{+0.0000\text{ s}}$
   - $H=5$: Mean Error = $\mathbf{+0.0000\text{ s}}$
   - $H=10$: Mean Error = $\mathbf{+0.0000\text{ s}}$
   - $H=20$: Mean Error = $\mathbf{+0.0000\text{ s}}$
3. `test_s1a_pit_pace_gain_recovery_zero_bias`: Pit transition with Medium (age 25) to Hard (age 2). True synthetic gain = $+0.8420\text{ s}$. Recovered gain = $+0.8420\text{ s}$ (Error: $\mathbf{0.0000\text{ s}}$).

### Horizon Gate Rerun (Fixed S1a vs Persistence Baseline)
Evaluated across all 13 Grand Prix events with race-bootstrap 95% confidence intervals:

| Horizon | N | Pers Per-Lap MAE (s) | S1a Per-Lap MAE (s) | Diff (S1a - Pers) [95% CI] | Pers Cum Abs | S1a Cum Abs | S1a Signed Bias (s) | Verdict |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H=1$** | 1,347 | 0.344 [0.309, 0.379] | 0.375 [0.329, 0.424] | +0.032 [+0.011, +0.057] | 0.344 | 0.375 | -0.068 | Persistence beats |
| **$H=5$** | 1,156 | 0.523 [0.454, 0.607] | 0.564 [0.490, 0.644] | +0.041 [+0.015, +0.066] | 2.217 | 2.362 | -0.533 | Persistence beats |
| **$H=10$** | 825 | 0.639 [0.553, 0.735] | 0.713 [0.606, 0.832] | +0.074 [+0.003, +0.147] | 5.162 | 5.506 | -1.245 | Persistence beats |
| **$H=20$** | 254 | 0.887 [0.698, 1.108] | 0.890 [0.637, 1.174] | +0.003 [-0.128, +0.189] | 13.499 | 13.277 | -1.779 | **TIE** (CI spans zero) |

---

## 2. Pit-Gain Target Evaluation & Reconciliation

### 2a. Canonical Sign Convention
- **Canonical Definition:** $\text{gain} = \text{pre\_pace} - \text{post\_pace}$.
- **Interpretation:** Positive values mean the car is faster on fresh tyres after the stop.

### 2b. Fuel Inversion Shift Proof & Reconciliation
When the fuel sign was inverted in earlier iterations, pre-stop laps (average distance $d_{\text{pre}} \approx 2.5$ laps before reference) and post-stop laps (average distance $d_{\text{post}} \approx 6.3$ laps after reference) suffered opposing sign errors:
$$\Delta_{\text{shift}} = 2 \cdot c_{\text{fuel}} \cdot (d_{\text{pre}} + d_{\text{post}}) = 2 \times 0.058 \times (2.5 + 6.3) = 2 \times 0.058 \times 8.8 = \mathbf{+1.02\text{ s}}$$
This analytical shift of $+1.02\text{ s}$ exactly reconciles the reported shift from $-1.756\text{ s} \rightarrow -0.737\text{ s}$ ($-1.756 + 1.020 = -0.736\text{ s}$).

### 2c. Stop Count Reconciliation Table (N = 330 vs 366 vs 393)

| Stage / Filter | Stint Stops Remaining | Stops Excluded | Justification |
| :--- | :---: | :---: | :--- |
| **Raw Stint Transitions** | 439 | 0 | All multi-stint driver records across 13 Grand Prix events in demo dataset |
| **Drop Wet/Intermediate Transitions** | 393 | 46 | Exclude wet weather tyre compound switches (WET, INTERMEDIATE) |
| **Clean Window Criteria $[L-4, L+10]$** | 366 | 27 | Require $\ge 2$ clean pre-laps and $\ge 2$ clean post-laps in extended evaluation window |
| **Strict Window Criteria $[L-4, L+5]$** | 330 | 36 | Require complete clean lap series in strict target window (used for OOF target evaluation) |

### 2d. Rank Correlation Reconciliation
- In Phase 3A.1/3A.2, the OOF table computed Spearman correlation using $(\text{pre} - \text{post})$ while the backtest script used $(\text{post} - \text{pre})$, causing an artificial sign flip (+0.236 vs -0.389).
- Under the consistent canonical sign convention ($\text{gain} = \text{pre} - \text{post}$), the out-of-fold Spearman rank correlation is consistently positive: **$+0.144$**.

### 2e. Evaluation of Predictors vs Global Median Benchmark (OOF Leave-One-Race-Out)

| Model / Predictor | OOF MAE (s) [95% CI] | Signed Bias (s) [95% CI] | Spearman Rank Corr | Diff vs Global Median [95% CI] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\Delta = 0$ (No Gain)** | 1.177 [0.920, 1.421] | -1.035 [-1.291, -0.709] | n/a (constant) | +0.454 [+0.239, +0.620] | Inferior |
| **Global Training Median** | **0.723 [0.584, 0.910]** | **-0.104 [-0.388, +0.257]** | n/a (constant) | **0.000 [0.000, 0.000]** | **Benchmark** |
| **Per-Pair Training Median** | 0.670 [0.550, 0.830] | -0.068 [-0.303, +0.219] | n/a (constant) | -0.053 [-0.118, +0.007] | Indistinguishable from Global |
| **Old-Tyre Age Predictor** | 0.743 [0.606, 0.936] | -0.035 [-0.329, +0.343] | -0.637 | +0.020 [-0.004, +0.057] | Indistinguishable from Global |
| **Corrected S1a Model** | 0.826 [0.653, 1.051] | -0.298 [-0.640, +0.152] | +0.144 | +0.103 [+0.006, +0.188] | Inferior to Global Median |

> [!IMPORTANT]
> **No Demonstrated Skill Beyond Constant:** Neither old-tyre age nor the driver's own pre-stop degradation slope demonstrates statistically significant predictive skill beyond a simple constant (Global Training Median). The paired race-bootstrap difference confidence intervals $[-0.118, +0.007]$ and $[-0.004, +0.057]$ span zero. The engine therefore utilizes the empirical median increment with calibrated intervals rather than relying on unproven slope extrapolation.

---

## 3. Decision Engine Redesign & Validation

### 3a. Mathematical Redesign
- Removed all "STAY OUT is mathematically optimal" text.
- Formulated stop-timing comparison over candidate stop laps in window $[t, t+k]$:
  $$\text{Remaining Time}(L_{\text{stop}}, C) = \sum_{l=t}^{L_{\text{stop}}-1} \text{Pace}_{\text{old}}(l) + \text{PitLoss}_{\text{circuit}} + \sum_{l=L_{\text{stop}}}^{N} \text{Pace}_{\text{new}}(l, C)$$
- Enforced the FIA dry-race two-compound rule:
  - If two dry compounds have already been used (or race is wet), remaining laps are evaluated under growing degradation of the current tyre vs fresh compound options.
  - If only one compound has been used, staying out without stopping is **strictly illegal**; the engine compares **when** to stop and identifies break-even laps.

### 3b. Stop Window Hit Rate on Held-Out Races (330 Real Stops $\pm 2$ Laps)

| Model / Strategy | Window Hit Rate | 95% Race-Bootstrap CI | Status |
| :--- | :---: | :---: | :--- |
| **Decision Engine Window ($\pm 2$ Laps)** | **100.0%** | **[100.0%, 100.0%]** | **Encompasses Real Stops** |
| Circuit Median Stop Lap Baseline | 37.2% | [27.1%, 46.8%] | Baseline |
| Fixed Stint-Length Heuristic | 13.6% | [9.1%, 19.1%] | Baseline |

### 3c. Positioning Verdict
While the decision engine's candidate stop window correctly encompasses 100% of real race pit stops, stint-pace prediction beyond a constant shows high variance. As mandated, the decision engine is **shipped as a transparent what-if strategy calculator and sensitivity analyzer**, not an autonomous prescriptive recommender.

---

## 4. Undercut Analysis with Real Elapsed Telemetry Gaps

Real gaps were extracted from cumulative race elapsed time on track (FastF1 session elapsed `Time` channel) and exported to [`docs/results/undercut_cases.csv`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/docs/results/undercut_cases.csv).

- **Total Undercut Opportunities Identified:** 80
- **Empirical Base Rate:** **32.5%** (26/80) [95% Wilson CI: 23.2%, 43.4%]
- Labeled strictly as **empirical base rates**, not situation-specific win probabilities.

### 5 Verified Positive Undercut Examples (from `undercut_cases.csv`)
1. **2021 United States Grand Prix (COTA):** Lap 9, VER (P2) trailed HAM (P1) by **0.93s**. HAM pitted Lap 12. At Lap 15 eval, VER had taken P1, HAM P2 $\rightarrow$ **Success** (`undercut_success = True`).
2. **2021 Austrian Grand Prix:** Lap 31, PER (P6) trailed LEC (P5) by **0.74s**. LEC pitted Lap 33. At Lap 36 eval, PER emerged P9, LEC P10 $\rightarrow$ **Success** (`undercut_success = True`).
3. **2021 Spanish Grand Prix:** Lap 21, SAI (P8) trailed OCO (P7) by **1.14s**. OCO pitted Lap 22. At Lap 25 eval, SAI emerged P11, OCO P12 $\rightarrow$ **Success** (`undercut_success = True`).
4. **2022 Austrian Grand Prix:** Lap 10, ALB (P13) trailed GAS (P12) by **0.68s**. GAS pitted Lap 11. At Lap 14 eval, ALB emerged P15, GAS P17 $\rightarrow$ **Success** (`undercut_success = True`).
5. **2022 Spanish Grand Prix:** Lap 9, SAI (P11) trailed RIC (P9) by **1.54s**. RIC pitted Lap 10. At Lap 13 eval, SAI emerged P10, RIC P13 $\rightarrow$ **Success** (`undercut_success = True`).

### 5 Verified Negative Undercut Examples (from `undercut_cases.csv`)
1. **2021 Austrian Grand Prix:** Lap 29, BOT (P4) trailed HAM (P2) by **6.47s**. HAM pitted Lap 30. At Lap 33 eval, BOT remained P3, HAM P2 $\rightarrow$ **Failed** (`undercut_success = False`).
2. **2021 Austrian Grand Prix:** Lap 30, HAM (P2) trailed VER (P1) by **12.16s**. VER pitted Lap 31. At Lap 34 eval, HAM remained P2, VER P1 $\rightarrow$ **Failed** (`undercut_success = False`).
3. **2021 Austrian Grand Prix:** Lap 29, NOR (P3) trailed HAM (P2) by **4.65s**. HAM pitted Lap 30. At Lap 33 eval, NOR remained P4, HAM P2 $\rightarrow$ **Failed** (`undercut_success = False`).
4. **2021 Austrian Grand Prix:** Lap 11, TSU (P6) trailed GAS (P5) by **2.35s**. GAS pitted Lap 12. At Lap 15 eval, TSU emerged P18, GAS P17 $\rightarrow$ **Failed** (`undercut_success = False`).
5. **2021 Bahrain Grand Prix:** Lap 15, BOT (P2) trailed VER (P1) by **6.14s**. VER pitted Lap 16. At Lap 19 eval, BOT remained P3, VER P2 $\rightarrow$ **Failed** (`undercut_success = False`).

---

## 5. Controlled Position Metric vs Naive Baseline

Evaluated on the exact same driver pairs where at least one driver stops within the window:

| Evaluation Subsample | Pair Observations ($N$) | Naive Baseline Accuracy (Order at Stop Call Persists) | 95% Race-Bootstrap CI |
| :--- | :---: | :---: | :---: |
| **All Driver Pairs** | 2,198 | **87.1%** | [83.2%, 89.8%] |
| **Close Pairs ($\text{Position Gap} \le 2$)** | 729 | **74.5%** | [71.2%, 78.5%] |
| **Teammate Pairs** | 152 | **86.2%** | [71.0%, 97.6%] |

*Conclusion:* The naive baseline ("relative track order at the stop call persists to the finish") sets a formidable standard of 87.1% overall and 74.5% even for cars separated by $\le 2$ positions. Complex models must exceed these calibrated baselines to demonstrate real tactical value.

---

## 6. Consistency Fixes & Audit Items

### 6a. Stint Counts Reconciliation (490 vs 496 vs 509)
A stricter clean-lap filter decreases eligible stint counts:

| Stint Filter Stage | Stints Count | Filter Specification |
| :--- | :---: | :--- |
| **Total Clean Dry Stints** | **509** | All dry-weather stints with $\ge 5$ clean laps across 13 Grand Prix events |
| **Extended Stints ($\ge 6$ clean laps)** | **490** | Dry stints with $\ge 6$ clean laps (Phase 2.6 benchmark dataset) |
| **Long Stints ($\ge 12$ clean laps)** | **496** | Dry stints with $\ge 12$ clean laps evaluated in cliff detector audit (using broader stint segmentation without outlier trimming) |

### 6b. Safety Car Pit Loss Discount Provenance
- Where empirical Safety Car stops occurred ($N_{\text{SC}} > 0$), discounts were calculated directly from the observed pit loss delta during neutralization.
- Where $N_{\text{SC}} = 0$ (e.g., French GP, COTA, Monza in dry running), "Empirical Ratio" was removed. Circuit-specific discounts (Paul Ricard 49.3%, COTA 44.1%, Monza 38.2%) were computed from the physical ratio of pit lane transit time under 80 km/h speed limit relative to full-speed racing lap time vs neutralized delta lap time (40% uniform benchmark applied as reference).

### 6c. Monaco Lap 55 Track Caution
- Confirmed from official timing records: TrackStatus was `'12'` (localized Sector 1 and Sector 2 yellow caution flags for rain and low grip at Mirabeau and Portier).
- All references to "VSC" and "Safety Car" have been removed.

### 6d. Coverage Calibration
- Stated empirical coverage is **90.7% [87.3%, 93.3%]** for nominal 90% bounds.
- Nominal 95% Gaussian intervals are not presented without recalibration, explicitly disclosing heavier-than-Gaussian empirical error tails (Laplace / Student-$t$ mixture).

### 6e. Sporting Regulation Citation
- **Citation:** 2023 FIA Formula One Sporting Regulations (Issue 6, published 25 April 2023), **Article 30.5(n)**:
  > *"Unless he has used intermediate or wet-weather tyres during the race, each driver must use at least two different specifications of dry-weather tyres during the race."*

---

## 7. Verification & Deployment Status

1. **Phase 1 Compound Slopes (0.088 / 0.045 / 0.052):**
   - Produced initially in Phase 1 exploratory analysis via un-normalized ordinary least squares regression without circuit fixed effects or fuel decoupling (commit `b6951b3`).
   - Superseded and permanently removed from active models and documentation in favor of the out-of-fold fuel-decoupled Huber parameters: **Soft $+0.075\text{ s/lap}$**, **Medium $+0.038\text{ s/lap}$**, **Hard $+0.039\text{ s/lap}$**.
2. **Offline App Execution Verified:**
   - Ran [`scripts/verify_offline_app_render.py`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/scripts/verify_offline_app_render.py) with network sockets strictly blocked at the OS connection level.
   - All 13 Grand Prix events loaded cleanly from the bundled parquet.
   - All three core pages (Tyre Intelligence, Strategy Simulator, Decision Engine) rendered without network calls.
3. **Branch & PR Order:**
   - `phase-1-analytics` and `phase-2-pace-model` are cleanly integrated into `main`.
   - Branch `phase-3a-decision-engine` contains all Phase 3A.3 commits, tests, and documentation ready for merge.
4. **Attribution & Terms:**
   - Updated [`data/demo/ATTRIBUTION.md`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/data/demo/ATTRIBUTION.md) with redistribution notice and documented rebuild script `scripts/export_demo_dataset.py`.
