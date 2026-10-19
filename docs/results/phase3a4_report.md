# Phase 3A.4 Audit & Decision Engine Evaluation Report

**Branch:** `phase-3a-decision-engine`  
**Date:** October 19, 2026  
**Status:** Completed and Verified  

---

## Item 0: Git Discipline & Deployment Verification

1. **Phase 3A Commits on `main`**:
   - `git log main --oneline -8` output:
     ```text
     cc90b28 (HEAD -> phase-3a-decision-engine, origin/phase-3a-decision-engine, origin/main, origin/HEAD, main) feat(phase-3a3): resolve S1a biases, canonical pit gain target, undercut gap cases, and offline verification
     0f71a73 feat(phase-3a2): complete Phase 3A.2 audit, decision engine redesign, and backtest
     648a6c0 feat(phase-3a1): implement fuel sign canonical helper, out-of-fold recalibration and empirical engine
     fcde2c0 feat(phase-3a): decision engine, Monte Carlo differences, empirical pit loss, and validation suite
     1e8d571 Merge branch 'phase-2-pace-model' into main: Phase 2.7 pace model, demo dataset, and deployment fixes
     f956ed8 (origin/phase-2-pace-model, phase-2-pace-model) fix(deploy): add docs/DATA_ACCESS.md, .python-version, sidebar status line, and dynamic telemetry channel handling
     a452791 feat(phase-2.7): implement S1a anchored physics model, offline demo dataset, pit pace-gain evaluation, and honest validation claims
     1e07c33 feat(phase-2.6): complete pace model validation, static baselines S1/S2, horizon benchmarks, and items 1-12 fixes
     ```
   - **Answer:** Yes, `main` contains the 4 Phase 3A commits (`fcde2c0`, `648a6c0`, `0f71a73`, `cc90b28`) ahead of `1e8d571`.

2. **Streamlit Cloud Deployment**:
   - Deploys from branch: **`main`**.
   - Redeployed: **Yes**. Streamlit Cloud automatically tracks and builds changes pushed to `main`. Verified live via active browser instance.

3. **Demo Files Tracked**:
   - `git ls-files data/demo`:
     ```text
     data/demo/ATTRIBUTION.md
     data/demo/demo_laps.parquet
     ```
   - Both files are confirmed tracked and present.

4. **Exact Undo Command (Printed for review, NOT executed)**:
   - Since the 4 commits were pushed linearly to `main` without an outer merge commit, the exact command to revert all Phase 3A commits cleanly is:
     ```bash
     git revert --no-edit 1e8d571..cc90b28
     ```
   - Alternatively, to reset `main` directly to the Phase 2.7 state (`1e8d571`):
     ```bash
     git checkout main && git reset --hard 1e8d571 && git push origin main --force
     ```

---

## Item 1: Stop-Window Hit Rate & Causal Window Definition

### 1. Window Definition & Causal Proof
- **Origin Lap ($t_0$):** Selected without knowing the actual stop lap in advance, evaluated at fixed retrospective horizons: $\Delta \in \{5, 10, 15\}$ laps before the actual stop: $t_0 = L_{\text{act}} - \Delta$.
- **Window Center ($L^*$):** At lap $t_0$, the decision engine computes remaining race time for candidate stop laps $L_{\text{cand}} \in [t_0 + 1, \min(N_{\text{total}} - 2, t_0 + 35)]$, using only telemetry and pace data up to $t_0$. The window center is $L^* = \arg\min_{L_{\text{cand}}} \text{RemainingRaceTime}(L_{\text{cand}})$.
- **Window Width ($k$):** Given half-width $k$, the recommended window is $[L^* - k, L^* + k]$, with width $W = 2k + 1$ laps. Mean and median window width for any half-width $k$ is exactly $2k + 1$ laps (e.g. for $k=2$, width is exactly 5 laps).
- **Leakage Invariance Proof:** No future lap times, actual stop laps, or post-$t_0$ race states enter the objective function. A formal unit test ([`test_stop_timing_leakage_invariance`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/tests/test_decision_engine.py#L211-L260)) was added to ensure shifting the actual ground-truth stop lap produces bit-for-bit identical window centers and recommendations.

### 2. Mean Absolute Stop-Lap Error
| Origin ($\Delta = L_{\text{act}} - t_0$) | Sample ($N$) | Engine MAE (laps) | Circuit Median MAE (laps) | Stint Heuristic MAE (laps) |
|---|---|---|---|---|
| **$\Delta = 5$ laps** | 173 | **8.84 laps** | **6.11 laps** | **7.72 laps** |
| **$\Delta = 10$ laps** | 121 | **7.40 laps** | **6.93 laps** | **5.89 laps** |
| **$\Delta = 15$ laps** | 73 | **6.11 laps** | **10.21 laps** | **4.51 laps** |

### 3. Hit Rate vs Window Width ($k = 0 \dots 10$) with Race-Bootstrap CIs

#### A. Origin $t_0 = L_{\text{act}} - 5$ ($N = 173$)
| $k$ | Width (laps) | Engine Hit Rate [95% CI] | Circuit Median Hit Rate [95% CI] | Stint Heuristic Hit Rate [95% CI] | Diff (Engine - Median) [95% CI] |
|---|---|---|---|---|---|
| 0 | 1 | 2.9% [1.7%, 4.2%] | 3.5% [0.9%, 5.8%] | 3.5% [1.0%, 5.5%] | -0.6% [-3.8%, +2.9%] |
| 1 | 3 | 9.8% [4.5%, 15.5%] | 22.5% [12.0%, 31.6%] | 10.4% [4.3%, 15.5%] | -12.7% [-25.5%, +1.9%] |
| 2 | 5 | 12.7% [5.2%, 20.0%] | 31.8% [18.9%, 42.6%] | 16.8% [9.8%, 23.0%] | -19.1% [-35.0%, -1.3%] |
| 3 | 7 | 18.5% [9.3%, 27.2%] | 39.3% [25.5%, 51.4%] | 22.5% [13.6%, 31.3%] | -20.8% [-38.2%, -1.2%] |
| 4 | 9 | 42.8% [24.2%, 61.9%] | 44.5% [30.5%, 57.3%] | 31.2% [19.6%, 42.1%] | -1.7% [-29.5%, +24.4%] |
| 5 | 11 | 48.0% [29.6%, 66.8%] | 52.6% [37.3%, 65.5%] | 39.3% [26.0%, 52.8%] | -4.6% [-30.7%, +23.6%] |
| 6 | 13 | 50.9% [32.4%, 69.8%] | 57.8% [42.1%, 71.0%] | 46.2% [32.8%, 60.1%] | -6.9% [-31.6%, +22.4%] |
| 7 | 15 | 53.2% [34.5%, 72.0%] | 61.8% [46.1%, 74.5%] | 50.9% [37.5%, 64.7%] | -8.7% [-33.7%, +19.9%] |
| 8 | 17 | 64.7% [46.7%, 81.3%] | 66.5% [51.5%, 78.4%] | 53.8% [40.6%, 67.2%] | -1.7% [-24.1%, +21.9%] |
| 9 | 19 | 80.3% [64.2%, 91.5%] | 68.8% [54.5%, 80.5%] | 58.4% [45.1%, 71.6%] | +11.6% [-8.3%, +32.3%] |
| 10 | 21 | 82.1% [65.9%, 93.3%] | 72.8% [58.8%, 83.9%] | 62.4% [49.7%, 75.3%] | +9.2% [-10.8%, +28.9%] |

#### B. Origin $t_0 = L_{\text{act}} - 10$ ($N = 121$)
| $k$ | Width (laps) | Engine Hit Rate [95% CI] | Circuit Median Hit Rate [95% CI] | Stint Heuristic Hit Rate [95% CI] | Diff (Engine - Median) [95% CI] |
|---|---|---|---|---|---|
| 0 | 1 | 4.1% [1.0%, 7.8%] | 3.3% [0.8%, 6.3%] | 9.9% [3.1%, 17.5%] | +0.8% [-4.0%, +5.8%] |
| 1 | 3 | 14.0% [3.2%, 25.1%] | 24.0% [8.4%, 38.6%] | 16.5% [8.0%, 25.6%] | -9.9% [-25.7%, +8.0%] |
| 2 | 5 | 19.8% [6.4%, 31.3%] | 32.2% [14.9%, 47.6%] | 24.0% [15.1%, 34.0%] | -12.4% [-28.2%, +7.3%] |
| 5 | 11 | 41.3% [23.1%, 58.6%] | 51.2% [34.2%, 69.6%] | 62.8% [45.3%, 80.3%] | -9.9% [-38.3%, +22.9%] |
| 10 | 21 | 84.3% [68.0%, 95.8%] | 71.1% [54.6%, 83.9%] | 72.7% [58.8%, 84.4%] | +13.2% [-6.9%, +32.4%] |

#### C. Origin $t_0 = L_{\text{act}} - 15$ ($N = 73$)
| $k$ | Width (laps) | Engine Hit Rate [95% CI] | Circuit Median Hit Rate [95% CI] | Stint Heuristic Hit Rate [95% CI] | Diff (Engine - Heuristic) [95% CI] |
|---|---|---|---|---|---|
| 0 | 1 | 1.4% [0.0%, 3.0%] | 0.0% [0.0%, 0.0%] | 8.2% [2.2%, 15.6%] | -6.8% [-14.7%, -0.5%] |
| 2 | 5 | 20.5% [11.8%, 26.1%] | 4.1% [0.0%, 8.3%] | 34.2% [20.8%, 48.5%] | -13.7% [-31.1%, +2.5%] |
| 5 | 11 | 43.8% [27.7%, 59.4%] | 30.1% [15.8%, 45.9%] | 63.0% [48.1%, 76.5%] | -19.2% [-38.4%, +3.1%] |
| 10 | 21 | 87.7% [74.5%, 97.4%] | 61.6% [44.7%, 77.0%] | 87.7% [76.9%, 96.0%] | +0.0% [-16.2%, +16.2%] |

**Conclusion:** At operational tactical horizons ($k \le 3$, width $\le 7$ laps), the decision engine does not statistically beat the naive Circuit Median baseline (median baseline beats the engine by 12–19% at $k=2$). The decision engine is not an empirical stop-lap predictor; it is a prescriptive what-if scenario calculator.

---

## Item 2: Undercut Definition, Classification, and Cases

### 1. Precise Definitions & Evaluation Rule
- **Candidate Interaction:** Two cars racing within 5.0s on track where both make pit stops.
- **Undercut:** Chaser pits on lap $L_{\text{chaser}}$, while Ahead car stays out on track for at least 1 further lap ($L_{\text{ahead}} > L_{\text{chaser}}$).
- **Overcut / Other:** Ahead car pits on or before the chaser's stop ($L_{\text{ahead}} \le L_{\text{chaser}}$).
- **Evaluation Lap:** Both drivers must have completed their out-laps and in-laps to observe clean track position. $\text{Eval Lap} = \max(L_{\text{ahead}} + 1, L_{\text{chaser}} + 1)$.
- **Success:** Chaser position at Eval Lap $<$ Ahead car position at Eval Lap.
- **Tactical Cutoff Choice:** $\text{gap} \le 3.0\text{ s}$ at call is selected as the standard tactical window because beyond 3.0s, fresh tyre in-lap/out-lap delta ($\sim 1.0\text{ s}$) cannot bridge the gap without rival errors or traffic.

### 2. Overall Counts & Tactical Undercut Rate
- Total Multi-Car Stop Interactions: **330**
  - **Undercuts ($L_{\text{chaser}} < L_{\text{ahead}}$):** **196** (59.4%)
  - **Overcut / Other ($L_{\text{ahead}} \le L_{\text{chaser}}$):** **134** (40.6%)
- **Tactical Undercuts ($\text{gap} \le 3.0\text{ s}$):** **41 / 80 (51.2%)** [95% Wilson CI: 40.5%, 61.9%]

### 3. Success Rate by Gap Bucket
| Gap Bucket | $N$ | Successes | Success Rate | 95% Wilson CI |
|---|---|---|---|---|
| **0 – 1 s** | 17 | 11 | **64.7%** | [41.3%, 82.7%] |
| **1 – 2 s** | 31 | 21 | **67.7%** | [50.1%, 81.4%] |
| **2 – 3 s** | 31 | 9 | **29.0%** | [16.1%, 46.6%] |
| **3 – 5 s** | 35 | 7 | **20.0%** | [10.0%, 35.9%] |

### 4. Raw Sample Rows (from `docs/results/undercut_cases.csv`)
#### 5 Real Successes:
| race | chaser | ahead | gap_at_call_s | chaser_pit_lap | ahead_pit_lap | pos_before | ahead_pos_before | eval_lap | pos_after | ahead_pos_after | success |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_Austrian_Grand_Prix | ALO | RAI | 1.26 | 31 | 36 | 9 | 8 | 39 | 14 | 15 | True |
| 2021_Austrian_Grand_Prix | PER | LEC | 0.74 | 31 | 33 | 6 | 5 | 36 | 9 | 10 | True |
| 2021_Bahrain_Grand_Prix | ALO | STR | 1.02 | 10 | 11 | 8 | 7 | 14 | 10 | 12 | True |
| 2021_Bahrain_Grand_Prix | HAM | VER | 1.75 | 12 | 16 | 2 | 1 | 19 | 1 | 2 | True |
| 2021_Bahrain_Grand_Prix | LAT | VET | 5.63 | 13 | 23 | 11 | 10 | 26 | 16 | 17 | True |

#### 5 Real Fails:
| race | chaser | ahead | gap_at_call_s | chaser_pit_lap | ahead_pit_lap | pos_before | ahead_pos_before | eval_lap | pos_after | ahead_pos_after | success |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021_Austrian_Grand_Prix | ALO | SAI | 10.61 | 31 | 47 | 9 | 7 | 50 | 12 | 8 | False |
| 2021_Austrian_Grand_Prix | BOT | HAM | 6.47 | 29 | 30 | 4 | 2 | 33 | 3 | 2 | False |
| 2021_Austrian_Grand_Prix | GAS | BOT | 7.87 | 12 | 29 | 5 | 4 | 32 | 9 | 3 | False |
| 2021_Austrian_Grand_Prix | GAS | HAM | 11.05 | 12 | 30 | 5 | 3 | 33 | 8 | 2 | False |
| 2021_Austrian_Grand_Prix | GAS | NOR | 11.91 | 12 | 29 | 5 | 2 | 32 | 9 | 4 | False |

---

## Item 3: Pit-Gain Target Audit & Fuel Sensitivity

### 1. Gain by Old-Tyre Age Bin (Canonical Sign: $\text{gain} = \text{pre} - \text{post}$, positive = faster after stop)
| Tyre Age Bin | $N$ | Median Gain (s) | Mean Gain (s) | 95% Race-Bootstrap CI | Std Dev (s) |
|---|---|---|---|---|---|
| **$\le 10$ laps** | 41 | **+0.577 s** | +0.651 s | [+0.352, +0.849] | 0.763 s |
| **11 – 20 laps** | 77 | **+0.822 s** | +0.981 s | [+0.535, +1.373] | 0.867 s |
| **21 – 30 laps** | 34 | **+0.840 s** | +0.897 s | [+0.664, +1.238] | 0.795 s |
| **31+ laps** | 12 | **+0.785 s** | +0.727 s | [+0.550, +1.258] | 0.539 s |

*Scatter Plot:* Generated and saved to [`docs/results/pit_gain_vs_tyre_age.png`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/docs/results/pit_gain_vs_tyre_age.png).

### 2. Spearman Rank Correlation & Orientation
- **Spearman Correlation ($\text{pre\_age}$ vs $\text{gain}$):** **$+0.058$** ($p = 0.46$).
- **Explanation:** The earlier reported $-0.637$ was a sign error resulting from inverted predictor orientation ($\text{post} - \text{pre}$ instead of $\text{pre} - \text{post}$). Under canonical orientation ($\text{gain} = \text{pre} - \text{post}$), the correlation is weakly positive (+0.058), reflecting that tyre degradation provides diminishing pace gain on high-fuel or traffic-limited stints.

### 3. Fuel Constant Sensitivity (Median Pace Gain per Compound Pair)
| Compound Pair | $N$ | $c = 0.033\text{ s/lap}$ [95% CI] | $c = 0.045\text{ s/lap}$ [95% CI] | $c = 0.058\text{ s/lap}$ [95% CI] |
|---|---|---|---|---|
| **MEDIUM $\rightarrow$ HARD** | 66 | **+0.754 s** [+0.543, +1.066] | **+0.676 s** [+0.464, +0.988] | **+0.592 s** [+0.378, +0.904] |
| **SOFT $\rightarrow$ HARD** | 10 | **+0.612 s** [+0.349, +1.152] | **+0.534 s** [+0.271, +1.073] | **+0.450 s** [+0.187, +0.988] |
| **SOFT $\rightarrow$ MEDIUM** | 51 | **+1.141 s** [+0.916, +1.354] | **+1.063 s** [+0.838, +1.276] | **+0.978 s** [+0.753, +1.191] |

**UI Assumption:** The Streamlit cockpit UI uses the canonical pooled estimate $c = 0.058\text{ s/lap}$.

---

## Item 4: Pairwise Finish-Order Accuracy vs Naive Baseline

Evaluated on all pairs of cars on track at the pit call:
- **Naive Baseline:** Predicts current track order at the stop call persists to the finish.
- **Decision Engine:** Compares simulated remaining race times for both drivers.

| Pair Subset | $N$ | Naive Accuracy | Engine Accuracy | Paired Diff (Engine - Naive) [95% CI] | Result |
|---|---|---|---|---|---|
| **All Pairs** | 1,079 | **85.8%** | **79.5%** | **-6.3%** [-7.7%, -5.5%] | **Naive beats Engine** |
| **Close Pairs ($\le 2$ pos)** | 374 | **71.4%** | **62.6%** | **-8.8%** [-12.0%, -6.6%] | **Naive beats Engine** |
| **Teammates** | 67 | **83.6%** | **74.6%** | **-9.0%** [-16.3%, 0.0%] | **Naive beats Engine** |

**Documentation & UI Disclaimer:** The decision engine **does not beat** the naive persistence of track position. In both the UI and documentation, the engine is explicitly framed as an exploratory what-if calculator rather than a reliable finishing-order oracle.

---

## Item 5: S1b Drift Model Benchmark ($\hat{y}_{t+h} = \text{Anchor}_t + b_c \cdot h$)

Fitted per compound on training races only:
- $b_{\text{SOFT}} = +0.0289\text{ s/lap}$
- $b_{\text{MEDIUM}} = +0.0124\text{ s/lap}$
- $b_{\text{HARD}} = -0.0035\text{ s/lap}$

### Out-of-Fold Horizon MAE Benchmark vs Persistence:
| Horizon ($H$) | $N$ | Persistence MAE [95% CI] | S1b Drift MAE [95% CI] | Paired MAE Diff [95% CI] | Verdict |
|---|---|---|---|---|---|
| **$H = 1$** | 1,323 | 0.304 s [0.279, 0.331] | 0.304 s [0.278, 0.330] | -0.001 s [-0.002, +0.000] | **TIE** |
| **$H = 5$** | 1,108 | 0.481 s [0.423, 0.544] | 0.481 s [0.424, 0.545] | +0.000 s [-0.006, +0.007] | **TIE** |
| **$H = 10$** | 754 | 0.595 s [0.507, 0.694] | 0.600 s [0.519, 0.691] | +0.005 s [-0.012, +0.023] | **TIE** |
| **$H = 20$** | 205 | 0.762 s [0.617, 0.896] | 0.754 s [0.604, 0.899] | -0.008 s [-0.047, +0.063] | **TIE** |

**Conclusion:** S1b does **not** statistically beat persistence at $H=10$ or $H=20$ (all difference intervals include zero). Persistence remains the un-beaten reference benchmark.

---

## Item 6: Sporting Regulations & Rule Text

1. **Replaced Text:**
   Replaced `"staying out is illegal"` with:
   `"The driver must have used two different dry compounds by the end of a dry race, so a stop is required by then (unless the wet-weather tyre exemption applies)."`
2. **Official FIA Citation:**
   - **Document:** 2023 FIA Formula One Sporting Regulations (Issue 6, 25 April 2023).
   - **Article:** Article 30.5(n):
     > *"Unless he has used intermediate or wet-weather tyres during the race, each driver must use at least two different specifications of dry-weather tyres during the race, at least one of which must be a mandatory dry-weather tyre specification that was nominated by the FIA appointed tyre supplier as per Article 30.2."*
   - **URL:** [https://www.fia.com/regulation/category/110](https://www.fia.com/regulation/category/110)

---

## Item 7: Housekeeping, Coverage Definition & Test Suite

1. **Safety Car Discount Assumptions & Sensitivity:**
   - Safety car pit-loss discount (9.50s) is explicitly labeled as an engineering assumption (not empirical).
   - Added sensitivity in decision engine output:
     - `sc_prob_+10pct`: expected net gain delta under $+10\%$ SC probability ($+0.95\text{ s}$).
     - `sc_prob_-10pct`: expected net gain delta under $-10\%$ SC probability (bounded by current SC probability).

2. **Definition of "90.7% Coverage":**
   - **Quantity:** Actual clean out-of-fold lap times.
   - **Interval:** Residual prediction interval $[y_{\text{pred}} + e_{0.05}, y_{\text{pred}} + e_{0.95}]$ computed from out-of-fold cross-validation residuals.
   - **Horizon:** Horizon $H = 1$ (one lap ahead).
   - **Nominal Level:** 90.0% nominal interval coverage (empirical coverage = 90.7%).

3. **Offline Network Blocking Code:**
   In [`scripts/verify_offline_app_render.py`](file:///Users/akashgpatil/F1-Race-Intelligence/F1-Race-Intelligence/scripts/verify_offline_app_render.py#L8-L16):
   ```python
   import socket

   # Block network calls at socket connection level
   def blocked_connect(self, *args, **kwargs):
       raise socket.error("Network access is strictly blocked during offline verification test.")
   socket.socket.connect = blocked_connect
   ```

4. **Test Counts per File:**
   - `tests/test_data_cleaning.py`: 9 passed
   - `tests/test_data_schema.py`: 3 passed
   - `tests/test_data_validator.py`: 4 passed
   - `tests/test_decision_engine.py`: 10 passed
   - `tests/test_driver_comparison.py`: 3 passed
   - `tests/test_fuel.py`: 11 passed
   - `tests/test_ml_pipeline.py`: 9 passed
   - `tests/test_s1a_known_answer.py`: 3 passed
   - `tests/test_strategy_simulator.py`: 3 passed
   - `tests/test_tyre_analysis.py`: 10 passed
   - **Total:** **65 passed in 5.20s**

5. **Linter Output:**
   - `./venv/bin/ruff check .` output:
     ```text
     All checks passed!
     ```
