# F1 Race Intelligence — Machine Learning Lap Pace Model & Phase 2.7 Validation

## 1. Executive Summary & Objective

In Formula 1 race engineering, predicting lap times across disparate circuits using raw second regression is fundamentally flawed. Lap times range from ~65 seconds at the Red Bull Ring to ~105 seconds at Spa-Francorchamps; a global regression on raw lap times inevitably memorizes circuit intercept constants rather than physical tyre degradation, fuel burn-off, and compound deltas.

Phase 2.7 establishes the formal validation of the **Anchored Physics Model ($S_1a$)** as the candidate for Phase 3's Strategy Decision Engine, resolves all open items from Phase 2.6, and provides definitive empirical benchmarks under Leave-One-Race-Out (LORO) cross-validation across 13 Grand Prix events (10,637 clean dry laps evaluated strictly on Laps $\ge 7$) spanning 2021–2023.

---

## 2. Target Formulation & Zero-Leakage Architecture

### Mathematical Target
For driver $d$ on lap $i$ of race $r$:
$$\text{RelativeLapTimeSec}_{d,i} = \text{LapTimeSec}_{d,i} - \text{DriverReferencePaceSec}_{d,r}$$

Where $\text{DriverReferencePaceSec}_{d,r}$ is defined as:
$$\text{DriverReferencePaceSec}_{d,r} = \text{median}\left(\{\text{LapTimeSec}_{d,k} \mid k \in \text{CleanLaps}_{d,r},\, 2 \le k \le 6\}\right)$$
*(with extreme outlier/damaged laps exceeding $1.15 \times \text{SessionMedian}$ excluded)*.

### Definition of Signed Error
Throughout the codebase, tests, and documentation, **signed error** is strictly defined as:
$$\text{Signed Error} = \hat{y} - y = \text{Predicted Value} - \text{Actual Value}$$
A positive signed error indicates the model predicted a lap slower than actual; a negative signed error indicates the model predicted a lap faster than actual.

### Zero-Leakage Guarantees
1. **Reference Window Isolation:** Reference pace is computed **strictly from Laps 2–6**.
2. **Evaluation & Training Boundary:** Both training and evaluation are conducted **strictly on Laps $\ge 7$**. Laps 1–6 are never scored or used as training targets.
3. **Strict Feature Lagging:** `PrevLapRelativePace` is strictly lagged from lap $t-1$:
   $$\text{PrevLapRelativePace}_t = \Delta t_{t-1}$$
   For the first evaluated lap (Lap 7), it uses Lap 6 relative pace. Across stint transitions, it resets to $0.0\text{ s}$ to prevent leaking prior stint tyre state.
4. **Dual Perturbation Unit Tests:** Verified by `tests/test_ml_pipeline.py`:
   - Perturbing Lap 4 inside the reference window alters reference pace and shifts relative targets by an exact offset, while feature rows $X_1 \dots X_4$ remain causal.
   - Perturbing Lap 7 strictly alters $X_8$ (`PrevLapRelativePace`) and target $y_7$, with zero effect on reference pace or prior rows $X_1 \dots X_7$.

---

## 3. Models Under Benchmark

We evaluate predictive models across the exact same 13-fold LORO cross-validation splits:

1. **Persistence Baseline ($\hat{y}_{t+h} = y_0$):**
   Projects the origin lap's relative pace into the future.
2. **Huber Extrapolation (Prior Stint Laps):**
   Fits a robust Huber regressor on the driver's current stint using only laps prior to the origin ($k < t_0$), with minimum 6 prior laps required.
3. **$S_1a$ Anchored Physics Model (Candidate for Phase 3):**
   Predicts increments from the driver's current anchored pace:
   - **Current Stint Projection ($h$ laps ahead):**
     $$\hat{y}(t_0 + h) = \text{anchor} + \beta_{\text{fuel}} \cdot h + \left[\text{deg}(c, \text{age}_0 + h) - \text{deg}(c, \text{age}_0)\right]$$
   - **New Stint on Compound $c_2$ After Pit Stop:**
     $$\hat{y}_{\text{new}}(t_{\text{stop}} + h) = \text{anchor} + \text{compound\_offset}(c_2 \text{ vs } c_{\text{curr}}) + \beta_{\text{fuel}} \cdot h + \text{deg}(c_2, h)$$
   - **Anchor Definition:** Median fuel- and tyre-corrected relative pace of the last 5 clean laps before origin:
     $$\text{anchor} = \text{median}_{k \in [t_0-4, t_0]}\left(y_k - \beta_{\text{fuel}} \cdot (k - t_0) - \text{deg}(c, \text{age}_k)\right)$$
   - **Fitted Parameters (Monotone Quadratic Degradation on Training Folds):**
     $$\text{deg}(c, \text{age}) = a_{1,c} \cdot \text{age} + a_{2,c} \cdot \text{age}^2 \quad (a_{1,c} \ge 0, a_{2,c} \ge 0)$$
4. **HGBR Variant A (Autoregressive):**
   Gradient boosted trees with `PrevLapRelativePace` (evaluated recursively for $H > 1$).
5. **HGBR Variant B (Static Origin):**
   Gradient boosted trees using only features known at stint origin (`TyreLife`, `Compound`, `LapsRemaining`, `LapNumber`).

---

## 4. Fitted $S_1a$ Parameters

Fitted across the full 13-race dry dataset using constrained non-negative least squares:
- **Fuel Effect:** $\beta_{\text{fuel}} = -0.058\text{ s/lap}$ (pooled estimate: $[-0.070, -0.046]$ s/lap).
- **Compound Offsets (Relative to MEDIUM):**
  - **SOFT:** **-1.4067 s** (faster baseline grip)
  - **MEDIUM:** **0.0000 s** (reference)
  - **HARD:** **+0.0342 s** (slightly slower baseline grip)
- **Monotone Degradation Curves $\text{deg}(c, \text{age}) = a_1 \cdot \text{age} + a_2 \cdot \text{age}^2$:**
  - **SOFT:** $a_1 = 0.13102$, $a_2 = 0.0$ ($\text{deg}(10) = +1.310\text{ s}$, $\text{deg}(20) = +2.620\text{ s}$)
  - **MEDIUM:** $a_1 = 0.04842$, $a_2 = 0.0$ ($\text{deg}(10) = +0.484\text{ s}$, $\text{deg}(20) = +0.968\text{ s}$)
  - **HARD:** $a_1 = 0.02171$, $a_2 = 0.0$ ($\text{deg}(10) = +0.217\text{ s}$, $\text{deg}(20) = +0.434\text{ s}$)

---

## 5. Horizon Evaluation across Multiple Stint Origins ($H \in \{1, 5, 10, 20\}$)

### Notice: Phase 2.5 Horizon Table Superseded
> **Important Note on Phase 2.5 vs Phase 2.6/2.7 Tables:**
> The preliminary horizon table presented in Phase 2.5 is **superseded**. In Phase 2.5, Variant B was evaluated on a small single-origin subset of long stints where Lap 7 relative pace was already established, yielding a deceptively low MAE of 0.586 s at $H=1$. In Phase 2.6 and 2.7, evaluation is conducted systematically across **all eligible stints from multiple per-origin cells** ($\text{TyreLife} \in [2, 4]$, $[7, 9]$, and $[14, 16]$) across all 13 races. Because static Variant B has no origin anchor, it carries an unconditional cross-race intercept offset, moving its $H=1$ MAE to 0.793–1.153 s.

### Horizon Evaluation Results Table (1,000 Bootstrap Resamples Over Races)

| Horizon ($H$) | Model | $N$ Evaluated | MAE (s) [95% CI] | Signed Bias (s) [95% CI] | Stint-Total Abs Error (s) [95% CI] |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$H=1$** | **Persistence** | 1,341 | **0.351** [0.317, 0.391] | -0.034 [-0.071, +0.002] | **0.351** [0.317, 0.391] |
| | Huber Extrapolation | 657 | 0.379 [0.332, 0.446] | -0.002 [-0.045, +0.038] | 0.379 [0.332, 0.446] |
| | **$S_1a$ (Anchored)** | 1,341 | 0.426 [0.387, 0.473] | +0.198 [+0.147, +0.250] | 0.426 [0.387, 0.473] |
| | Variant B (Static) | 1,341 | 0.793 [0.699, 0.907] | -0.016 [-0.178, +0.155] | 0.793 [0.699, 0.907] |
| | Variant A (Recursive) | 1,341 | 0.360 [0.322, 0.399] | -0.012 [-0.043, +0.020] | 0.360 [0.322, 0.399] |
| **$H=5$** | **Persistence** | 1,150 | **0.527** [0.458, 0.604] | -0.078 [-0.154, +0.001] | **1.887** [1.658, 2.107] |
| | Huber Extrapolation | 514 | 0.708 [0.613, 0.849] | +0.117 [-0.022, +0.263] | 2.334 [2.080, 2.713] |
| | **$S_1a$ (Anchored)** | 1,150 | 0.759 [0.665, 0.870] | +0.586 [+0.485, +0.686] | 2.583 [2.272, 2.957] |
| | Variant B (Static) | 1,150 | 0.843 [0.715, 0.975] | +0.000 [-0.169, +0.171] | 3.765 [3.136, 4.426] |
| | Variant A (Recursive) | 1,150 | 0.561 [0.485, 0.639] | +0.046 [-0.026, +0.125] | 1.985 [1.749, 2.246] |
| **$H=10$** | **Persistence** | 852 | **0.606** [0.523, 0.697] | -0.075 [-0.165, +0.016] | **4.272** [3.737, 4.911] |
| | Huber Extrapolation | 342 | 1.134 [0.989, 1.344] | +0.327 [+0.082, +0.597] | 7.017 [6.042, 8.488] |
| | **$S_1a$ (Anchored)** | 852 | 1.168 [1.007, 1.359] | +1.072 [+0.887, +1.272] | **7.152** [6.203, 8.218] |
| | Variant B (Static) | 852 | 0.972 [0.787, 1.161] | +0.000 [-0.187, +0.198] | 7.844 [6.092, 9.462] |
| | Variant A (Recursive) | 852 | 0.706 [0.568, 0.841] | +0.106 [-0.035, +0.252] | 4.677 [3.739, 5.625] |
| **$H=20$** | **Persistence** | 284 | **0.782** [0.672, 0.930] | +0.134 [-0.063, +0.354] | **9.510** [8.203, 11.375] |
| | Huber Extrapolation | 86 | 2.193 [1.613, 3.057] | +1.176 [+0.395, +2.181] | 25.543 [18.959, 35.479] |
| | **$S_1a$ (Anchored)** | 284 | 2.133 [1.802, 2.389] | +2.089 [+1.758, +2.378] | **23.163** [19.498, 26.059] |
| | Variant B (Static) | 284 | 1.350 [0.855, 1.931] | +0.300 [-0.250, +0.940] | 19.638 [13.122, 26.799] |
| | Variant A (Recursive) | 284 | 0.989 [0.680, 1.279] | +0.368 [+0.035, +0.722] | 12.686 [8.592, 16.318] |

---

## 6. Primary Metric: Pit Pace-Gain Accuracy

For every actual clean dry pit stop in held-out races ($N=331$ pit stops with no caution within $\pm 3$ laps), we predict:
$$\Delta_{\text{pace}} = \overline{\text{Pace}}_{\text{post}} - \overline{\text{Pace}}_{\text{pre}}$$
where $\overline{\text{Pace}}_{\text{post}}$ is the mean of 5 laps after out-lap on the new tyre, and $\overline{\text{Pace}}_{\text{pre}}$ is the mean of 5 laps before the in-lap, fuel-adjusted to the pit stop lap.

### Aggregate Performance ($N = 331$ stops)
- **Naive Baseline ($\Delta = 0$):** MAE = **1.792 s**, Bias = +1.780 s, Spearman $\rho = 0.0$
- **Huber Baseline:** MAE = **1.600 s**, Bias = +1.385 s, Spearman $\rho = \mathbf{0.456}$
- **$S_1a$ Anchored Model:** MAE = **1.213 s**, Bias = +1.145 s, Spearman $\rho = \mathbf{0.057}$
- **Improvement:** $S_1a$ beats Naive baseline MAE by **0.579 s (32.3% error reduction)**.

### Breakdown by Compound Transition Pair

| Compound Transition | Count ($N$) | Actual $\Delta$ (s) | Naive MAE (s) | Huber MAE (s) | $S_1a$ MAE (s) | $S_1a$ Bias (s) | $S_1a$ Rank Corr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MEDIUM $\to$ HARD** | 109 | -1.531 | 1.535 | 1.418 | **0.935** | +0.742 | +0.222 |
| **SOFT $\to$ MEDIUM** | 52 | -1.945 | 1.945 | 1.839 | **1.443** | +1.386 | +0.138 |
| **MEDIUM $\to$ SOFT** | 46 | -2.108 | 2.108 | 1.637 | **0.982** | +0.867 | +0.338 |
| **HARD $\to$ MEDIUM** | 30 | -1.978 | 1.978 | 1.666 | **1.838** | +1.777 | -0.191 |
| **HARD $\to$ HARD** | 24 | -1.522 | 1.522 | 1.701 | **1.237** | +1.037 | -0.155 |
| **MEDIUM $\to$ MEDIUM** | 22 | -1.933 | 1.933 | 1.562 | **1.488** | +1.401 | +0.077 |
| **SOFT $\to$ HARD** | 19 | -2.593 | 2.593 | 2.052 | **1.428** | +1.328 | -0.172 |
| **SOFT $\to$ SOFT** | 18 | -1.956 | 1.956 | 1.543 | **1.353** | +1.284 | +0.169 |
| **HARD $\to$ SOFT** | 11 | -1.989 | 1.989 | 1.621 | **1.396** | +1.256 | -0.036 |

### Breakdown by Outgoing Tyre Age

| Outgoing Tyre Age | Count ($N$) | Actual $\Delta$ (s) | Naive MAE (s) | Huber MAE (s) | $S_1a$ MAE (s) | $S_1a$ Bias (s) | $S_1a$ Rank Corr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Young Tyres (Age < 15)** | 62 | -1.676 | 1.706 | 1.632 | **1.458** | +1.439 | +0.038 |
| **Mid-Life Tyres (Age 15–25)** | 210 | -1.861 | 1.861 | 1.616 | **1.265** | +1.189 | +0.046 |
| **Old Tyres (Age > 25)** | 59 | -1.629 | 1.636 | 1.509 | **0.771** | +0.681 | +0.125 |

---

## 7. Gate Decision Rule & Formal Status

### Gate Criteria
> $S_1a$ goes to Phase 3 only if:
> 1. It beats Persistence on stint-total error at $H=10$ and $H=20$ (say explicitly if CI includes zero).
> 2. Its pit pace-gain MAE beats the naive baseline ($\Delta = 0$).
> Otherwise, STOP and report for decision.

### Empirical Verdict
1. **Pit Pace-Gain MAE:** $S_1a$ achieves **1.213 s** vs Naive **1.792 s**. **PASSED (32.3% improvement).**
2. **Stint-Total Error vs Persistence:**
   - **At $H=10$:** Persistence achieves **4.272 s** [3.737, 4.911] vs $S_1a$ **7.152 s** [6.203, 8.218].
     - Difference $\Delta(S_1a - \text{Persistence}) = \mathbf{+2.880\text{ s}}$ (95% CI: `[+1.599, +4.076]`).
     - **Does the 95% CI include zero? NO.**
   - **At $H=20$:** Persistence achieves **9.510 s** [8.203, 11.375] vs $S_1a$ **23.163 s** [19.498, 26.059].
     - Difference $\Delta(S_1a - \text{Persistence}) = \mathbf{+13.653\text{ s}}$ (95% CI: `[+9.066, +17.008]`).
     - **Does the 95% CI include zero? NO.**

### Gate Outcome: FAILED
Because $S_1a$ carries an empirical positive degradation slope applied along the horizon while actual clean race laps exhibit pace maintenance (due to tyre management and fuel offsetting), $S_1a$ accumulates positive bias (+1.07s at H=10, +2.09s at H=20), failing to beat Persistence.

Per instructions, **execution is STOPPED at Phase 2.7 pending user decision**.

---

## 8. Honest Claims & Disclosures

1. **Multi-Step Scoring:** In multi-step autoregressive forecasting ($H \ge 5$), HGBR Variant A degrades rapidly due to error compounding and performs worse than Persistence. Static baselines ($B$, $S_1$, $S_2$) carry an unconditional cross-race intercept offset of ~0.6–0.9 s/lap.
2. **Borderline One-Step Significance:** The original Phase 2 claim that "HGBR beats persistence" holds **only** for 1-step all-laps scoring, and the statistical margin is borderline:
   $$\text{Paired } t = -1.995, \quad p = 0.0693 \quad (\text{not significant at } \alpha = 0.05)$$
3. **Collinearity Resolution for Variant B:** The rejection of Variant B in Phase 2.6 on grounds of "implausible fuel partial-dependence" was misattributed. In race conditions, $\text{LapNumber} + \text{LapsRemaining} \approx \text{TotalLaps}$. Sweeping `LapsRemaining` while holding `LapNumber` constant creates an unphysical counterfactual off the data manifold.
4. **Predicted Pace vs Tyre Age and Compound at Fixed Lap (Lap 25):**

| Compound | Tyre Age | $S_1a$ Predicted Rel Pace (s) | Variant B Predicted Rel Pace (s) | Difference ($S_1a - B$) |
| :--- | :---: | :---: | :---: | :---: |
| **SOFT** | 5 | -0.752 | -1.502 | +0.750 s |
| | 15 | +0.559 | -0.176 | +0.735 s |
| | 25 | +1.869 | -0.059 | +1.928 s |
| **MEDIUM** | 5 | +0.242 | -1.260 | +1.502 s |
| | 15 | +0.726 | -0.269 | +0.995 s |
| | 25 | +1.211 | -0.188 | +1.399 s |
| **HARD** | 5 | +0.143 | -1.157 | +1.300 s |
| | 15 | +0.360 | -0.218 | +0.578 s |
| | 25 | +0.577 | -0.113 | +0.690 s |

---

## 9. Recalibrated Tyre Cliff Detector

Under autocorrelated AR(1) lap time noise ($\phi = 0.40$), uncalibrated change-point heuristics produce excessive false positives ($>40\%$). 

We recalibrated the detector by enforcing:
1. Knot search constrained to $\text{TyreAge} \ge 10$, $\text{laps\_pre} \ge 6$, $\text{laps\_post} \ge 5$.
2. Meaningful slope acceleration: $\text{slope}_{\text{post}} \ge 2.0 \times \text{slope}_{\text{pre}}$ and $(\text{slope}_{\text{post}} - \text{slope}_{\text{pre}}) \ge 0.08\text{ s/lap}$.
3. Minimum sum of squared error reduction $\ge 30\%$.
4. Formal F-test vs single linear fit: $F \ge 10.0$ ($p < 0.005$).

### 1,000-Simulation Monte Carlo Benchmark (with Wilson 95% CIs)

| Noise Level ($\sigma$) | False Positive Rate | 95% Wilson CI (FP) | Power (+0.10 s/lap Cliff) | 95% Wilson CI (Power) |
| :---: | :---: | :---: | :---: | :---: |
| $\sigma = 0.20\text{ s}$ | **0.8%** | [0.4%, 1.6%] | **88.5%** | [86.4%, 90.3%] |
| $\sigma = 0.35\text{ s}$ | **2.3%** | [1.5%, 3.4%] | **45.9%** | [42.8%, 49.0%] |
| $\sigma = 0.50\text{ s}$ | **3.8%** | [2.8%, 5.2%] | **28.1%** | [25.4%, 31.0%] |

**Conclusion:** At the target benchmark noise of $\sigma = 0.35\text{ s}$, the recalibrated detector achieves $\text{FP} = 2.3\% \le 5.0\%$, satisfying the reliability threshold.

---

## 10. Monaco 2023 Reconciliation & Provenance Audit

### Commit Timeline & Fitting Logic
1. **Phase 1 Implementation:**
   - Commit `abda3d4` (merged via `99009df` / `8c919cf` ancestors).
   - Fitting Function: Ordinary Least Squares (`np.polyfit(deg=1)`) on untruncated Laps 2–53, using config fuel default $0.033\text{ s/lap}$.
   - Printed Output: VER Stint 1 = **+0.0660 s/lap**, ALO Stint 1 = **+0.0647 s/lap**.
2. **Phase 1.5 Implementation:**
   - Commit `b7a1272` (and refactored `d8cc29d`).
   - Fitting Function: Robust Huber Regressor with rain truncation (VER to Lap 51, ALO to Lap 52), using config fuel $0.058\text{ s/lap}$.
   - Printed Output: VER Stint 1 = **+0.0534 s/lap**, ALO Stint 1 = **+0.0446 s/lap**.

### Alonso Truncation Lap: 50 vs 52
- **Lap 51 Conditions:** In the 2023 Monaco GP, rain hit the track on Lap 51 (Turn 6 hairpin and Portier). 
- **Telemetry Reality:** Alonso pitted on Lap 54. On Lap 51, Alonso posted a clean $1:16.890$; on Lap 52, dampness caused a minor pace fluctuation ($1:18.150$); on Lap 53, heavy rain caused lap times to spike to $1:25.0+$.
- Truncating at **Lap 50** is the strictest dry-only boundary; truncating at **Lap 52** includes the greasy transitional laps before pit entry.

### Provenance of Phase 1 "Reference" Slopes (+0.0534, +0.0390, +0.0410)
An exhaustive search of the git commit history confirmed that the figures Soft (+0.0534), Medium (+0.0390), Hard (+0.0410) were manually transcribed reference values and possess **no algorithmic data provenance** in the codebase. Consequently, they have been removed from benchmark comparisons.
