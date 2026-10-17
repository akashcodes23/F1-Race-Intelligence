# F1 Race Intelligence — Machine Learning Lap-Time Prediction Model (Phase 2)

## 1. Executive Summary & Objective

In Formula 1 race engineering, predicting lap times across different circuits with a single global regression model is fundamentally hindered by circuit scale variance: lap times range from ~65 seconds at the Red Bull Ring to ~105 seconds at Spa-Francorchamps. Attempting to regress raw lap times directly leads to models that memorize circuit intercept constants rather than learning tyre degradation, fuel burn-off, and compound deltas.

Phase 2 replaces raw second regression with a **relative lap pace prediction engine** evaluated strictly under **Leave-One-Race-Out (LORO)** cross-validation across 13 diverse Grand Prix events (11,563 clean dry laps) spanning 2021–2024.

---

## 2. Target Formulation & Justification

### Mathematical Definition
For driver $d$ on lap $i$ of race $r$:
$$\text{RelativeLapTimeSec}_{d,i} = \text{LapTimeSec}_{d,i} - \text{DriverReferencePaceSec}_{d,r}$$

Where $\text{DriverReferencePaceSec}_{d,r}$ is defined as:
$$\text{DriverReferencePaceSec}_{d,r} = \text{median}\left(\{\text{LapTimeSec}_{d,k} \mid k \in \text{CleanLaps}_{d,r},\, 2 \le k \le 6\}\right)$$

### Engineering Justification
1. **Circuit Invariance:** By subtracting the early-race reference pace, the target variable $\Delta t_{d,i}$ is normalized around $0.0 \text{ s}$ across all circuits, regardless of whether a baseline lap is 70 seconds or 100 seconds.
2. **Known at Prediction Time:** By Lap 6 of any Grand Prix, this reference pace is locked in. For subsequent laps (laps 7 through 70+), predictions are made strictly without lookahead bias.
3. **Driver & Car Benchmark Normalization:** The reference pace encapsulates the baseline car performance delta (e.g., Red Bull vs Williams) and engine fuel modes established early in the race, isolating tyre degradation and fuel consumption dynamics.

---

## 3. Causal Feature Architecture (Zero-Leakage Guarantee)

Every feature in the prediction matrix $X_t$ is strictly causal: it uses only telemetry and state information available **before** lap $t$ begins.

| Feature Name | Type | Description | Causal Guarantee |
| :--- | :--- | :--- | :--- |
| `TyreLife` | Integer | Laps completed on current tyre set | Tracked from pit-out lap count |
| `LapsRemaining` | Integer | Total expected race distance minus current lap | Fuel mass proxy ($m_{\text{fuel}} \propto N_{\text{rem}}$) |
| `LapNumber` | Integer | Race lap progression | Captures track rubbering & grip evolution |
| `PrevLapRelativePace` | Float | $\Delta t_{t-1}$ from immediately preceding lap | Strictly lagged from $t-1$; $0.0$ on stint in-lap |
| `Compound_SOFT` | Binary | One-hot indicator for Soft compound | Known at tyre fitting |
| `Compound_MEDIUM` | Binary | One-hot indicator for Medium compound | Known at tyre fitting |
| `Compound_HARD` | Binary | One-hot indicator for Hard compound | Known at tyre fitting |

### Strict Non-Leakage Validation
Our unit test suite (`tests/test_ml_pipeline.py::test_feature_causality_and_no_leakage`) proves that injecting severe pace perturbations (e.g., $+15.0\text{ s}$) at lap $t$ produces **zero change** in feature rows $X_1$ through $X_t$. The perturbation only registers at $X_{t+1}$ via the lagged `PrevLapRelativePace` input.

---

## 4. Models Under Evaluation

1. **Naive Reference Pace Baseline:**
   Predicts $\Delta t = 0.0\text{ s}$ for all laps (i.e. driver maintains their early reference pace throughout the entire race).
2. **Phase 1 Linear Model (Ridge Regression, $\alpha=1.0$):**
   A regularized linear formulation modeling linear tyre wear, linear fuel burn-off, and compound offset intercepts.
3. **HistGradientBoostingRegressor (HGBR):**
   Gradient boosted decision trees capable of modeling non-linear tyre wear regimes (early thermal degradation, linear plateau, and exponential tyre cliff transitions).
   - Parameters: `learning_rate=0.05`, `max_iter=150`, `max_depth=5`, `l2_regularization=1.0`.

---

## 5. Cross-Validation Methodology: Leave-One-Race-Out (LORO)

Random k-fold cross validation on time series race data causes severe data leakage: laps from the same race share identical weather, track rubbering, and race interruptions.

Instead, we employ **Leave-One-Race-Out (LORO)** cross-validation:
- For each fold, **12 Grand Prix events** are used for training.
- The model is tested on the **1 completely unseen held-out Grand Prix**.
- This process is repeated across all 13 races (11,563 laps total).
- Hyperparameters are fixed across folds; no tuning occurs on the held-out race.

---

## 6. Empirical Benchmark Results

### Overall Performance Across 13 Held-Out Races

| Model | Mean MAE (s) | Std MAE (s) | Mean RMSE (s) | Std RMSE (s) | vs Linear MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Reference Baseline** | 1.122 | ±0.326 | 1.446 | ±0.384 | — |
| **Linear Model (Ridge)** | 0.459 | ±0.096 | 0.768 | ±0.438 | Baseline |
| **HistGradientBoosting (HGBR)** | **0.372** | **±0.064** | **0.620** | **±0.256** | **-0.087s (-18.9%)** |

### Per-Race Held-Out Breakdown (LORO)

| Held-Out Grand Prix | Laps | Naive MAE (s) | Linear MAE (s) | HGBR MAE (s) | Linear RMSE (s) | HGBR RMSE (s) | HGBR Advantage (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2021 Austrian Grand Prix | 1,163 | 1.171 | 0.413 | 0.296 | 0.522 | 0.400 | **+0.117** |
| 2021 Bahrain Grand Prix | 826 | 1.326 | 0.389 | 0.346 | 0.559 | 0.493 | **+0.043** |
| 2021 British Grand Prix | 817 | 1.607 | 0.477 | 0.317 | 0.639 | 0.462 | **+0.160** |
| 2021 Italian Grand Prix | 703 | 0.607 | 0.328 | 0.299 | 0.456 | 0.480 | **+0.028** |
| 2021 Spanish Grand Prix | 1,036 | 1.288 | 0.472 | 0.423 | 0.674 | 0.629 | **+0.048** |
| 2021 United States Grand Prix | 881 | 1.601 | 0.532 | 0.444 | 0.733 | 0.638 | **+0.089** |
| 2022 Austrian Grand Prix | 1,132 | 0.752 | 0.480 | 0.393 | 0.632 | 0.539 | **+0.088** |
| 2022 Bahrain Grand Prix | 841 | 1.364 | 0.477 | 0.411 | 0.715 | 0.592 | **+0.066** |
| 2022 British Grand Prix | 637 | 1.118 | 0.481 | 0.365 | 2.111 | 1.111 | **+0.116** |
| 2022 French Grand Prix | 801 | 0.748 | 0.315 | 0.294 | 0.437 | 0.419 | **+0.021** |
| 2022 Italian Grand Prix | 737 | 0.640 | 0.379 | 0.326 | 0.493 | 0.455 | **+0.053** |
| 2022 Spanish Grand Prix | 1,061 | 1.140 | 0.530 | 0.410 | 0.738 | 0.572 | **+0.120** |
| 2023 Monaco Grand Prix | 928 | 1.220 | 0.694 | 0.513 | 1.276 | 1.272 | **+0.181** |

### Residual Diagnostics Breakdown

#### By Tyre Age Band
- **1–10 Laps (Warm-up & Traffic Phase):** Count = 4,594 \| Mean Absolute Error = **0.410s** (Median = 0.273s)
- **11–20 Laps (Linear Degradation Plateau):** Count = 4,349 \| Mean Absolute Error = **0.343s** (Median = 0.238s)
- **21–30 Laps (Late Stint):** Count = 1,768 \| Mean Absolute Error = **0.333s** (Median = 0.233s)
- **31+ Laps (Tyre Cliff Wear Phase):** Count = 562 \| Mean Absolute Error = **0.386s** (Median = 0.246s)

#### By Compound
- **HARD:** Count = 4,660 \| Mean Absolute Error = **0.368s** (Median = 0.255s)
- **MEDIUM:** Count = 4,681 \| Mean Absolute Error = **0.348s** (Median = 0.232s)
- **SOFT:** Count = 1,973 \| Mean Absolute Error = **0.436s** (Median = 0.284s)

*Note: Soft compound exhibits higher variance due to rapid thermal degradation and early degradation cliffs.*

---

## 7. Honest Evaluation & Conclusion

Does the machine learning model beat the linear baseline?
**Yes.** HistGradientBoosting demonstrates a consistent and statistically robust advantage over the linear baseline across all 13 held-out Grand Prix events:
1. **MAE Reduction:** Drops from **0.459s to 0.372s** (an **18.9% error reduction**).
2. **RMSE Reduction:** Drops from **0.768s to 0.620s** (a **19.3% error reduction**).
3. **Variance Reduction:** Standard deviation of MAE across circuits drops from **±0.096s to ±0.064s**, demonstrating superior cross-circuit generalization.
4. **Non-Linear Dynamics:** The decision trees successfully capture the plateau and steepening degradation slopes that a single linear slope inevitably under- or over-estimates.

---

## 8. Artifact & Reproduction

- **Model Artifact:** `models/lap_pace_model.joblib` (93.0 KB, compressed joblib, easily committable to git).
- **Dataset Generation:**
  ```bash
  python scripts/build_dataset.py --output data/race_laps_dataset.parquet
  ```
- **Model Training & LORO Evaluation:**
  ```python
  import pandas as pd
  from src.ml_pipeline import train_and_evaluate_models
  df = pd.read_parquet("data/race_laps_dataset.parquet")
  results = train_and_evaluate_models(df, save_artifact_path="models/lap_pace_model.joblib")
  ```
