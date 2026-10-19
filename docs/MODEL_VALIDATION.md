# 🔬 F1 Race Intelligence 3.0 — Model Validation & Scientific Benchmark Report

## 1. Executive Summary & Honest Scientific Disclosures

In accordance with rigorous empirical standards, this report documents all model validations, benchmark comparisons, and negative results across the F1-Race-Intelligence evaluation suite.

### Primary Disclosed Findings:
1. **Persistence is the Unbeaten Multi-Step Pace Benchmark:** For multi-step stint forecasts ($H \ge 5$ laps), simple Persistence of the origin pace ($\hat{y}_{t+h} = y_t$) outperforms recursive machine learning models (HGBR Variant A) and anchored physics models ($S_1a$).
2. **Track Position Persistence Beats Decision Engine on Finishing Order:** A naive baseline predicting that current track order persists to the checkered flag achieves **85.8%** pairwise accuracy across 1,079 pairs in held-out races, whereas the Decision Engine achieves **79.5%** (-6.3% gap, 95% CI: `[-7.7%, -5.5%]`).
3. **The Decision Engine is an Exploratory Scenario Tool:** The engine is validated as a tactical what-if simulator and sensitivity calculator, not as an unconstrained finishing-order oracle.

---

## 2. Leave-One-Race-Out (LORO) Cross-Validation Benchmark

### 2.1 Protocol
- **Dataset:** 13 Grand Prix events (2021–2023 seasons).
- **Sample Size:** 10,637 clean dry laps evaluated strictly on laps $\ge 7$ to avoid lap-1 and caution anomalies.
- **Evaluation:** Strict Leave-One-Race-Out (LORO). Models are fitted on 12 races and tested on the 13th held-out race.

### 2.2 Leaderboard ($H=1$ Next-Lap Forecast)

| Model Architecture | Features / Methodology | Mean MAE (s) | Mean RMSE (s) | Relative to Persistence |
|---|---|---|---|---|
| **$S_1$ Physics Linear** | Reference pace + fuel burn + compound degradation | 0.695 s | 0.957 s | +81.0% |
| **$S_2$ Scaled Ridge** | Ridge regression on static stint features | 0.718 s | 0.984 s | +87.0% |
| **Huber Per-Stint** | Stint-specific robust regression | 0.424 s | 0.664 s | +10.4% |
| **Persistence Baseline** | Flat projection of previous lap pace ($\hat{y} = y_0$) | **0.384 s** | **0.626 s** | **Reference (0.0%)** |
| **HGBR Variant B (Static)** | Gradient boosting on static features (no lagged pace) | 0.776 s | 1.042 s | +102.1% |
| **HGBR Variant A (Autoregressive)**| Gradient boosting with lagged pace feature | 0.367 s | 0.573 s | -4.4% (Borderline, $p=0.0693$) |

> **Statistical Note on Variant A:** While Variant A achieves a 0.017s lower MAE at $H=1$, a paired $t$-test across the 13 held-out folds yields $t = -1.995, p = 0.0693$, failing conventional $\alpha = 0.05$ significance.

---

## 3. Multi-Step Stint Forecast Gate ($H = 1$ to $H = 20$)

When evaluating stint-total absolute cumulative error $\sum_{h=1}^H |e_h|$ across multi-origin horizons:

| Horizon ($H$) | Sample ($N$) | Persistence MAE | $S_1a$ Anchored MAE | Paired Difference ($S_1a$ - Pers) | 95% Bootstrap CI | Verdict |
|---|---|---|---|---|---|---|
| **$H = 1$** | 1,842 | **0.384 s** | 0.392 s | +0.008 s | `[-0.004, +0.021]` | Statistically Indistinguishable |
| **$H = 5$** | 1,420 | **1.842 s** | 2.310 s | +0.468 s | `[+0.210, +0.725]` | Persistence Outperforms $S_1a$ |
| **$H = 10$** | 754 | **4.272 s** | 7.152 s | +2.880 s | `[+1.599, +4.076]` | Persistence Outperforms $S_1a$ |
| **$H = 20$** | 205 | **9.510 s** | 23.163 s | +13.653 s | `[+9.066, +17.008]` | Persistence Outperforms $S_1a$ |

**Gate Conclusion:** Because the 95% confidence interval for long-horizon stint-total error is strictly positive, $S_1a$ failed the promotion gate. Persistence remains the active production benchmark.

---

## 4. Pairwise Finishing Order Accuracy

Evaluated across 1,079 pairs at tactical pit stop origins in held-out races:

| Driver Pair Cohort | Sample ($N$) | Naive Baseline (Track Persistence) | Decision Engine Accuracy | Difference [95% Paired CI] | Verdict |
|---|---|---|---|---|---|
| **All Competing Pairs** | 1,079 | **85.8%** | 79.5% | -6.3% `[-7.7%, -5.5%]` | Naive Baseline Wins |
| **Close Pairs ($\le 2$ pos)** | 374 | **71.4%** | 62.6% | -8.8% `[-12.0%, -6.6%]` | Naive Baseline Wins |
| **Teammate Pairs** | 67 | **83.6%** | 74.6% | -9.0% `[-16.3%, 0.0%]` | Naive Baseline Wins |

---

## 5. Stop-Window Hit Rate vs Equal-Width Baselines

Evaluated at causal origins $\Delta \in \{5, 10, 15\}$ laps prior to the real stop without future information:

| Window Half-Width ($k$) | Window Width ($2k+1$) | Decision Engine Hit Rate | Circuit Median Baseline | Stint Heuristic Baseline |
|---|---|---|---|---|
| **$k = 0$** | 1 lap | 2.9% | **3.5%** | **3.5%** |
| **$k = 1$** | 3 laps | 9.8% | **22.5%** | 10.4% |
| **$k = 2$** | 5 laps | 12.7% | **31.8%** | 16.8% |
| **$k = 3$** | 7 laps | 18.5% | **39.3%** | 22.5% |
| **$k = 4$** | 9 laps | 42.8% | **44.5%** | 31.2% |
| **$k = 5$** | 11 laps | 48.0% | **52.6%** | 39.3% |
| **$k = 8$** | 17 laps | 64.7% | **66.5%** | 53.8% |
| **$k = 10$** | 21 laps | **82.1%** | 72.8% | 62.4% |

**Insight:** At narrow tactical windows ($k \le 4$, width $\le 9$ laps), the historical Circuit Median baseline outperforms both the engine and simple heuristics, proving that track-specific race habits strongly govern stop windows. At wide windows ($k \ge 9$), the engine captures dynamic degradation trends.
