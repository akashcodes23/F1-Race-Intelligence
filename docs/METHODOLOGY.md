# 📐 F1 Race Intelligence 3.0 — Mathematical & Analytical Methodology

## 1. Fuel Mass Burn-Off & Normalization Methodology

### 1.1 Physical Principles & Sign Conventions
During a Formula 1 Grand Prix, an F1 car starts with approximately 100–110 kg of fuel. Fuel consumption is roughly linear over race distance. As fuel burns off, vehicle mass decreases, yielding an empirical lap time improvement of $0.02 - 0.06\text{ s/lap}$.

- **Canonical Sign Definition:**
  $$\text{Lap Time Falls as Fuel Burns: } \frac{\partial T}{\partial L} < 0$$
  Let $c \ge 0$ denote the fuel burn rate in seconds saved per lap (empirically calibrated across dry races as $c \approx 0.058\text{ s/lap}$).

### 1.2 Normalization to Race-Start Fuel Mass
To isolate pure tyre degradation from fuel burn-off, lap times are normalized to reference lap $L_{ref} = 1$:
$$T_{\text{fuel-corrected}}(L) = T_{\text{observed}}(L) + c \cdot (L - L_{\text{ref}})$$
- As lap number $L$ increases, fuel has burned off, making the car lighter.
- Adding $c \cdot (L - 1)$ adds back the fuel-burn advantage, normalizing the car to full fuel load.
- Any remaining pace drift is attributable strictly to tyre degradation, traffic, or track evolution.

### 1.3 Empirical Stint-Transition Calibration
Fuel effect is empirically estimated by comparing early tyre-life baseline pace (laps 2–5) across compound-matched, dry consecutive stints:
$$\hat{c} = \frac{\bar{T}_{\text{stint } i, \text{laps 2-5}} - \bar{T}_{\text{stint } i+1, \text{laps 2-5}}}{L_{\text{start}, i+1} - L_{\text{start}, i}}$$
- Clamped strictly to physical bounds: $c \in [0.02, 0.08]\text{ s/lap}$.
- When insufficient dry compound-matched pairs exist, the system safely falls back to the physics default $c = 0.058\text{ s/lap}$ and logs an explicit warning.

---

## 2. Robust Tyre Degradation Modeling

### 2.1 Huber Loss Stint Regression
Ordinary Least Squares (OLS) is highly sensitive to traffic, yellow flags, and lock-ups. The platform fits stint degradation using Huber $M$-estimation:
$$\min_{\beta_0, \beta_1} \sum_{i=1}^N \rho_\delta\left(T_{\text{corr}, i} - (\beta_0 + \beta_1 \cdot \text{TyreLife}_i)\right)$$
where Huber loss $\rho_\delta(r)$ is defined as:
$$\rho_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{for } |r| \le \delta \\ \delta |r| - \frac{1}{2} \delta^2 & \text{for } |r| > \delta \end{cases}$$
with tuning constant $\delta = 1.345 \cdot \text{MAD}$ for 95% asymptotic efficiency on normal residuals.

### 2.2 Driver Consistency Index (Scaled MAD)
Driver lap pace variability is evaluated using the Median Absolute Deviation (MAD), scaled by $1.4826$ to yield an asymptotically unbiased estimator of standard deviation for normal distributions:
$$\text{Consistency Index} = 1.4826 \cdot \text{median}\left(\left|T_{\text{corr}, i} - \text{median}(T_{\text{corr}})\right|\right)$$
- **Interpretation:** Lower values denote higher lap-to-lap pace repeatability.

---

## 3. S1a Anchored Physics Pace Model

### 3.1 Model Formulation
The $S_1a$ model projects future lap pace $h$ steps ahead from origin lap $t$:
$$\hat{y}_{t+h} = \text{AnchorPace}(t) + (\beta_{\text{compound}} - c) \cdot h$$
where:
1. $\text{AnchorPace}(t)$: Median fuel-corrected pace over the most recent 3 clean laps.
2. $\beta_{\text{compound}}$: Compound-specific linear degradation slope (s/lap) fitted out-of-fold.
3. $c$: Canonical fuel effect ($0.058\text{ s/lap}$).
4. Net Slope: $\beta_{\text{net}} = \beta_{\text{compound}} - c$.

### 3.2 Evaluation Gate Protocol
In multi-horizon benchmarks ($H=1$ to $H=20$), models are evaluated against static Persistence ($\hat{y}_{t+h} = y_t$). The gate requires that $S_1a$ achieve statistically significant improvement over Persistence. (See `docs/MODEL_VALIDATION.md` for gate evaluation results).

---

## 4. Difference-Based Monte Carlo Decision Engine

### 4.1 Causal Stint Formulation
At origin lap $L_{\text{current}}$, the decision engine evaluates two competing tactical actions across the remaining race horizon $H = L_{\text{total}} - L_{\text{current}}$:
1. **Plan A (Stay Out):** Continue on current tyres until $L_{\text{total}}$.
2. **Plan B (Pit Now):** Pit on lap $L_{\text{current}}$, incur circuit pit loss $T_{\text{pit}}$, and race to $L_{\text{total}}$ on fresh compound.

### 4.2 Difference Statistic
Rather than estimating absolute race times (which introduces correlated variance), the engine samples directly from the **difference distribution**:
$$\Delta T = T_{\text{stay}} - T_{\text{pit}}$$
$$\Delta T = \sum_{h=1}^H T_{\text{stay}}(h) - \left[ T_{\text{pit-loss}} + \sum_{h=1}^H T_{\text{fresh}}(h) \right]$$
- **Positive $\Delta T > 0$:** Pitting saves cumulative race time.
- **Negative $\Delta T < 0$:** Staying out yields faster race completion.

### 4.3 Probabilistic Verdict Rules & Strict Inconclusive Gate
Across $N = 1,000$ Monte Carlo simulations:
1. $P(\text{Pit Beats Stay}) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\Delta T_i > 0)$
2. Compute 80% credible interval $[Q_{0.10}(\Delta T), Q_{0.90}(\Delta T)]$.
3. **Strict Inconclusive Rule:**
   - If the 80% credible interval spans zero ($Q_{0.10} \le 0 \le Q_{0.90}$), the verdict is forced to **`INCONCLUSIVE`**, prohibiting speculative pit recommendations.
   - If $Q_{0.10} > 0$ and $P > 0.60$: **`PIT NOW`**.
   - If $Q_{0.90} < 0$ and $P < 0.40$: **`STAY OUT`**.

---

## 5. Tactical Undercut / Overcut Physics

### 5.1 Out-Lap Delta Formulation
The undercut margin $\Delta_{\text{out}}$ evaluates whether pitting before a direct rival allows the chasing driver to emerge ahead after the rival stops:
$$\Delta_{\text{out}} = \Delta_{\text{tyre pace}} - \Delta_{\text{in-out loss}} - \text{TrackGap}$$
where $\Delta_{\text{tyre pace}}$ is the pace differential of fresh compound vs worn rival compound over the out-lap.

### 5.2 Empirical Base Rates
Empirical evaluation across 114 held-out undercut scenarios yields:
- $\text{Gap} < 1.0\text{ s}$: **64.7%** success rate ($N=17$)
- $\text{Gap } 1.0 - 2.0\text{ s}$: **67.7%** success rate ($N=31$)
- $\text{Gap } 2.0 - 3.0\text{ s}$: **29.0%** success rate ($N=31$)
- $\text{Gap } 3.0 - 5.0\text{ s}$: **20.0%** success rate ($N=35$)
