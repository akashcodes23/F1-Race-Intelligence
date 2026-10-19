# ⚠️ F1 Race Intelligence 3.0 — Known Limitations & Scientific Boundaries

This document provides a transparent, exhaustive catalog of known physical limitations, modeling assumptions, and domain boundaries in F1 Race Intelligence 3.0.

---

## 1. Physical & Environmental Confounders

### 1.1 Lack of Traffic & Wake Turbulence (Dirty Air) Modeling
- **Confounder:** Following another Formula 1 car within $1.5\text{ seconds}$ disrupts aerodynamic downforce, overheating the front tyres and inducing thermal degradation that is not intrinsic to tyre compound age.
- **Current Behavior:** The system filters caution laps and extreme outliers ($>3.0 \times \text{MAD}$), but does not model proximity wake deficits due to lack of high-frequency relative distance telemetry in the tabular timing data.
- **Guidance:** In-traffic stints may exhibit steeper degradation slopes than free-air stints.

### 1.2 Track Evolution (Rubbering-In)
- **Confounder:** As racing cars lay rubber on the racing line, track grip increases over the session duration, counteracting tyre degradation.
- **Current Behavior:** In low-wear races (e.g. Monaco, Monza), track grip evolution and fuel burn-off can outpace tyre degradation, causing raw lap times to fall even on aging tyres. The fuel correction module offsets mass burn-off, but track surface temperature and grip evolution are not explicitly modeled.

### 1.3 Weather Transitions & Wet Sessions
- **Boundary:** Tyre degradation regressions and fuel effect estimations are strictly restricted to dry tyre compounds (`SOFT`, `MEDIUM`, `HARD`).
- **Behavior:** Intermediate and Full Wet tyres exhibit radically non-linear degradation driven by water dispersal and surface drying. Stints on wet weather compounds are flagged as non-comparable.

---

## 2. Statistical & Machine Learning Limitations

### 2.1 Persistence Baseline Superiority on Stint Forensics
- **Negative Finding:** In multi-step stint forecasting ($H \ge 5$ laps), static Persistence ($\hat{y}_{t+h} = y_0$) consistently outperforms autoregressive gradient boosting (HGBR Variant A) and anchored physics regression ($S_1a$).
- **Explanation:** In Formula 1 racing, modern Pirelli tyres operate within a thermal management window where drivers deliberately manage pace to a target delta. Autoregressive models project compound wear aggressively forward, accumulating bias, whereas drivers dynamically adjust their pace.
- **Engineering Discipline:** The system retains Persistence as the primary benchmark standard and honestly reports the gate failure of candidate models.

### 2.2 Finish-Order Baseline Superiority
- **Negative Finding:** A naive baseline projecting that current track order persists to the checkered flag achieves **85.8%** pairwise accuracy across 1,079 pairs in held-out races, outperforming the Decision Engine's **79.5%**.
- **Explanation:** Formula 1 cars suffer severe overtaking difficulty (dirty air, track width, DRS train lock). Pitting for fresh tyres yields pace advantage, but recovering lost track position is constrained by track geometry and dirty air.
- **Role:** The Decision Engine is an exploratory what-if simulator, not a finish-order oracle.

### 2.3 Stint Length Selection Bias
- **Survival Bias:** Teams voluntarily pit cars when tyres show severe wear or thermal degradation. Consequently, tyres that run for 30+ laps are disproportionately those that did not experience rapid degradation. Regressing on surviving long stints underestimates true degradation rates.

---

## 3. Data & Visualization Boundaries

### 3.1 Reconstructed Telemetry in Offline Mode
- **Boundary:** To maintain a lightweight offline demo footprint ($<1\text{ MB}$), the offline dataset stores lap timing, sector splits, and speed traps (`SpeedST`, `SpeedFL`, `SpeedI1`, `SpeedI2`), but does NOT contain continuous millisecond time-series telemetry.
- **Reconstruction:** Offline telemetry traces are generated from circuit apex geometry and scaled to observed speed-trap maxima. Throttle, Brake, Gear, and RPM are derived from longitudinal acceleration gradients.
- **Integrity Rule:** Reconstructed telemetry is visibly badged as `[SIMULATED]` in the UI and must not be presented as physical onboard telemetry.

### 3.2 Approximate 3D Circuit Geometry
- **Boundary:** 3D circuit meshes are periodic cubic splines fitted through validated landmark waypoints and elevation profiles.
- **Precision:** Waypoints capture real elevation features (Monaco Casino climb, Red Bull Ring uphill to Turn 3), but are smoothed geometric representations rather than sub-centimeter LIDAR surveys.
- **Elevation Accentuation:** The 3D engine defaults to an elevation scale of $2.0\times$ to render dramatic vertical changes legible in web viewports.
