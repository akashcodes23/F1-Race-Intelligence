# Repository Audit: F1 Race Intelligence

**Audit Date:** 2026-10-16  
**Auditor:** Senior Motorsport Data Scientist & Python Systems Engineer  
**Scope:** `app.py`, `config.py`, `src/data_loader.py`, `src/metrics.py`, `src/visualization.py`, `requirements.txt`, `README.md`

---

## 1. Executive Summary

F1-Race-Intelligence started as a Streamlit visualization dashboard displaying retrospective FastF1 timing charts. While recent modularization introduced data validation, head-to-head deltas, per-stint linear regressions, high-frequency telemetry traces, and chronological ML splits, the application is fundamentally an **event visualizer**, not yet a **predictive Strategy Decision Engine**.

To support real-time pit wall decision-making, the analytical engine requires:
1. Rigorous fuel-corrected lap times ($~0.03\text{ s/lap}$ mass effect).
2. Systematic safety car, VSC, and track-status filtering (`TrackStatus` & `IsAccurate`).
3. Robust non-linear degradation estimation (Huber/Theil-Sen with tyre cliff change-point detection).
4. Circuit-specific pit-loss estimation derived from historical in/out laps.
5. Monte Carlo race simulation incorporating stochastic safety car risks and traffic penalties.

---

## 2. What Works

- **FastF1 Ingestion & Local HTTP/Pickle Cache:** Robust loading of events, sessions, timing data, and high-frequency telemetry (`Speed`, `Throttle`, `Brake`, `Distance`) with ~125MB SQLite cache.
- **Head-to-Head Delta Analysis:** Common lap alignment, lap-by-lap delta, and cumulative gap tracking with verified sign conventions.
- **Spatial Telemetry Normalization:** Distance-based interpolation (`np.interp`) onto a common 500-point distance grid for corner-by-corner overlay.
- **Circuit Layout Coordinates:** Extraction of $(X, Y)$ track coordinates for velocity heatmaps.
- **Dark Theme Interface:** Clean Streamlit UI with custom CSS and Plotly figure generation.
- **Basic Modular Architecture:** Layered modules in `src/` tested by a 15-test unit suite.

---

## 3. Analytical Flaws

| Category | Flaw Description | Motorsport Impact |
|---|---|---|
| **Fuel Correction** | Lap times are analyzed without fuel-mass correction. Cars burn ~100–110 kg of fuel across a Grand Prix (~0.03 to 0.06 s/lap pace gain). | Fuel burn directly counteracts tyre degradation. A flat stint lap-time profile actually masks significant tyre wear. |
| **Track Status & Yellow Flags** | Data cleaning does not filter using FastF1's `TrackStatus` flags (Safety Car, Virtual Safety Car, Red Flag, Double Yellows). | Laps under yellow flags or caution periods distort degradation regressions and inflate consistency metrics. |
| **Lap Accuracy (`IsAccurate`)** | `IsAccurate` boolean flag from FastF1 timing is not enforced. Out-laps, track limit violations, and off-track excursions contaminate the clean lap pool. | High variance in lap times compromises regression slope ($s/\text{lap}$) and statistical quantiles. |
| **Linear Degradation Assumption** | Degradation is fitted via standard OLS ($y = mx + c$). Pirelli tyres experience non-linear phases: graining, cleaning, thermal degradation, and a catastrophic thermal cliff. | Linear regression underpredicts rapid pace drops when tyres hit the cliff, leading to delayed pit stop calls. |
| **Pit Loss Estimation** | Strategy simulation uses hardcoded pit loss penalties (e.g. 22.0s) rather than calculating true circuit-specific pit transit loss from in-lap and out-lap telemetry deltas. | Pit window recommendations can be biased by several seconds depending on track pit-lane speed limits and layout. |
| **Deterministic Simulation** | Strategy simulator projects a single deterministic remaining race time without Monte Carlo variance, safety car probability, or traffic congestion rejoining penalties. | Real pit walls make decisions under uncertainty. A single point estimate provides false confidence. |
| **ML Generalization** | Regressor models are trained only on single-driver, single-race data without multi-race cross-validation, weather covariates, or gap-to-car-ahead features. | Overfits to individual race conditions and cannot generalize to prospective strategy planning. |

---

## 4. Bugs and Code Hygiene Issues

1. **Root Directory Hygiene:** Stray `__init__.py` files or empty directories at the workspace root confuse package resolution.
2. **Streamlit Syntax Deprecations:** `use_container_width=True` is deprecated in modern Streamlit versions in favor of `width="stretch"`.
3. **FastF1 Deprecation Warnings:** `pick_driver()` is deprecated in FastF1 3.3+ in favor of `pick_drivers()`.
4. **Lack of Strict Data Schema:** Laps DataFrames pass between modules without type or schema validation, making functions vulnerable to missing FastF1 columns across historical seasons.

---

## 5. README Claims Not Backed by Code

- **Live Demo Link:** `HEAD:README.md` pointed to a broken placeholder badge `YOUR_STREAMLIT_LINK` rather than the deployed Streamlit Cloud URL.
- **Stray Git Headings:** `HEAD:README.md` contained dangling artifact strings like `## YOLO Trigger Clean`.
- **"Strategic Performance Engine" & "Pit Strategy Simulation":** The original code provided retrospective Plotly charts and a toy simulation loop that was disconnected from real tyre data and pit-entry constraints.

---

## 5.1 Current Accuracy Limits of ML Pipeline & Strategy Simulator (Phase 1 Baseline)

### `src/ml_pipeline.py` Accuracy Limits
1. **Feature Sparsity:** Currently utilizes only `LapNumber`, `TireAge`, and one-hot `Compound_*`. It omits fuel-mass load proxy (`LapsRemaining`), environmental variables (`TrackTemp`, `AirTemp`, `Humidity`), driver/constructor categorical effects, dirty air/traffic proxies (`GapToAhead`), and prior-lap momentum deltas.
2. **Within-Session Overfitting:** The 80/20 chronological split is restricted to laps within the single active session. It lacks Leave-One-Session-Out (LOSO) cross-validation to assess generalization across varied circuit layouts, track grip levels, and weather transitions.
3. **Model Class Limitations:** Ordinary Linear Regression cannot model non-linear thermal tyre cliffs, while the small single-session Random Forest (100 estimators, max_depth=6) struggles on small sample sizes (30–60 laps per driver). Phase 2 will transition to HistGradientBoosting / LightGBM with LOSO validation.
4. **Deterministic Point Predictions:** Lacks prediction intervals and residual-based error bands.

### `src/strategy_simulator.py` Accuracy Limits
1. **Deterministic Projection:** Computes a single fixed outcome over the horizon without variance or Monte Carlo sampling.
2. **Static Pit Loss:** Relies on hardcoded pit loss inputs (default 22.0s) rather than empirical circuit-specific pit-lane transit times derived from historical in-lap and out-lap timing.
3. **Absence of Stochastic Hazards:** Does not model safety car (SC) or virtual safety car (VSC) probabilities, which fundamentally alter real-world pit wall strategy windows.
4. **Zero Traffic Modeling:** Assumes a pit exit drops the driver into clean air with unrestricted pace, ignoring traffic rejoining windows, blue flag delta delays, and DRS trains. Phase 3 will introduce 1,000-run Monte Carlo simulations with stochastic hazards and traffic penalties.

---

## 6. Implementation Action Plan

- **Phase 0:** Hygiene cleanup, pinned requirements, CI/CD GitHub Actions, and `src/data/` schema-validated lap ingestion.
- **Phase 1:** Safety car / `TrackStatus` filtering, fuel burn-off correction ($~0.03\text{ s/lap}$), Huber/Theil-Sen robust degradation, and tyre cliff detector.
- **Phase 2:** HistGradientBoosting / LightGBM lap-time prediction model with Leave-One-Session-Out cross-validation.
- **Phase 3:** Empirical circuit pit-loss estimation and Monte Carlo strategy engine with safety car probabilities and undercut/overcut calculators.
- **Phase 4:** Interactive Pit Wall Decision Cockpit UI with lap scrubber and recommendation action cards.
- **Phase 5:** Backtesting framework on 5+ historical races comparing engine recommendations against actual pit stops.
- **Phase 6:** Structured AI race engineer briefing grounded in JSON telemetry metrics.
- **Phase 7:** Production Docker container, documentation, and release polish.
