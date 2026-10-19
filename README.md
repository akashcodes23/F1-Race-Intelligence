# 🏎️ F1 Race Intelligence 3.0 — Mission Control & 3D Spatial Cockpit
[![Live Demo](https://img.shields.io/badge/Live-Demo-red?style=for-the-badge)](https://f1-race-intelligence-qmzz4nuvmzql4hac6ajbbq.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)]()
[![Test Suite](https://img.shields.io/badge/Tests-70%20Passing-brightgreen?style=for-the-badge)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff%20Clean-00F5D4?style=for-the-badge)]()

**F1 Race Intelligence 3.0** is an enterprise-grade motorsport analytics, predictive strategy modeling, and 3D spatial telemetry platform designed to replicate real-world Formula 1 pit-wall race engineering workflows.

---

## 🎯 Platform Highlights

1. **🏎️ Mission Control Pit-Wall Cockpit**
   - **Interactive 60 FPS Lap Scrubber:** Synchronously advances 3D circuit positions, driver gaps, and strategy engine recommendations.
   - **Real-Time Strategy Matrix:** Evaluates `PIT NOW` vs `STAY OUT` using difference-based Monte Carlo simulations with strict `INCONCLUSIVE` verdicts when 80% credible intervals span zero.
   - **Undercut Attack Radar:** Evaluates rival gap thresholds ($0.0 - 5.0\text{ s}$) with empirical gap bucket base rates ($64.7\%$ for $<1\text{ s}$, $67.7\%$ for $1-2\text{ s}$, $29.0\%$ for $2-3\text{ s}$) and expected out-lap margins.
   - **FIA Sporting Regulation 30.5(n) Compliance:** Validates mandatory two-compound dry rule compliance with automatic wet-weather tyre exemptions.

2. **🌐 Cinematic 3D Spatial Telemetry Engine**
   - **Hardware-Accelerated WebGL (Three.js):** Cubic spline centerline reconstruction with uniform arc-length parameterization.
   - **Elevation Accentuation ($2.0\times$):** Highlights iconic circuit elevation profiles (Monaco Casino climb, Austin Turn 1, Spa-Francorchamps).
   - **Apex Speed Heatmapping:** Continuous track gradient from heavy braking (red `#FF1801`) to acceleration (green `#39B54A`) to DRS top speed (cyan `#00F5D4`).
   - **Dynamic Camera Presets:** Switch seamlessly between Tactical 3D, Helicopter Drone, and Cockpit Chase cams.
   - **Delta Ghosting Ribbon:** Visualizes spatial track separation and micro-deltas ($\Delta t$) between leading cars.

3. **⚡ High-Frequency Telemetry Lab**
   - **50m Micro-Sector Delta Ribbons:** Pinpoints meter-by-meter where lap time is won or lost.
   - **Corner-by-Corner Dynamics:** Automated braking point onset distance and minimum corner apex speeds.
   - **Aerodynamic Speed Trap Matrix:** Maximum velocity breakdown across Intermediate 1, Intermediate 2, Finish Line, and Speed Trap.

4. **🔊 Broadcast Pit-Wall Audio Effects**
   - **Synthesized Web Audio Engine:** Generates authentic team radio telemetry pit chimes ("Box, Box, Box") and Safety Car caution horn pulses with zero external MP3 dependencies.

5. **🔬 Rigorous Statistical Strategy Validation**
   - **Zero Leakage Stop Windows:** Preserves causal integrity evaluated at origin horizons ($\Delta \in \{5, 10, 15\}$ laps before real stop) with race-bootstrap confidence intervals.
   - **Canonical Fuel Correction ($c = 0.058\text{ s/lap}$):** Shared physical normalization ensuring identical sign conventions across all models.
   - **Honest Negative Result Disclosures:** Transparently discloses that Persistence remains the benchmark standard and that the decision engine serves as an exploratory what-if calculator.

6. **📦 Built-in Offline Demo Mode**
   - Bundled dataset (`data/demo/demo_laps.parquet`, 0.79 MB) containing 12,012 clean laps across 13 Grand Prix events.
   - Fully operational with OS-level blocked network sockets.

---

## 🏗️ System Architecture

```
F1-Race-Intelligence/
├── app.py                          # Streamlit application entry point & unified router
├── config.py                       # Centralized configuration & theme tokens
├── requirements.txt                # Pinned production dependencies
├── docs/                           # Technical documentation suite
│   ├── ARCHITECTURE.md             # System design, dataflow, and component structure
│   ├── METHODOLOGY.md              # Mathematical derivations, formulas, and physics models
│   ├── MODEL_VALIDATION.md         # Empirical benchmarks, LORO evaluation & negative gates
│   ├── DATA_PROVENANCE.md          # Data sources, FastF1 schemas & offline demo dataset
│   └── KNOWN_LIMITATIONS.md        # Physical confounders, survival bias & modeling boundaries
├── src/
│   ├── data/                       # Modular cleaning, loader, and schema validation
│   ├── features/fuel.py            # Canonical fuel normalization & S1a anchor helper
│   ├── decision_engine.py          # Monte Carlo decision engine & undercut evaluator
│   ├── telemetry.py                # Telemetry extraction & distance alignment
│   ├── tyre_analysis.py            # Per-stint degradation regression & tyre life
│   ├── strategy_simulator.py       # Deterministic pit window optimizer
│   ├── ml_pipeline.py              # LORO cross-validation & ML pipelines
│   └── visualization/
│       ├── __init__.py             # Unified visualization package exports
│       ├── charts.py               # 2D Plotly dark theme chart factory
│       ├── track_3d.py             # 3D WebGL / Three.js track & telemetry engine
│       └── audio.py                # Synthesized Web Audio pit-wall radio chimes
├── pages/
│   ├── race_cockpit.py             # Mission Control Pit-Wall Cockpit UI
│   ├── race_overview.py            # Session Overview UI
│   ├── driver_comparison.py        # Head-to-Head Driver Comparison UI
│   ├── tyre_intelligence.py        # Tyre Intelligence UI
│   ├── telemetry_lab.py            # Driver Telemetry Lab UI
│   ├── strategy_simulator.py       # Strategy Simulator UI
│   ├── ml_lab.py                   # Machine Learning Lab & Validation UI
│   └── season_explorer.py          # Season Explorer UI
└── tests/                          # 70 unit tests passing offline (Pytest & Ruff)
```

---

## 📚 Technical Documentation Suite

For deep technical evaluations, refer to the dedicated documentation:
* [System Architecture](docs/ARCHITECTURE.md) — Layered platform design, state management, and 3D WebGL pipeline.
* [Mathematical Methodology](docs/METHODOLOGY.md) — Fuel mass burn-off normalization, Huber robust regression, S1a derivation, and difference-based Monte Carlo decision theory.
* [Model Validation & Benchmarks](docs/MODEL_VALIDATION.md) — LORO cross-validation leaderboard, stop-window hit rates, and finish-order comparisons.
* [Data Provenance & Offline Dataset](docs/DATA_PROVENANCE.md) — FastF1 / Ergast schema definitions, 13 offline demo races, and cleaning protocols.
* [Known Limitations & Physical Confounders](docs/KNOWN_LIMITATIONS.md) — Traffic dirty air, track rubbering, stint selection bias, and negative results.

---

## 📊 Scientific Benchmark Summary

| Evaluation Benchmark | Sample ($N$) | Metric | Baseline | Engine / Model | Result |
|---|---|---|---|---|---|
| **Stop-Window Hit Rate ($k=2$, width=5)** | 173 | Hit Rate | Circuit Median: **31.8%** | Engine: **12.7%** | Median Baseline Beats Engine |
| **Pairwise Finish Order Accuracy** | 1,079 | Accuracy | Track Persistence: **85.8%** | Engine: **79.5%** | Track Persistence Beats Engine |
| **Close Pair Order ($\le 2$ positions)** | 374 | Accuracy | Track Persistence: **71.4%** | Engine: **62.6%** | Track Persistence Beats Engine |
| **S1b Linear Drift ($H=10$)** | 754 | MAE | Persistence: **0.595 s** | S1b Drift: **0.600 s** | Tie (Diff CI includes zero) |
| **S1b Linear Drift ($H=20$)** | 205 | MAE | Persistence: **0.762 s** | S1b Drift: **0.754 s** | Tie (Diff CI includes zero) |
| **Tactical Undercut ($\text{gap} \le 3\text{ s}$)** | 80 | Success Rate | Constant: $50.0\%$ | Empirical: **51.2%** | [95% Wilson CI: 40.5%, 61.9%] |

---

## 🎮 Local Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/akashcodes23/F1-Race-Intelligence.git
cd F1-Race-Intelligence

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Run automated test suite (70 tests)
pytest tests/

# 5. Launch Streamlit Cockpit
streamlit run app.py
```

---

## 📜 License & Compliance
Licensed under the [MIT License](LICENSE). Telemetry and timing data processed in accordance with Formula 1 sporting regulations and FastF1 non-commercial research provisions.
