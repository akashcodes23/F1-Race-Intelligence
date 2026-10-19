# 🏎️ F1 Race Intelligence 2.0
[![Live Demo](https://img.shields.io/badge/Live-Demo-red?style=for-the-badge)](https://f1-race-intelligence-qmzz4nuvmzql4hac6ajbbq.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)]()
[![Test Suite](https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge)]()

**F1 Race Intelligence 2.0** is an enterprise-grade motorsport analytics, predictive modeling, and strategic decision-support platform designed to replicate real-world Formula 1 pit wall race engineering workflows.

---

## 🎯 Platform Highlights

1. **🏁 Session Overview & Leaderboard**
   - Official session classifications, gap to leader, and fastest lap tracking.
   - Dynamic race position progression and driver pacing distribution boxplots.
   - Comprehensive tyre stint strategy timelines across the full field.

2. **⚔️ Driver Performance Lab**
   - Lap-by-lap delta and cumulative race gap tracking (+ indicates Driver 1 slower).
   - Driver pace consistency profiling (standard deviation, median pace, IQR).
   - Sector-by-sector analysis (S1, S2, S3 bests) and hypothetical optimal lap times.

3. **🛞 Tyre Intelligence & Degradation Modeling**
   - Per-stint Robust Huber degradation regression ($s / \text{lap}$) with fuel-burn mass correction.
   - 95% Confidence Intervals (standard and HAC autocorrelation-consistent).
   - Monotone compound benchmarking with confidence intervals (tyre cliff change-point detection evaluated as research-only).

4. **⚡ High-Frequency Telemetry Lab**
   - Millisecond-aligned Speed (km/h), Throttle %, and Brake traces.
   - Distance-normalized driver overlays for corner-by-corner performance deltas.
   - 2D circuit map with velocity heatmap.

5. **⏱️ Pit Wall Strategy Simulator**
   - Projected race time comparison: Stay-out vs Pit-now over equal horizons.
   - Mathematical Pit Window Optimizer evaluating candidate pit laps.
   - Sensitivity matrix testing pit loss variance (Standard vs VSC / Safety Car).

6. **🧠 Machine Learning Lab**
   - Zero-leakage chronological and Leave-One-Race-Out (LORO) cross-validation across 13 Grand Prix events.
   - Anchored Physics Model ($S_1a$) and HistGradientBoostingRegressor benchmarked against persistence baselines.
   - Multi-origin horizon forecasting ($H \in \{1, 5, 10, 20\}$) and pit pace-gain accuracy diagnostics.

7. **📦 Built-in Offline Demo Mode**
   - Offline bundled demo dataset (`data/demo/demo_laps.parquet`, 0.79 MB) containing 12,012 clean laps across 13 Grand Prix events.
   - Runs seamlessly without live network access or FastF1 live timing dependencies.

---

## 🏗️ System Architecture

```
F1-Race-Intelligence/
├── app.py                          # Streamlit application entry point & router
├── config.py                       # Centralized configuration & theme tokens
├── requirements.txt                # Pinned production dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit dark theme & client settings
├── src/
│   ├── __init__.py                 # Source package init
│   ├── data_loader.py              # Dynamic FastF1 session loading & caching
│   ├── data_validator.py           # Robust lap cleaning, outlier & pit filtering
│   ├── race_analytics.py           # Session overview, leaderboards & stint stats
│   ├── driver_comparison.py        # Head-to-head delta, consistency & sector pacing
│   ├── tyre_analysis.py            # Per-stint degradation regression & tyre life
│   ├── telemetry.py                # High-frequency traces & distance alignment
│   ├── strategy_simulator.py       # Pit wall horizon simulation & sensitivity
│   ├── ml_pipeline.py              # Chronological ML models & diagnostics
│   ├── visualization.py            # Unified Plotly dark theme chart factory
│   └── utils.py                    # Time formatters & numeric helpers
├── pages/
│   ├── race_overview.py            # Session Overview UI
│   ├── driver_comparison.py        # Driver Comparison UI
│   ├── tyre_intelligence.py        # Tyre Intelligence UI
│   ├── telemetry_lab.py            # Telemetry Lab UI
│   ├── strategy_simulator.py       # Strategy Simulator UI
│   ├── ml_lab.py                   # Machine Learning Lab UI
│   └── season_explorer.py          # Season Explorer UI
└── tests/
    ├── test_data_validator.py      # Lap validation unit tests
    ├── test_driver_comparison.py   # Driver delta & consistency tests
    ├── test_tyre_analysis.py       # Tyre degradation tests
    ├── test_strategy_simulator.py  # Pit strategy simulation tests
    └── test_ml_pipeline.py         # ML pipeline & leakage verification tests
```

---

## 🔬 Analytical Methodology & Integrity

- **No Data Leakage:** All lap prediction ML models are evaluated strictly on chronological future test splits. Cross-validation utilizes `TimeSeriesSplit` confined to the training partition.
- **Categorical Integrity:** Tyre compounds are one-hot encoded instead of arbitrary ordinal integer mappings.
- **Per-Stint Degradation:** Regression models are fitted per tyre stint rather than across pit stops or full races.
- **Defensive Data Validation:** Quantile trimming removes pit in/out outliers and safety car anomalies without raising unhandled exceptions on short or practice sessions.

---

## 🎮 Local Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/akashcodes23/F1-Race-Intelligence.git
cd F1-Race-Intelligence

# 2. Create virtual environment & activate
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run test suite
python -m unittest discover tests

# 5. Launch application
streamlit run app.py
```

---

## 🧪 Running the Test Suite

```bash
python -m unittest discover tests -v
```

---

## 📄 License

Licensed under the MIT License.
