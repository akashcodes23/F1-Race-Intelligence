# 🏛️ F1 Race Intelligence 3.0 — System Architecture

## 1. System Overview

**F1 Race Intelligence 3.0** is an enterprise-grade motorsport engineering and telemetry analytics platform. The application is structured in decoupled, testable layers separating raw data ingestion, schema validation, physical normalization, predictive modeling, tactical scenario simulation, and visualization.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION & COCKPIT LAYER                      │
│   Streamlit Web Interface (app.py) • Dynamic Router • Navigation Bar    │
│   Pit-Wall Cockpit HUD • 3D WebGL / Three.js Engine • Web Audio Alerts │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  ANALYTICAL & DECISION ENGINE LAYER                    │
│   Monte Carlo Strategy Engine (src/decision_engine.py)                 │
│   Tactical Undercut / Overcut Radar (evaluate_undercut_overcut)        │
│   Deterministic Horizon Simulator (src/strategy_simulator.py)         │
│   Machine Learning LORO Validation Pipeline (src/ml_pipeline.py)       │
│   Tyre Degradation Huber Regressor (src/tyre_analysis.py)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│               FEATURE ENGINEERING & PHYSICAL NORMALIZATION              │
│   Canonical Fuel Normalization Helper (src/features/fuel.py)           │
│   Distance-Aligned Telemetry Normalizer (src/telemetry.py)             │
│   Driver Performance Delta Matrices (src/driver_comparison.py)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                     DATA INGESTION & DATA INTEGRITY                    │
│   FastF1 Live Session Loader & Schedule Discovery (src/data/loader.py) │
│   Outlier Filtering & Track Status Cleaner (src/data/cleaning.py)      │
│   Pydantic Schema Validation & Type Coercion (src/data/schema.py)      │
│   Offline Parquet Demo Repository (data/demo/demo_laps.parquet)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Layout & Module Responsibilities

```
F1-Race-Intelligence/
├── app.py                          # Streamlit application entry point & unified router
├── config.py                       # Centralized configuration, physical bounds & color tokens
├── pyproject.toml                  # Toolchain configuration (Pytest, Ruff, coverage)
├── requirements.txt                # Pinned production dependencies
├── data/
│   └── demo/
│       ├── demo_laps.parquet       # Offline repository (12,012 clean laps across 13 GPs)
│       └── ATTRIBUTION.md          # Open data provenance & FastF1/Ergast attribution
├── docs/
│   ├── ARCHITECTURE.md             # System design & component interaction
│   ├── METHODOLOGY.md              # Physical models, formulas, and derivations
│   ├── MODEL_VALIDATION.md         # Empirical gates, benchmark tables, and baselines
│   ├── DATA_PROVENANCE.md          # Raw data sources, schemas, and offline coverage
│   └── KNOWN_LIMITATIONS.md        # Explicit disclosure of bounds and negative results
├── pages/
│   ├── race_cockpit.py             # Pit-Wall Command Center (HUD, 3D Scrubber, MC Matrix)
│   ├── race_overview.py            # Session telemetry summary, leaderboard & progression
│   ├── driver_comparison.py        # Head-to-head lap delta, dispersion & sector analysis
│   ├── tyre_intelligence.py        # Robust Huber degradation & fuel correction engine
│   ├── telemetry_lab.py            # High-frequency overlays, 50m micro-sectors & speed traps
│   ├── strategy_simulator.py       # Deterministic pit window & pit loss sensitivity sweep
│   ├── ml_lab.py                   # Machine learning LORO benchmark leaderboard & gate audit
│   └── season_explorer.py          # Championship calendar & event discovery
├── scripts/
│   ├── verify_offline_app_render.py# Socket-blocked offline validation across all 13 demo GPs
│   └── run_phase3a4_backtest.py    # Zero-leakage stop-window & finish-order evaluation
├── src/
│   ├── data/
│   │   ├── loader.py               # FastF1 ingestion, cache management & offline fallback
│   │   ├── cleaning.py             # Caution filtering, in-lap/out-lap & MAD outlier rejection
│   │   └── schema.py               # Pydantic schema validation & dtype coercion
│   ├── features/
│   │   └── fuel.py                 # Canonical fuel normalization & S1a anchor helper
│   ├── decision_engine.py          # Monte Carlo decision engine & undercut evaluator
│   ├── telemetry.py                # Telemetry extraction & distance alignment
│   ├── tyre_analysis.py            # Per-stint degradation regression & tyre life
│   ├── strategy_simulator.py       # Deterministic pit window optimizer
│   ├── ml_pipeline.py              # LORO cross-validation & ML pipelines
│   └── visualization/
│       ├── __init__.py             # Unified chart exports
│       ├── charts.py               # 2D Plotly dark theme chart factory
│       ├── track_3d.py             # 3D WebGL / Three.js track & telemetry engine
│       └── audio.py                # Synthesized Web Audio pit-wall radio chimes
└── tests/                          # Automated Pytest suite (70 unit & integration tests)
```

---

## 3. Four-Category Information Architecture

To guarantee scientific honesty, the system labels every visual element into one of four strict categories:

1. **`[OBSERVED]` (Raw Source Data):** Direct telemetry and timing observations recorded by official FIA timing equipment (e.g. `LapTime`, `Sector1Time`, `SpeedST`, `Compound`, `TrackStatus`).
2. **`[DERIVED]` (Deterministic Calculations):** Pure deterministic computations calculated without fitting stochastic or statistical models (e.g. cumulative lap deltas, micro-sector time gaps, driver positions, median lap pace).
3. **`[ESTIMATED]` (Statistical / Machine Learning Models):** Outputs produced by empirical parameter fitting, regression, or probabilistic models (e.g. Huber degradation slopes, fuel burn coefficients $c$, Monte Carlo win probabilities, S1a pace projections).
4. **`[SIMULATED]` (Explicit Scenario Projections):** Outputs generated under hypothetical scenario parameters or reconstructed geometric assumptions (e.g. 3D track spline positions, candidate pit lap sweeps, hypothetical optimal lap combinations).

---

## 4. Frontend & 3D WebGL Engine

- **Streamlit Single-Page Router:** `app.py` acts as the single application controller. Multipage sub-routing is handled via an in-page top radio navigation bar with state preserved in `st.session_state`.
- **Hardware-Accelerated 3D Engine (`src/visualization/track_3d.py`):**
  - Parameterized cubic spline with periodic boundary conditions ($C^2$ continuity) through validated circuit geometry waypoints.
  - Uniform arc-length resampling with tangent vector calculation ($\hat{t}$).
  - Multi-car positioning dynamically derived from cumulative timing gaps and reference lap times.
  - Dual rendering modes: Embedded Three.js WebGL (60 FPS hardware acceleration) and interactive Plotly 3D fallback.
- **Synthesized Web Audio (`src/visualization/audio.py`):**
  - Generates authentic pit-wall chime audio cues ("Box, Box, Box" at 880 Hz / 1760 Hz dual-tone) and Caution Horn sweeps via pure browser Web Audio API oscillator synthesis.
  - Zero external MP3/WAV dependencies; 100% offline compatible; strictly honors user mute settings.

---

## 5. Caching & Memory Management

- **FastF1 Disk Cache:** Initialized once via `src/data_loader.init_cache(CACHE_DIR)` to store raw Ergast and F1 timing payloads.
- **Streamlit Cache (`@st.cache_data`):**
  - `fetch_session_data`: Caches parsed lap timing DataFrames by `(year, gp, session_id, with_tel)`.
  - Immutable DataFrame patterns are enforced: functions return `.copy()` to avoid Streamlit MutationWarnings.
