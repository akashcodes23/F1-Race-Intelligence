# F1 Race Intelligence 3.0
### Motorsport Analytics · Race Strategy Simulation · Tyre Degradation Modelling · 3D Circuit Rendering

[![Live Demo](https://img.shields.io/badge/Live%20Demo-f1--race--intelligencee.streamlit.app-E10600?style=for-the-badge&labelColor=0c0d14)](https://f1-race-intelligencee.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-F5A623?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-81%20passing-39B54A?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Linter](https://img.shields.io/badge/Linter-Ruff-00F5D4?style=for-the-badge)](https://docs.astral.sh/ruff/)

---

F1 Race Intelligence is an open-source motorsport analytics platform built in Python and Streamlit.
It combines tyre degradation modelling, difference-based Monte Carlo strategy simulation, machine
learning benchmarking, and a hardware-accelerated 3D circuit visualizer into a single,
locally-runnable application.

Users can explore historical lap timing across 13 Grand Prix events (2021–2023), simulate pit stop
scenarios, compare driver head-to-head pace, and examine why simple baselines remain remarkably
competitive against more complex models — an honest result this project treats as a core engineering
finding rather than something to hide.

> **Disclaimer:** This is a non-commercial, open-source research and education project. It is not
> associated with, sponsored by, or affiliated with Formula 1, the FIA, Formula One Management,
> or any Formula 1 constructor.

---

## Table of Contents

- [Platform overview](#platform-overview)
- [Architecture](#architecture)
- [Scientific integrity](#scientific-integrity)
- [Benchmark scorecard](#benchmark-scorecard)
- [Data and methodology](#data-and-methodology)
- [Quick start](#quick-start)
- [Offline demo mode](#offline-demo-mode)
- [Testing and quality checks](#testing-and-quality-checks)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License and attribution](#license-and-attribution)

---

## Platform overview

<!--
  SCREENSHOT PLACEHOLDER
  Add a real screenshot of the application here when available.
  Suggested path:  docs/assets/f1-race-intelligence-cockpit.png
  Suggested capture: Pit-Wall Cockpit page, 2023 Monaco Grand Prix demo data loaded.
  Example Markdown once the file exists:
    ![F1 Race Intelligence — Pit-Wall Cockpit](docs/assets/f1-race-intelligence-cockpit.png)
-->

The platform is navigated through eight sections, each implemented as a dedicated Streamlit page
and routed via `app.py`.

| Section | Engineering question answered | Key capability | Data classification |
|---|---|---|---|
| **Pit-Wall Cockpit** | When should this car pit? | Monte Carlo PIT / STAY OUT verdict with strict INCONCLUSIVE gate; undercut radar; lap scrubber advancing 3D car positions | [ESTIMATED] MC probabilities · [DERIVED] gap deltas · [SIMULATED] 3D positions |
| **Race Overview** | How did the race unfold lap by lap? | Position progression, leaderboard, stint summary, and speed-trap matrix | [OBSERVED] lap timing and sector splits |
| **Driver Comparison** | Who was faster and where? | Head-to-head lap-delta analysis, sector breakdown, consistency index (scaled MAD). Deltas are completed-lap elapsed proxies, not continuous timing intervals | [DERIVED] from observed lap times |
| **Tyre Intelligence** | How fast are these tyres degrading? | Huber-regression degradation slope per compound and stint; fuel-corrected pace; tyre cliff change-point detection | [ESTIMATED] regression coefficients · [OBSERVED] speed traps |
| **Strategy Simulator** | What does a one- vs two-stop strategy cost? | Deterministic horizon sweep across pit-lap and compound combinations; pit-loss sensitivity | [SIMULATED] scenario projections; not empirically validated recommendations |
| **Telemetry Lab** | Where is lap time won and lost? | Speed traces, braking onset distances, apex speeds, 50 m micro-sector delta ribbons. Offline telemetry is reconstructed, not high-frequency onboard data | [RECONSTRUCTED] offline · [OBSERVED] when live FastF1 telemetry loads |
| **ML Lab & Validation** | Can a model beat persistence? | LORO cross-validation leaderboard across six architectures; multi-horizon gate evaluation; honest disclosure of negative results | [ESTIMATED] model outputs |
| **Season Explorer** | What events are available? | Championship calendar and event discovery for FastF1 live mode | [OBSERVED] schedule metadata |

---

## Architecture

```
F1-Race-Intelligence/
├── app.py                          # Streamlit entry point and single-page router
├── config.py                       # Physical constants, colour tokens, and season config
├── pyproject.toml                  # Toolchain config (Pytest, Ruff, build system)
├── requirements.txt                # Pinned production dependencies
├── data/
│   └── demo/
│       ├── demo_laps.parquet       # Offline dataset: 12,012 laps across 13 Grand Prix events
│       └── ATTRIBUTION.md          # Data provenance and FastF1/Ergast attribution notice
├── docs/
│   ├── ARCHITECTURE.md             # Layered system design and component interaction
│   ├── METHODOLOGY.md              # Mathematical derivations, formulas, and physics models
│   ├── MODEL_VALIDATION.md         # Benchmark tables, LORO evaluation, and negative gates
│   ├── DATA_PROVENANCE.md          # Data sources, schemas, and offline dataset coverage
│   └── KNOWN_LIMITATIONS.md        # Explicit disclosure of physical and statistical boundaries
├── pages/
│   ├── race_cockpit.py             # Pit-Wall Command Centre: HUD, 3D scrubber, MC matrix
│   ├── race_overview.py            # Session leaderboard and lap progression
│   ├── driver_comparison.py        # Head-to-head delta and sector analysis
│   ├── tyre_intelligence.py        # Huber degradation regression and tyre life estimates
│   ├── telemetry_lab.py            # Speed traces, micro-sectors, and speed-trap matrix
│   ├── strategy_simulator.py       # Deterministic pit-window and compound scenario sweep
│   ├── ml_lab.py                   # LORO benchmark leaderboard and gate audit
│   └── season_explorer.py          # Championship calendar and event discovery
├── scripts/
│   ├── verify_offline_app_render.py  # Socket-blocked offline validation (13 demo events)
│   └── run_phase3a4_backtest.py      # Zero-leakage stop-window and finish-order backtest
├── src/
│   ├── data/
│   │   ├── loader.py               # FastF1 ingestion, cache management, and offline fallback
│   │   ├── cleaning.py             # Safety car filtering, in/out-lap exclusion, MAD rejection
│   │   └── schema.py               # Pydantic schema validation and dtype coercion
│   ├── features/
│   │   └── fuel.py                 # Canonical fuel normalization (c = 0.058 s/lap) and S1a helper
│   ├── decision_engine.py          # Monte Carlo strategy engine and undercut evaluator
│   ├── driver_comparison.py        # Driver delta computation and consistency metrics
│   ├── telemetry.py                # Telemetry extraction and distance-grid alignment
│   ├── tyre_analysis.py            # Per-stint Huber regression and tyre life modelling
│   ├── strategy_simulator.py       # Deterministic pit-window optimizer
│   ├── ml_pipeline.py              # LORO cross-validation and ML model pipelines
│   └── visualization/
│       ├── charts.py               # Plotly dark-theme chart factory
│       ├── track_3d.py             # Three.js WebGL track engine and Plotly 3D fallback
│       └── audio.py                # Synthesized Web Audio pit chimes (zero MP3 dependencies)
└── tests/                          # 81 unit and integration tests (Pytest)
```

### Data flow

```
Bundled Parquet (offline) ──┐
FastF1 live session (online) ┘
        │
        ▼
  Data validation (Pydantic schema) + cleaning (MAD filter, caution-lap exclusion)
        │
        ▼
  Feature engineering: fuel normalization · sector deltas · stint segmentation
        │
        ▼
  Analytics layer: Huber regression · Monte Carlo engine · LORO ML pipeline
        │
        ▼
  Streamlit interface: eight navigable sections · 3D WebGL circuit · Web Audio
```

Technical documentation:
[Architecture](docs/ARCHITECTURE.md) ·
[Methodology](docs/METHODOLOGY.md) ·
[Model Validation](docs/MODEL_VALIDATION.md) ·
[Data Provenance](docs/DATA_PROVENANCE.md) ·
[Known Limitations](docs/KNOWN_LIMITATIONS.md)

---

## Scientific integrity

Every value displayed in the application is labelled according to how it was produced. This is not
a cosmetic choice — it is an engineering discipline that distinguishes what the data directly
supports from what the model infers or assumes.

| Label | Meaning | Examples in this project |
|---|---|---|
| **[OBSERVED]** | Directly present in source data with no derivation | Lap times, sector splits, speed-trap readings, compound, track status |
| **[DERIVED]** | Calculated deterministically from observations, no stochastic fitting | Cumulative lap deltas, position history, micro-sector time gaps, driver standings |
| **[ESTIMATED]** | Inferred using a statistical or machine learning procedure | Huber degradation slopes, fuel-burn coefficient *c*, Monte Carlo win probabilities, S1a projections |
| **[RECONSTRUCTED]** | Rebuilt from lower-resolution or incomplete data | Offline telemetry traces (generated from circuit geometry and speed-trap maxima; throttle and brake derived from acceleration gradients) |
| **[SIMULATED]** | Generated under hypothetical scenario assumptions | 3D car positions from timing gaps; candidate pit-lap sweeps; strategy comparisons |

**Three specific claims this project does not make:**

1. **Timing proxies ≠ continuous intervals.** Head-to-head deltas measure the difference in elapsed
   time to complete equivalent laps. They cannot capture intra-lap position changes.

2. **Reconstructed ≠ measured telemetry.** The offline demo does not contain continuous speed,
   throttle, brake, or RPM time series. The application labels all reconstructed traces as
   [RECONSTRUCTED] or [SIMULATED].

3. **Strategy output ≠ validated prediction.** The Monte Carlo engine is an exploratory scenario
   calculator. When its 80% credible interval spans zero, it outputs INCONCLUSIVE and suppresses
   speculative recommendations.

---

## Benchmark scorecard

All results are drawn from [`docs/MODEL_VALIDATION.md`](docs/MODEL_VALIDATION.md) and the
associated backtest artifacts. Negative results are reported alongside positive ones.

### LORO leaderboard — H=1 next-lap pace forecast

Evaluated via Leave-One-Race-Out cross-validation across 13 held-out Grand Prix events
(10,637 clean dry laps, laps ≥ 7 only):

| Model | Mean MAE | Mean RMSE | vs Persistence |
|---|---|---|---|
| S1 Physics Linear | 0.695 s | 0.957 s | +81.0% |
| S2 Scaled Ridge | 0.718 s | 0.984 s | +87.0% |
| Huber Per-Stint | 0.424 s | 0.664 s | +10.4% |
| **Persistence Baseline** | **0.384 s** | **0.626 s** | **Reference** |
| HGBR Variant B (Static) | 0.776 s | 1.042 s | +102.1% |
| HGBR Variant A (Autoregressive) | 0.367 s | 0.573 s | −4.4% (p = 0.0693, not significant at α = 0.05) |

> **Interpretation:** Persistence — predicting that the next lap will match the current lap — is
> the unbeaten benchmark. HGBR Variant A achieves a 0.017 s lower MAE, but a paired t-test across
> 13 folds yields p = 0.0693, failing conventional significance. No candidate model has cleared
> the promotion gate at multi-step horizons (H ≥ 5 laps).

### Pairwise finishing-order accuracy

Evaluated at tactical pit-stop origins across held-out races:

| Cohort | N | Naive baseline (track persistence) | Decision Engine | Difference [95% CI] | Verdict |
|---|---|---|---|---|---|
| All competing pairs | 1,079 | **85.8%** | 79.5% | −6.3% [−7.7%, −5.5%] | Baseline wins |
| Close pairs (≤ 2 positions) | 374 | **71.4%** | 62.6% | −8.8% [−12.0%, −6.6%] | Baseline wins |
| Teammate pairs | 67 | **83.6%** | 74.6% | −9.0% [−16.3%, 0.0%] | Baseline wins |

> **Interpretation:** A naive baseline that projects the current track order forward outperforms the
> Decision Engine on all cohorts. This reflects Formula 1's severe overtaking constraints — dirty
> air, narrow track widths, and DRS trains make track position extremely valuable. The engine is
> designed as a what-if simulator, not a finishing-order oracle.

### Stop-window hit rate vs equal-width baselines

Evaluated at causal origins Δ ∈ {5, 10, 15} laps before the real stop, without future information:

| Window half-width (k) | Window width | Engine hit rate | Circuit median baseline | Verdict |
|---|---|---|---|---|
| k = 0 | 1 lap | 2.9% | **3.5%** | Baseline wins |
| k = 1 | 3 laps | 9.8% | **22.5%** | Baseline wins |
| k = 2 | 5 laps | 12.7% | **31.8%** | Baseline wins (N = 173) |
| k = 3 | 7 laps | 18.5% | **39.3%** | Baseline wins |
| k = 4 | 9 laps | 42.8% | **44.5%** | Baseline wins |
| k = 5 | 11 laps | 48.0% | **52.6%** | Baseline wins |
| k = 8 | 17 laps | 64.7% | **66.5%** | Baseline wins |
| k = 10 | 21 laps | **82.1%** | 72.8% | Engine wins |

> At narrow tactical windows (k ≤ 4), the historical circuit-median baseline consistently
> outperforms the engine. Track-specific pit habits are a strong prior. At very wide windows
> (k ≥ 9), the engine's degradation modelling begins to add value.

### Tactical undercut probability (gap ≤ 3 s)

| N | Constant baseline | Empirical success rate | 95% Wilson CI | Verdict |
|---|---|---|---|---|
| 80 | 50.0% | 51.2% | [40.5%, 61.9%] | Improvement is uncertain |

> The confidence interval spans 50%, meaning the observed 1.2-point improvement over random
> cannot be distinguished from chance at this sample size. This result is retained and disclosed
> rather than suppressed.

**Central message:** A model is not useful simply because it is more complex than its baseline.
Persistence is a meaningful benchmark for sequential lap-time prediction because F1 drivers
actively manage pace to a target delta, making lap times more stationary than most autoregressive
models assume.

---

## Data and methodology

### Bundled offline dataset

| Property | Value |
|---|---|
| File | `data/demo/demo_laps.parquet` |
| Format | Snappy-compressed columnar Parquet |
| File size | ~0.79 MB |
| Total laps | 12,012 clean laps |
| Events | 13 Grand Prix races across 2021, 2022, and 2023 |
| Drivers per event | 20 |

**Event inventory (verified from parquet):**

| Season | Grand Prix events |
|---|---|
| 2021 | Austrian · Bahrain · British · Italian · Spanish · United States |
| 2022 | Austrian · Bahrain · British · French · Italian · Spanish |
| 2023 | Monaco |

The dataset contains lap timing, sector splits, speed-trap readings, compound, tyre life, track
status, and driver/team identifiers. It does **not** contain continuous millisecond telemetry arrays.
See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for the full column schema.

### Key analytical methods

**Fuel normalization** — Lap times are corrected to a full-fuel reference using an empirically
calibrated coefficient c = 0.058 s/lap (clamped to [0.02, 0.08] s/lap). This isolates tyre
degradation from mass burn-off. See [docs/METHODOLOGY.md §1](docs/METHODOLOGY.md).

**Huber robust regression** — Stint degradation slopes are fitted with Huber M-estimation
(δ = 1.345 · MAD) rather than OLS to reduce sensitivity to traffic-affected laps and anomalous
outliers.

**Monte Carlo decision engine** — The engine samples from the difference distribution
ΔT = T_stay − T_pit across 1,000 runs, incorporating stochastic pit loss, safety car occurrence,
and trend-scenario uncertainty. When the 80% credible interval spans zero, the verdict is forced to
INCONCLUSIVE.

**LORO cross-validation** — ML models are evaluated via Leave-One-Race-Out cross-validation:
trained on 12 events, tested on the 13th. This prevents within-session overfitting and approximates
out-of-sample generalisation across circuits.

**FIA two-compound rule check** — The application checks whether at least two different dry-weather
compounds have been used during a race, consistent with FIA Sporting Regulation Article 30.5(n).
Wet-weather tyre exemptions are applied automatically. This is a rule-interpretation implementation,
not a formally certified compliance check.

---

## Quick start

**Prerequisites:** Python 3.10 or 3.11 · pip ≥ 23.0 recommended · No API credentials required
for offline demo mode.

```bash
# 1. Clone the repository
git clone https://github.com/akashcodes23/F1-Race-Intelligence.git
cd F1-Race-Intelligence

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the test suite (expected: 81 tests passing)
pytest tests/

# 5. Launch the application
streamlit run app.py
```

The application opens at `http://localhost:8501`. On first launch, select
**📦 Bundled Demo (13 Offline GPs)** in the sidebar and click **⚡ Load Session Data** —
no network connection is required.

**Live FastF1 mode (optional):** Switch to **🌐 Live FastF1 Exploration** to query historical
sessions via FastF1. A network connection is required. Note that live Formula 1 timing feeds
require authentication (HTTP 401); FastF1 accesses cached historical archives rather than a
real-time stream.

---

## Offline demo mode

The application ships with a self-contained dataset that makes all core analytical features
available without a network connection.

**What works offline:**
- All eight navigation sections
- Pit-Wall Cockpit with Monte Carlo strategy matrix and 3D circuit engine
- Tyre Intelligence degradation regression
- Strategy Simulator scenario sweep
- ML Lab LORO benchmark leaderboard
- Driver Comparison and Race Overview lap analysis
- Synthesized Web Audio pit chimes

**What requires network access:**
- Live FastF1 session loading for events outside the 13 bundled races
- Season Explorer live calendar queries

**How demo data is selected:** `get_demo_races()` in `src/data_loader.py` reads the Parquet file
and returns the list of available (Year, GrandPrix) pairs. The sidebar presents these as a dropdown.
When demo mode is active, the application displays a caption noting the offline source.

**Distinguishing demo from live data:** A sidebar indicator labels the active data source as either
"Bundled Demo (Offline)" or "Live FastF1 Session Object" in colour-coded text.

**Offline rendering verification:**
```bash
python scripts/verify_offline_app_render.py
```
This script blocks OS-level network sockets and confirms all 13 demo events render without errors.

---

## Testing and quality checks

```bash
# Full test suite (81 tests across 12 modules)
pytest tests/

# Lint and style check
ruff check .

# Offline rendering validation
python scripts/verify_offline_app_render.py
```

### Test modules

| Module | What it covers |
|---|---|
| `test_data_cleaning.py` | Safety car filtering, in/out-lap exclusion, MAD outlier rejection |
| `test_data_schema.py` | Pydantic schema validation and dtype coercion |
| `test_data_validator.py` | Data validator logic |
| `test_decision_engine.py` | Monte Carlo verdict rules, INCONCLUSIVE gate, undercut evaluator |
| `test_driver_comparison.py` | Lap delta alignment and sector breakdown |
| `test_fuel.py` | Fuel normalization, canonical coefficient, S1a anchor |
| `test_ml_pipeline.py` | LORO cross-validation, persistence baseline, feature engineering |
| `test_s1a_known_answer.py` | Known-answer regression tests for S1a model output |
| `test_strategy_simulator.py` | Deterministic horizon sweep and pit-loss sensitivity |
| `test_telemetry.py` | Distance-grid alignment and speed-trace normalisation |
| `test_track_3d.py` | Cubic spline geometry, arc-length parameterisation, car positioning |
| `test_tyre_analysis.py` | Huber regression, cliff detector, consistency index |

> Unit tests verify computational correctness against synthetic and fixture data. They do not
> establish physical accuracy of the strategy model or guarantee that recommendations match
> real-world pit-wall decisions.

### CI configuration

A GitHub Actions workflow at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs Ruff,
Pytest, and the offline rendering script on Python 3.10 and 3.11 for pushes to `main` and
feature branches.

> **Known issue:** The CI workflow file contains a typo (`nname:` instead of `name:`) and has
> not yet been pushed to the remote repository. A CI badge will be added once the workflow has
> been verified against a passing remote run.

---

## Known limitations

These are engineering boundaries that guide further work.

**Timing proxies vs continuous intervals.** The completed-lap elapsed delta measures how long
each driver took to complete a lap. It is not a continuous timing interval and cannot capture
intra-lap position changes or sector-level gaps within the same lap.

**Reconstructed telemetry.** The offline dataset (~0.79 MB) stores lap timing, sector splits,
and speed-trap readings. Continuous telemetry — speed, throttle, brake, RPM, GPS at millisecond
resolution — is not included. Offline Telemetry Lab traces are reconstructed from circuit geometry
and labelled [RECONSTRUCTED].

**Dirty air and traffic.** MAD filtering and caution-lap exclusion are applied, but aerodynamic
wake deficit from following within 1.5 s is not modelled. In-traffic stints may show steeper
apparent degradation.

**Track evolution.** Fuel correction addresses mass burn-off, but grip increase from rubber
laid on the racing line is not explicitly modelled.

**Wet-weather compounds.** Tyre regressions are restricted to dry compounds (SOFT, MEDIUM, HARD).
Intermediate and full-wet laps are filtered and excluded.

**Stint selection bias.** Teams pit when tyres show excessive wear. Long-surviving stints
disproportionately represent cases without severe degradation, biasing regression slopes toward
shallower values.

**Strategy output is not a validated recommendation.** Safety Car periods, red flags, incidents,
reliability failures, and traffic can invalidate any pace projection. The INCONCLUSIVE gate is
applied precisely because uncertainty is frequently too large to support a confident recommendation.

**Persistence remains the unbeaten multi-step benchmark.** For H ≥ 5 laps ahead, no candidate
model in the current evaluation suite has cleared the promotion gate over static persistence.

See [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) for the complete catalogue.

---

## Roadmap

The following are **planned future directions**, not current capabilities.

- Stronger independent validation across additional seasons (2024+)
- Improved uncertainty calibration and prediction-interval reporting for ML models
- Cleaner UI separation of [OBSERVED] and [INFERRED] data labels
- Additional scenario-based strategy evaluation (overcut, undercut under VSC)
- Multi-lap stint simulation with stochastic safety car events in the Strategy Simulator
- Fix the `nname:` typo in `.github/workflows/ci.yml` and add a verified CI badge
- Real application screenshots in `docs/assets/` to replace the placeholder comment
- Dataset expansion beyond the current 13 events
- Accessibility improvements (WCAG contrast, keyboard navigation)
- Reproducible Jupyter notebooks for benchmark replication

---

## Contributing

Contributions are welcome. Before opening a pull request:

1. Fork the repository and create a feature branch from `main`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run `ruff check .` — no new lint errors should be introduced.
4. Run `pytest tests/` — all 81 tests must pass.
5. If adding new analytical logic, include corresponding unit tests.
6. If your change affects benchmark results, update [`docs/MODEL_VALIDATION.md`](docs/MODEL_VALIDATION.md) with the new figures and methodology notes.

Please respect the scientific integrity labelling conventions described in the
[Scientific integrity](#scientific-integrity) section. New features should clearly label
derived or estimated outputs.

---

## License and attribution

This project is released under the [MIT License](LICENSE) — Copyright © 2026 Akash G Patil.

**Third-party data and libraries:**

- **[FastF1](https://github.com/theOehrly/Fast-F1)** (MIT License) — used to ingest Formula 1
  timing, telemetry, and session data.
- **[Ergast Developer API](https://ergast.com/mrd/)** — historical results archive under
  CC BY-NC-SA 3.0 (non-commercial research use).

The MIT license covering this repository's source code does **not** extend to the underlying
Formula 1 timing or telemetry data. Redistribution of derived timing data may be subject to the
upstream provider's terms of service (Formula One Licensing B.V. / Formula One Management).

Formula 1, F1, and related marks are trademarks of Formula One Licensing B.V. This project has
no commercial use and no affiliation with any rights holder.

See [`data/demo/ATTRIBUTION.md`](data/demo/ATTRIBUTION.md) for the full data provenance and
redistribution notice.
