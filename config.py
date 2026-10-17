"""
F1 Race Intelligence — Central Configuration

All application-wide constants, thresholds, and styling tokens.
"""
import datetime
import os

# ─── Paths ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# ─── Seasons ─────────────────────────────────────────────
CURRENT_YEAR = datetime.datetime.now().year
SEASONS = list(range(min(CURRENT_YEAR, 2025), 2017, -1))

# ─── Data Validation ─────────────────────────────────────
MIN_LAPS_FOR_ANALYSIS = 3     # Minimum for any summary statistic
MIN_LAPS_FOR_REGRESSION = 5   # Minimum for regression fit
MIN_LAPS_FOR_ML = 20          # Minimum for ML model training
OUTLIER_QUANTILE_LOW = 0.05
OUTLIER_QUANTILE_HIGH = 0.95

# ─── ML ──────────────────────────────────────────────────
CV_SPLITS = 3
TEST_SIZE = 0.2

# ─── Strategy & Race Physics ──────────────────────────────
DEFAULT_PIT_LOSS_SECONDS = 22.0
DEFAULT_FUEL_EFFECT_SEC_PER_LAP = 0.058  # Pooled median (0.0587 s/lap) across 64 compound-matched dry stint pairs (races >= 3 pairs)
MIN_FUEL_EFFECT_BOUND = 0.02             # Configurable plausible range minimum (default 0.02 s/lap)
MAX_FUEL_EFFECT_BOUND = 0.08             # Configurable plausible range maximum (default 0.08 s/lap)

# ─── Outlier Filtering (MAD Residuals) ───────────────────
OUTLIER_MAD_K = 3.5                       # Configurable robust MAD multiplier for stint residuals

# ─── Tyre Degradation & Cliff Detection ───────────────────
CLIFF_MIN_TYRE_AGE = 10                  # Minimum tyre age before a knot is allowed (default 10)
CLIFF_MIN_LAPS_PRE = 5                   # Minimum laps before candidate cliff knot
CLIFF_MIN_LAPS_POST = 4                  # Minimum laps after candidate cliff knot
CLIFF_SLOPE_MULTIPLIER = 1.6
DROP_YELLOW_LAPS_DEFAULT = False

# ─── Chart Theme ─────────────────────────────────────────
CHART_BG = "#1a1a2e"
CHART_PAPER_BG = "#0f0f1a"
CHART_FONT_COLOR = "#e0e0e0"
CHART_GRID_COLOR = "#2a2a3e"
CHART_ACCENT = "#e10600"
CHART_ACCENT_SECONDARY = "#ff6b35"
CHART_POSITIVE = "#00d97e"
CHART_NEGATIVE = "#e63757"
CHART_MUTED = "#6c757d"

# ─── Tyre Compound Colors ────────────────────────────────
COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC300",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#00AEEF",
    "UNKNOWN": "#888888",
    "TEST_UNKNOWN": "#888888",
}
