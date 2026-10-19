"""
F1 Race Intelligence 2.0 — Main Application Entry Point

Cinematic, technically rigorous Formula 1 analytics, tyre degradation modeling,
pit-wall strategy simulation, and machine learning decision support.
"""

import os
import sys

import streamlit as st

# Add project root and src/ to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Tuple

import requests

from config import CACHE_DIR, SEASONS
from pages.driver_comparison import render_driver_comparison
from pages.ml_lab import render_ml_lab

# Import Pages
from pages.race_overview import render_race_overview
from pages.season_explorer import render_season_explorer
from pages.strategy_simulator import render_strategy_simulator
from pages.telemetry_lab import render_telemetry_lab
from pages.tyre_intelligence import render_tyre_intelligence
from src.data_loader import (
    get_available_sessions,
    get_demo_races,
    get_event_names,
    init_cache,
    load_session,
)

# Initialize FastF1 Cache (Single point of initialization)
init_cache(CACHE_DIR)

# ─── Streamlit Page Configuration ─────────────────────────
st.set_page_config(
    page_title="F1 Race Intelligence 2.0",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Cinematic Dark CSS ────────────────────────────
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Main Container background */
    .stApp {
        background-color: #0c0d14;
        color: #e0e0e0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #12131f;
        border-right: 1px solid #1f2238;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161828 0%, #1a1c30 100%);
        border: 1px solid #282b47;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #e10600;
        transform: translateY(-2px);
    }

    /* F1 Red Accent Header */
    .f1-brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e10600;
        margin-bottom: 16px;
    }
    .f1-badge {
        background-color: #e10600;
        color: #ffffff;
        font-weight: 800;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 1px;
    }

    /* Dataframe table styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #23263e;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Primary buttons */
    .stButton>button {
        background: linear-gradient(135deg, #e10600 0%, #b80500 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff1a14 0%, #e10600 100%);
        box-shadow: 0 0 15px rgba(225, 6, 0, 0.4);
        border: none;
        color: #ffffff;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #141624;
        border: 1px solid #23263e;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #a0a0b0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1c1f33 !important;
        border-color: #e10600 !important;
        color: #ffffff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=120, show_spinner=False)
def check_live_api_status() -> Tuple[bool, str]:
    """Check reachability of public F1 archive endpoints."""
    try:
        resp = requests.get("https://ergast.com/api/f1/current.json", timeout=1.5)
        if resp.status_code == 200:
            return True, "Online (Archive Mirror)"
        return False, f"HTTP {resp.status_code}"
    except Exception:
        return False, "Offline / Unreachable"


# ─── Sidebar Configuration ────────────────────────────────
st.sidebar.markdown("""
<div class="f1-brand-header">
    <span class="f1-badge">PRO</span>
    <h2 style="margin:0; font-size:1.35rem; font-weight:800; letter-spacing:0.5px;">F1 INTELLIGENCE</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("#### 🏁 Session Selection")

demo_races_list = get_demo_races()
demo_options = [f"{y} {gp}" for (y, gp) in demo_races_list] if demo_races_list else ["2023 Monaco Grand Prix"]

source_mode = st.sidebar.radio(
    "Data Source Mode",
    ["📦 Bundled Demo (13 Offline GPs)", "🌐 Live FastF1 Exploration"],
    index=0,
    help="Bundled demo guarantees instant zero-network access to 13 verified Grand Prix events. Live mode queries external F1 feeds.",
)

if source_mode == "📦 Bundled Demo (13 Offline GPs)":
    default_demo_idx = 0
    for idx, opt in enumerate(demo_options):
        if "2023 Monaco" in opt:
            default_demo_idx = idx
            break
    selected_demo = st.sidebar.selectbox("Select Offline Grand Prix", demo_options, index=default_demo_idx)
    season = int(selected_demo.split()[0])
    grand_prix = " ".join(selected_demo.split()[1:])
    selected_session_name = "Race"
    selected_session_id = "R"
    load_telemetry_opt = False
else:
    default_season_idx = SEASONS.index(2023) if 2023 in SEASONS else 0
    season = st.sidebar.selectbox("Championship Season", SEASONS, index=default_season_idx)

    event_names = get_event_names(season)
    if not event_names:
        event_names = ["Monaco Grand Prix", "Bahrain Grand Prix", "British Grand Prix"]

    default_gp_idx = 0
    for idx, name in enumerate(event_names):
        if "Monaco" in name:
            default_gp_idx = idx
            break

    grand_prix = st.sidebar.selectbox("Grand Prix", event_names, index=default_gp_idx)

    available_sessions_map = get_available_sessions(season, grand_prix)
    session_labels = list(available_sessions_map.keys())
    default_session_idx = session_labels.index("Race") if "Race" in session_labels else len(session_labels) - 1

    selected_session_name = st.sidebar.selectbox("Session Type", session_labels, index=default_session_idx)
    selected_session_id = available_sessions_map.get(selected_session_name, selected_session_name)
    load_telemetry_opt = st.sidebar.checkbox("Include Telemetry Traces", value=True, help="Enables high-frequency Speed, Throttle, and Brake overlays")

# Load action button
btn_load = st.sidebar.button("⚡ Load Session Data", use_container_width=True)

# ─── Data Caching Helper ──────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_session_data(year: int, gp: str, session_id, with_tel: bool):
    return load_session(year, gp, session_id, load_telemetry=with_tel)


# Store in Session State
if "session" not in st.session_state:
    st.session_state["session"] = None
if "laps" not in st.session_state:
    st.session_state["laps"] = None
if "meta" not in st.session_state:
    st.session_state["meta"] = {}
if "has_loaded_initial" not in st.session_state:
    st.session_state["has_loaded_initial"] = False

if btn_load or not st.session_state["has_loaded_initial"]:
    st.session_state["has_loaded_initial"] = True
    with st.spinner(f"Loading {season} {grand_prix} ({selected_session_name})..."):
        session_obj, laps_df = fetch_session_data(season, grand_prix, selected_session_id, load_telemetry_opt)
        st.session_state["session"] = session_obj
        st.session_state["laps"] = laps_df
        st.session_state["meta"] = {
            "year": season,
            "gp": grand_prix,
            "session": selected_session_name,
        }

current_session = st.session_state["session"]
current_laps = st.session_state["laps"]
current_meta = st.session_state["meta"]

# Sidebar Connection & Data Status
st.sidebar.markdown("---")
st.sidebar.markdown("##### 🔌 Connection & Data Status")
is_live_reachable, reachability_msg = check_live_api_status()
reach_color = "#39b54a" if is_live_reachable else "#ffaa00"
st.sidebar.markdown(
    f"- **Live API Reachability:** <span style='color:{reach_color};font-weight:600;'>{reachability_msg}</span>",
    unsafe_allow_html=True,
)
if current_session is not None:
    src_label = "Live FastF1 Session Object"
    src_color = "#00d2be"
elif current_laps is not None and not current_laps.empty:
    src_label = "Bundled Demo (Offline)"
    src_color = "#39b54a"
else:
    src_label = "No Session Loaded"
    src_color = "#ff4444"

st.sidebar.markdown(
    f"- **Active Data Source:** <span style='color:{src_color};font-weight:600;'>{src_label}</span>",
    unsafe_allow_html=True,
)

# ─── Navigation Header ────────────────────────────────────
sections = [
    "🏁 Race Overview",
    "⚔️ Driver Comparison",
    "🛞 Tyre Intelligence",
    "⚡ Telemetry Lab",
    "⏱️ Strategy Simulator",
    "🧠 ML Lab",
    "📅 Season Explorer",
]

selected_section = st.radio(
    "Navigation",
    sections,
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

# Session Status Banner
if current_meta:
    st.markdown(
        f"**Active Session:** `{current_meta.get('year')}` **{current_meta.get('gp')}** — `{current_meta.get('session')}`"
    )

if current_laps is None or current_laps.empty:
    st.error(
        f"🚨 **FastF1 Data Source Unavailable:** Unable to load lap timing data for "
        f"`{current_meta.get('year')}` **{current_meta.get('gp')}** ({current_meta.get('session')}).\n\n"
        "**Troubleshooting & Data Source Status:**\n"
        "- Upstream Formula 1 live timing feeds require authentication credentials (HTTP 401 Unauthorized).\n"
        "- The application includes full offline coverage for 13 Grand Prix events (2021–2023) in `data/demo/demo_laps.parquet`.\n"
        "- Select any of the 13 supported Grand Prix events in the sidebar and click **⚡ Load Session Data**."
    )
elif current_session is None:
    st.caption("ℹ️ Loaded from offline demo dataset (`data/demo/demo_laps.parquet`)")

# ─── Router ───────────────────────────────────────────────
if selected_section == "🏁 Race Overview":
    render_race_overview(current_session, current_laps)

elif selected_section == "⚔️ Driver Comparison":
    render_driver_comparison(current_laps)

elif selected_section == "🛞 Tyre Intelligence":
    render_tyre_intelligence(current_laps)

elif selected_section == "⚡ Telemetry Lab":
    render_telemetry_lab(current_session, current_laps)

elif selected_section == "⏱️ Strategy Simulator":
    render_strategy_simulator(current_laps)

elif selected_section == "🧠 ML Lab":
    render_ml_lab(current_laps)

elif selected_section == "📅 Season Explorer":
    render_season_explorer(season)
