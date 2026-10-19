"""
F1 Race Intelligence 3.0 — Telemetry Lab Page
"""

import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pages.telemetry_lab import render_telemetry_lab  # noqa: E402

session_obj = st.session_state.get("session", None)
laps_data = st.session_state.get("laps", None)

render_telemetry_lab(session_obj, laps_data)
