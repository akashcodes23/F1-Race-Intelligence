"""
F1 Race Intelligence 3.0 — Mission Control Pit-Wall Cockpit Page
"""

import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pages.race_cockpit import render_race_cockpit  # noqa: E402

session_obj = st.session_state.get("session", None)
laps_data = st.session_state.get("laps", None)
meta_info = st.session_state.get("meta", None)

render_race_cockpit(session_obj, laps_data, meta_info)
