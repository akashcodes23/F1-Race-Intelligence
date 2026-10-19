"""
F1 Race Intelligence 3.0 — Model Validation & Statistical Strategy Page
"""

import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pages.ml_lab import render_ml_lab  # noqa: E402

laps_data = st.session_state.get("laps", None)

render_ml_lab(laps_data)
