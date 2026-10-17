"""
F1 Race Intelligence — Season Explorer UI
"""

import pandas as pd
import streamlit as st

from src.data_loader import get_event_schedule


def render_season_explorer(selected_year: int):
    st.markdown(f"### 📅 Season Explorer: {selected_year} FIA Formula One World Championship")
    st.caption("Official race calendar, round metadata, and session formats.")

    with st.spinner(f"Loading calendar for {selected_year}..."):
        schedule = get_event_schedule(selected_year)

    if schedule.empty:
        st.warning(f"⚠️ Unable to load schedule for {selected_year}.")
        return

    cols = ["RoundNumber", "Country", "Location", "EventName", "EventDate", "EventFormat"]
    avail_cols = [c for c in cols if c in schedule.columns]

    df_display = schedule[avail_cols].copy()
    if "EventDate" in df_display.columns:
        df_display["EventDate"] = pd.to_datetime(df_display["EventDate"]).dt.strftime("%Y-%m-%d")

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.info("💡 To load and analyze any event, select the Grand Prix name from the left sidebar and click **Load Session**.")
