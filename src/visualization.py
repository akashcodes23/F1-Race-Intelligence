"""
F1 Race Intelligence — Visualization Layer

Unified Plotly chart factory adhering to cinematic dark theme guidelines:
- High contrast, dark graphite background (#0f0f1a, #1a1a2e)
- Official F1 Red accent (#e10600) and secondary accents
- Clear tooltips, legends, and axis units
- Pure DataFrame/data inputs (no Streamlit or FastF1 coupled methods)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    CHART_ACCENT,
    CHART_ACCENT_SECONDARY,
    CHART_BG,
    CHART_FONT_COLOR,
    CHART_GRID_COLOR,
    CHART_NEGATIVE,
    CHART_PAPER_BG,
    CHART_POSITIVE,
    COMPOUND_COLORS,
    DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
)
from src.data_validator import clean_laps

logger = logging.getLogger(__name__)


def apply_chart_theme(fig: go.Figure, title: str = "", height: int = 480) -> go.Figure:
    """Apply unified F1 Race Intelligence dark theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, -apple-system, sans-serif", color=CHART_FONT_COLOR, size=12),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            x=0.03,
            y=0.96,
            font=dict(size=15, color="#ffffff"),
        ),
        height=height,
        xaxis=dict(
            gridcolor=CHART_GRID_COLOR,
            zerolinecolor=CHART_GRID_COLOR,
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=CHART_GRID_COLOR,
            zerolinecolor=CHART_GRID_COLOR,
            showgrid=True,
        ),
        margin=dict(l=55, r=35, t=65, b=50),
        hoverlabel=dict(
            bgcolor="#131322",
            bordercolor="#444",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#ffffff",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.98,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ─── 1. Driver Comparison ──────────────────────────────────

def compare_drivers_plot(laps: pd.DataFrame, driver1: str, driver2: str) -> go.Figure:
    """
    Lap-by-lap and cumulative lap-time delta between two drivers.
    Sign convention: delta = driver1 - driver2 (+ve = driver1 slower).
    """
    laps1 = clean_laps(laps[laps["Driver"] == driver1])
    laps2 = clean_laps(laps[laps["Driver"] == driver2])

    if laps1.empty or laps2.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for driver comparison",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=CHART_FONT_COLOR),
        )
        return apply_chart_theme(fig, f"{driver1} vs {driver2} — Delta Analysis")

    merged = laps1[["LapNumber", "LapTimeSec"]].merge(
        laps2[["LapNumber", "LapTimeSec"]],
        on="LapNumber",
        suffixes=(f"_{driver1}", f"_{driver2}"),
    ).sort_values("LapNumber")

    if merged.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No common laps found between drivers",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=CHART_FONT_COLOR),
        )
        return apply_chart_theme(fig, f"{driver1} vs {driver2} — Delta Analysis")

    merged["Delta"] = merged[f"LapTimeSec_{driver1}"] - merged[f"LapTimeSec_{driver2}"]
    merged["CumulativeDelta"] = merged["Delta"].cumsum()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Lap-by-Lap Delta (+ = {driver1} Slower, − = {driver1} Faster)",
            "Cumulative Delta (Race Gap Progression)",
        ),
    )

    # Lap delta trace
    fig.add_trace(
        go.Bar(
            x=merged["LapNumber"],
            y=merged["Delta"],
            name="Lap Delta",
            marker=dict(
                color=np.where(merged["Delta"] > 0, CHART_NEGATIVE, CHART_POSITIVE),
                opacity=0.85,
            ),
            hovertemplate="Lap %{x}<br>Delta: %{y:+.3f}s<extra></extra>",
        ),
        row=1, col=1,
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1, row=1, col=1)

    # Cumulative delta trace
    fig.add_trace(
        go.Scatter(
            x=merged["LapNumber"],
            y=merged["CumulativeDelta"],
            mode="lines+markers",
            name="Cumulative Delta",
            line=dict(color=CHART_ACCENT, width=2.5),
            marker=dict(size=4),
            hovertemplate="Lap %{x}<br>Cumulative Delta: %{y:+.3f}s<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1, row=2, col=1)

    fig = apply_chart_theme(fig, f"Head-to-Head Pace: {driver1} vs {driver2}", height=550)
    fig.update_xaxes(title_text="Lap Number", row=2, col=1)
    fig.update_yaxes(title_text="Delta (s)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative (s)", row=2, col=1)

    return fig


# ─── 2. Tyre Degradation Curve (Per-Stint Robust Huber + 95% CI) ───

def degradation_curve(
    laps_df: pd.DataFrame,
    driver_name: str,
    fuel_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    use_fuel_correction: bool = True,
) -> go.Figure:
    """
    Per-stint tyre degradation scatter chart with robust Huber fit lines,
    95% Confidence Interval ribbons, and tyre cliff markers.
    """
    from src.tyre_analysis import compute_stint_degradation

    cleaned = clean_laps(laps_df)

    if cleaned.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid laps for {driver_name}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=CHART_FONT_COLOR),
        )
        return apply_chart_theme(fig, f"{driver_name} — Tyre Degradation")

    fig = go.Figure()

    # Identify stints
    stints_to_plot = cleaned["Stint"].dropna().unique() if "Stint" in cleaned.columns else [1]

    for stint_idx in stints_to_plot:
        stint_data = cleaned[cleaned["Stint"] == stint_idx] if "Stint" in cleaned.columns else cleaned
        if len(stint_data) < 3:
            continue

        comp = stint_data["Compound"].iloc[0] if "Compound" in stint_data.columns else "UNKNOWN"
        color = COMPOUND_COLORS.get(str(comp).upper(), "#888888")

        res = compute_stint_degradation(
            stint_data,
            min_laps=4,
            fuel_effect=fuel_effect,
            use_fuel_correction=use_fuel_correction,
        )

        y_plot_col = "FuelCorrectedLapTimeSec" if (use_fuel_correction and "FuelCorrectedLapTimeSec" in res["cleaned_data"].columns) else "LapTimeSec"
        c_data = res["cleaned_data"]

        # 1. Scatter points for this stint
        fig.add_trace(
            go.Scatter(
                x=c_data["LapNumber"],
                y=c_data[y_plot_col],
                mode="markers",
                name=f"S{stint_idx}: {comp}",
                marker=dict(color=color, size=7, opacity=0.85, line=dict(width=1, color="#1a1a2e")),
                hovertemplate=(
                    f"<b>{driver_name} Stint {stint_idx}</b><br>"
                    f"Lap %{{x}} (Tyre Age %{{customdata}})<br>"
                    f"Pace: %{{y:.3f}}s<br>"
                    f"Compound: {comp}<extra></extra>"
                ),
                customdata=c_data.get("TyreLife", c_data.get("TireAge", np.arange(1, len(c_data) + 1))),
            )
        )

        # 2. Robust Fit Line and CI Ribbon if valid
        if res["valid"] and len(res["fit_x"]) > 1:
            # Map TyreLife grid to LapNumber space
            lap_start = c_data["LapNumber"].min()
            x_laps = lap_start + (res["fit_x"] - res["fit_x"].min())

            # CI upper bound
            fig.add_trace(
                go.Scatter(
                    x=x_laps,
                    y=res["ci_upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # CI lower bound (shaded area)
            fig.add_trace(
                go.Scatter(
                    x=x_laps,
                    y=res["ci_lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(255, 255, 255, 0.08)",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Huber Trendline
            fig.add_trace(
                go.Scatter(
                    x=x_laps,
                    y=res["fit_y"],
                    mode="lines",
                    name=f"S{stint_idx} Fit ({res['slope']:+.3f} s/l)",
                    line=dict(color=color, width=2.5, dash="solid"),
                    hovertemplate=f"S{stint_idx} Fit: %{{y:.3f}}s<br>Deg: {res['slope']:+.3f} s/lap (95% CI: [{res['slope_ci_lower']:+.3f}, {res['slope_ci_upper']:+.3f}])<extra></extra>",
                )
            )

            # 3. Tyre Cliff Marker if detected
            if res["cliff_detected"] and res["cliff_lap"] is not None:
                cliff_lap_num = lap_start + (res["cliff_lap"] - 1)
                fig.add_vline(
                    x=cliff_lap_num,
                    line_dash="dot",
                    line_color="#ff3333",
                    line_width=1.5,
                    annotation_text=f"S{stint_idx} Cliff (L{cliff_lap_num})",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color="#ff3333"),
                )

    fuel_label = f" (Fuel Corrected @ {fuel_effect:.3f} s/lap)" if use_fuel_correction else " (Raw)"
    fig = apply_chart_theme(fig, f"{driver_name} — Per-Stint Robust Degradation & Tyre Cliffs{fuel_label}")
    fig.update_xaxes(title_text="Lap Number")
    fig.update_yaxes(title_text="Lap Time (seconds)")
    return fig


# ─── 3. Position Progression ──────────────────────────────

def position_progression_plot(pos_df: pd.DataFrame, focus_drivers: Optional[List[str]] = None) -> go.Figure:
    """
    Position progression chart across race laps.
    """
    if pos_df is None or pos_df.empty:
        fig = go.Figure()
        return apply_chart_theme(fig, "Position Progression (No Data)")

    fig = go.Figure()
    all_drivers = pos_df["Driver"].unique()

    for drv in all_drivers:
        sub = pos_df[pos_df["Driver"] == drv].sort_values("LapNumber")
        is_focused = (focus_drivers is None) or (drv in focus_drivers)
        opacity = 1.0 if is_focused else 0.25
        width = 2.5 if is_focused else 1.0

        fig.add_trace(
            go.Scatter(
                x=sub["LapNumber"],
                y=sub["Position"],
                mode="lines",
                name=drv,
                line=dict(width=width),
                opacity=opacity,
                hovertemplate=f"<b>{drv}</b><br>Lap %{{x}}<br>Position: P%{{y}}<extra></extra>",
            )
        )

    fig = apply_chart_theme(fig, "Race Position Progression", height=500)
    fig.update_yaxes(autorange="reversed", title_text="Position", dtick=2)
    fig.update_xaxes(title_text="Lap Number")
    return fig


# ─── 4. Lap Time Distributions ────────────────────────────

def lap_time_distribution_plot(laps_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Boxplot distribution of clean race lap times across top drivers.
    """
    cleaned = clean_laps(laps_df)
    if cleaned.empty:
        fig = go.Figure()
        return apply_chart_theme(fig, "Lap Time Distribution (No Data)")

    # Sort drivers by median pace
    medians = cleaned.groupby("Driver")["LapTimeSec"].median().sort_values()
    selected_drivers = medians.head(top_n).index.tolist()

    sub = cleaned[cleaned["Driver"].isin(selected_drivers)].copy()

    fig = go.Figure()
    for drv in selected_drivers:
        drv_times = sub[sub["Driver"] == drv]["LapTimeSec"]
        fig.add_trace(
            go.Box(
                y=drv_times,
                name=drv,
                boxpoints="outliers",
                jitter=0.2,
                pointpos=-1.5,
                marker=dict(size=4),
                line=dict(width=1.5),
            )
        )

    fig = apply_chart_theme(fig, f"Pace Distribution — Top {len(selected_drivers)} Drivers (Clean Laps)")
    fig.update_xaxes(title_text="Driver")
    fig.update_yaxes(title_text="Lap Time (seconds)")
    return fig


# ─── 5. Stint Strategy Timeline ───────────────────────────

def stint_strategy_chart(stint_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal stacked/range bar chart showing stint breakdown and tyre compound per driver.
    """
    if stint_df is None or stint_df.empty:
        fig = go.Figure()
        return apply_chart_theme(fig, "Stint Strategy (No Data)")

    fig = go.Figure()

    drivers = sorted(stint_df["Driver"].unique(), reverse=True)

    for comp in stint_df["Compound"].unique():
        comp_sub = stint_df[stint_df["Compound"] == comp]
        color = COMPOUND_COLORS.get(str(comp).upper(), "#888888")

        for _, row in comp_sub.iterrows():
            fig.add_trace(
                go.Bar(
                    y=[row["Driver"]],
                    x=[row["EndLap"] - row["StartLap"] + 1],
                    base=[row["StartLap"] - 1],
                    orientation="h",
                    name=str(comp).upper(),
                    showlegend=False,
                    marker=dict(
                        color=color,
                        line=dict(color="#1a1a2e", width=1),
                    ),
                    hovertemplate=(
                        f"<b>{row['Driver']}</b> — Stint {row['Stint']}<br>"
                        f"Compound: {row['Compound']}<br>"
                        f"Laps: {row['StartLap']} → {row['EndLap']} ({row['TotalLaps']} laps)<br>"
                        f"Avg Pace: {row.get('AvgLapTimeFormatted', 'N/A')}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    # Add dummy traces for legend compounds
    seen_compounds = set(stint_df["Compound"].unique())
    for comp in seen_compounds:
        color = COMPOUND_COLORS.get(str(comp).upper(), "#888888")
        fig.add_trace(
            go.Bar(
                x=[0], y=[drivers[0] if drivers else ""],
                orientation="h",
                name=str(comp).upper(),
                showlegend=True,
                marker=dict(color=color),
            )
        )

    fig = apply_chart_theme(fig, "Session Tyre Stint Strategy Breakdown", height=max(400, len(drivers) * 26))
    fig.update_xaxes(title_text="Race Lap")
    fig.update_yaxes(title_text="")
    return fig


# ─── 6. Telemetry Traces Overlay ──────────────────────────

def telemetry_overlay_plot(aligned_tel: pd.DataFrame, driver1: str, driver2: str) -> go.Figure:
    """
    Multi-trace telemetry comparison: Speed overlay + Speed delta, Throttle %, Brake %.
    """
    if aligned_tel is None or aligned_tel.empty:
        fig = go.Figure()
        fig.add_annotation(text="Telemetry data unavailable for this lap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return apply_chart_theme(fig, "Telemetry Overlay (No Data)")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"Speed Profile (km/h) & Delta [+{driver1} / −{driver2}]",
            "Throttle Application (%)",
            "Braking Application",
        ),
        row_heights=[0.5, 0.25, 0.25],
    )

    dist = aligned_tel["Distance"]

    # Speed
    if "Speed_1" in aligned_tel.columns:
        fig.add_trace(
            go.Scatter(x=dist, y=aligned_tel["Speed_1"], name=f"{driver1} Speed", line=dict(color=CHART_ACCENT, width=2)),
            row=1, col=1,
        )
    if "Speed_2" in aligned_tel.columns:
        fig.add_trace(
            go.Scatter(x=dist, y=aligned_tel["Speed_2"], name=f"{driver2} Speed", line=dict(color=CHART_ACCENT_SECONDARY, width=2)),
            row=1, col=1,
        )

    # Throttle
    if "Throttle_1" in aligned_tel.columns:
        fig.add_trace(
            go.Scatter(x=dist, y=aligned_tel["Throttle_1"], name=f"{driver1} Throttle", line=dict(color=CHART_ACCENT, width=1.5)),
            row=2, col=1,
        )
    if "Throttle_2" in aligned_tel.columns:
        fig.add_trace(
            go.Scatter(x=dist, y=aligned_tel["Throttle_2"], name=f"{driver2} Throttle", line=dict(color=CHART_ACCENT_SECONDARY, width=1.5)),
            row=2, col=1,
        )

    # Brake
    if "Brake_1" in aligned_tel.columns:
        fig.add_trace(
            go.Scatter(x=dist, y=aligned_tel["Brake_1"], name=f"{driver1} Brake", line=dict(color=CHART_ACCENT, width=1.5)),
            row=3, col=1,
        )
    if "Brake_2" in aligned_tel.columns:
        fig.add_trace(
            go.Scatter(x=dist, y=aligned_tel["Brake_2"], name=f"{driver2} Brake", line=dict(color=CHART_ACCENT_SECONDARY, width=1.5)),
            row=3, col=1,
        )

    fig = apply_chart_theme(fig, f"Telemetry Overlay: {driver1} vs {driver2}", height=650)
    fig.update_xaxes(title_text="Lap Distance (meters)", row=3, col=1)
    fig.update_yaxes(title_text="km/h", row=1, col=1)
    fig.update_yaxes(title_text="Throttle %", row=2, col=1)
    fig.update_yaxes(title_text="Brake On/Off", row=3, col=1)
    return fig


# ─── 7. Circuit Map ───────────────────────────────────────

def circuit_map_plot(track_df: pd.DataFrame, circuit_name: str = "") -> go.Figure:
    """
    2D track layout colored by corner/straight speed.
    """
    if track_df is None or track_df.empty or "X" not in track_df.columns or "Y" not in track_df.columns:
        fig = go.Figure()
        return apply_chart_theme(fig, "Circuit Layout (No Data)")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=track_df["X"],
            y=track_df["Y"],
            mode="lines+markers",
            marker=dict(
                size=4,
                color=track_df["Speed"] if "Speed" in track_df.columns else "#e10600",
                colorscale="Viridis",
                colorbar=dict(title="Speed (km/h)", len=0.7),
                showscale=True if "Speed" in track_df.columns else False,
            ),
            line=dict(color="rgba(255,255,255,0.4)", width=2),
            hovertemplate="Speed: %{marker.color:.1f} km/h<br>Distance: %{customdata:.0f}m<extra></extra>",
            customdata=track_df["Distance"] if "Distance" in track_df.columns else np.zeros(len(track_df)),
        )
    )

    title = f"Circuit Layout & Speed Profile — {circuit_name}" if circuit_name else "Circuit Layout"
    fig = apply_chart_theme(fig, title, height=520)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
    )
    return fig


# ─── 8. Strategy Horizon & Optimizer ──────────────────────

def strategy_horizon_plot(sim_df: pd.DataFrame, pit_lap: int) -> go.Figure:
    """
    Cumulative projected race time comparison between Stay-Out and Pit-Now.
    """
    if sim_df is None or sim_df.empty:
        fig = go.Figure()
        return apply_chart_theme(fig, "Strategy Simulation (No Data)")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Projected Lap Times", "Cumulative Race Time (Horizon)"),
    )

    laps = sim_df["LapNumber"]

    fig.add_trace(
        go.Scatter(x=laps, y=sim_df["StayOut_LapTime"], mode="lines", name="Stay Out (Aging Tyres)", line=dict(color="#888888", width=2, dash="dash")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=laps, y=sim_df["Pit_LapTime"], mode="lines+markers", name=f"Pit Lap {pit_lap}", line=dict(color=CHART_ACCENT, width=2.5)),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=laps, y=sim_df["StayOut_CumTime"], mode="lines", name="Stay Out Cum Time", line=dict(color="#888888", width=2, dash="dash"), showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=laps, y=sim_df["Pit_CumTime"], mode="lines", name="Pit Cum Time", line=dict(color=CHART_ACCENT, width=2.5), showlegend=False),
        row=2, col=1,
    )

    fig = apply_chart_theme(fig, f"Strategy Projection: Pit Lap {pit_lap} vs Stay Out", height=540)
    fig.update_xaxes(title_text="Lap Number", row=2, col=1)
    fig.update_yaxes(title_text="Lap Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Total Seconds", row=2, col=1)
    return fig


def pit_window_optimizer_plot(eval_df: pd.DataFrame) -> go.Figure:
    """
    Net advantage vs candidate pit lap.
    """
    if eval_df is None or eval_df.empty:
        fig = go.Figure()
        return apply_chart_theme(fig, "Pit Window Evaluation (No Data)")

    fig = go.Figure()
    colors = np.where(eval_df["AdvantageVsStayOut"] > 0, CHART_POSITIVE, CHART_NEGATIVE)

    fig.add_trace(
        go.Bar(
            x=eval_df["PitLap"],
            y=eval_df["AdvantageVsStayOut"],
            marker=dict(color=colors),
            hovertemplate="Candidate Pit Lap %{x}<br>Advantage vs Stay-Out: %{y:+.2f}s<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)

    fig = apply_chart_theme(fig, "Pit Window Optimization: Advantage vs Stay-Out (Seconds Saved)")
    fig.update_xaxes(title_text="Candidate Pit Lap")
    fig.update_yaxes(title_text="Net Time Saved (seconds)")
    return fig


# ─── 9. Machine Learning Diagnostic Visuals ────────────────

def ml_prediction_vs_actual_plot(test_laps: np.ndarray, y_test: np.ndarray, test_preds: np.ndarray, model_name: str) -> go.Figure:
    """
    Test set actual vs predicted lap time over chronological race laps.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=test_laps, y=y_test, mode="markers+lines", name="Actual Lap Time", line=dict(color="#ffffff", width=1.5), marker=dict(size=6)),
    )
    fig.add_trace(
        go.Scatter(x=test_laps, y=test_preds, mode="markers+lines", name=f"{model_name} Prediction", line=dict(color=CHART_ACCENT, width=2, dash="dash"), marker=dict(size=6)),
    )

    fig = apply_chart_theme(fig, f"Held-Out Chronological Test: Actual vs Predicted ({model_name})")
    fig.update_xaxes(title_text="Lap Number")
    fig.update_yaxes(title_text="Lap Time (seconds)")
    return fig


def ml_residuals_plot(y_test: np.ndarray, residuals: np.ndarray, model_name: str) -> go.Figure:
    """
    Residual scatter plot (Residual = Actual - Predicted).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=residuals,
            mode="markers",
            marker=dict(color=CHART_ACCENT_SECONDARY, size=7, opacity=0.8),
            hovertemplate="Actual Time: %{x:.3f}s<br>Residual: %{y:+.3f}s<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)

    fig = apply_chart_theme(fig, f"Residual Diagnostics: {model_name} (y_test vs Error)")
    fig.update_xaxes(title_text="Actual Lap Time (s)")
    fig.update_yaxes(title_text="Residual Error (s)")
    return fig


def ml_feature_importance_plot(importances: Dict[str, float], model_name: str) -> go.Figure:
    """
    Horizontal bar chart of feature importances or coefficients.
    """
    if not importances:
        fig = go.Figure()
        return apply_chart_theme(fig, f"Feature Importance — {model_name} (None)")

    sorted_feats = sorted(importances.items(), key=lambda x: abs(x[1]))
    names = [x[0] for x in sorted_feats]
    values = [x[1] for x in sorted_feats]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker=dict(color=CHART_ACCENT),
            hovertemplate="Feature: %{y}<br>Weight / Importance: %{x:.4f}<extra></extra>",
        )
    )

    fig = apply_chart_theme(fig, f"Feature Importance / Influence: {model_name}")
    fig.update_xaxes(title_text="Relative Importance / Coefficient")
    fig.update_yaxes(title_text="Feature")
    return fig
