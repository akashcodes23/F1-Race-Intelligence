from src.metrics import clean_laps
import plotly.graph_objects as go
import numpy as np


# =====================================================
# DRIVER COMPARISON — LAP DELTA
# =====================================================
def compare_drivers_plot(laps, driver1, driver2):

    laps1 = clean_laps(laps.pick_driver(driver1))
    laps2 = clean_laps(laps.pick_driver(driver2))

    merged = laps1[['LapNumber', 'LapTimeSec']].merge(
        laps2[['LapNumber', 'LapTimeSec']],
        on='LapNumber',
        suffixes=(f'_{driver1}', f'_{driver2}')
    )

    merged['Delta'] = (
        merged[f'LapTimeSec_{driver1}']
        - merged[f'LapTimeSec_{driver2}']
    )

    merged['CumulativeDelta'] = merged['Delta'].cumsum()

    fig = go.Figure()

    # Zero reference line
    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
        line_color="gray"
    )

    # Lap-by-lap delta
    fig.add_trace(go.Scatter(
        x=merged['LapNumber'],
        y=merged['Delta'],
        mode='lines',
        name='Lap Delta',
        line=dict(width=3),
        hovertemplate=
        "<b>Lap %{x}</b><br>" +
        "Delta: %{y:.3f} sec<br>" +
        "<extra></extra>"
    ))

    # Cumulative delta
    fig.add_trace(go.Scatter(
        x=merged['LapNumber'],
        y=merged['CumulativeDelta'],
        mode='lines',
        name='Cumulative Delta',
        line=dict(width=2, dash='dash'),
        hovertemplate=
        "<b>Lap %{x}</b><br>" +
        "Cumulative: %{y:.3f} sec<br>" +
        "<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"{driver1} vs {driver2} — Lap Time Delta",
        title_x=0.5,
        height=500,
        xaxis_title="Lap Number",
        yaxis_title="Delta (Seconds)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig


# =====================================================
# TIRE DEGRADATION CURVE
# =====================================================
def degradation_curve(laps_df, driver_name):

    x = laps_df["LapNumber"]
    y = laps_df["LapTimeSec"]

    # Linear regression fit
    coef = np.polyfit(x, y, 1)
    trendline = np.poly1d(coef)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Lap Times",
        marker=dict(size=6)
    ))

    # Trend line
    fig.add_trace(go.Scatter(
        x=x,
        y=trendline(x),
        mode="lines",
        name="Degradation Trend",
        line=dict(width=3)
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"{driver_name} Tire Degradation Trend",
        title_x=0.5,
        height=450,
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (sec)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=70, b=40)
    )

    return fig