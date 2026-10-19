"""
F1 Race Intelligence — Unified Visualization Package

Exposes standard 2D telemetry, tyre, and strategy Plotly charts,
alongside 3D spatial WebGL track models and broadcast audio cues.
"""

from src.visualization.audio import (
    play_caution_warning,
    play_pit_radio_chime,
    render_audio_controller,
)
from src.visualization.charts import (
    apply_chart_theme,
    circuit_map_plot,
    compare_drivers_plot,
    degradation_curve,
    lap_time_distribution_plot,
    ml_feature_importance_plot,
    ml_prediction_vs_actual_plot,
    ml_residuals_plot,
    pit_window_optimizer_plot,
    position_progression_plot,
    stint_strategy_chart,
    strategy_horizon_plot,
    telemetry_overlay_plot,
)
from src.visualization.track_3d import (
    CIRCUIT_3D_WAYPOINTS,
    Track3DEngine,
)

__all__ = [
    "CIRCUIT_3D_WAYPOINTS",
    "Track3DEngine",
    "apply_chart_theme",
    "circuit_map_plot",
    "compare_drivers_plot",
    "degradation_curve",
    "lap_time_distribution_plot",
    "ml_feature_importance_plot",
    "ml_prediction_vs_actual_plot",
    "ml_residuals_plot",
    "pit_window_optimizer_plot",
    "play_caution_warning",
    "play_pit_radio_chime",
    "position_progression_plot",
    "render_audio_controller",
    "stint_strategy_chart",
    "strategy_horizon_plot",
    "telemetry_overlay_plot",
]
