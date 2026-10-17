"""
F1 Race Intelligence — Machine Learning Lab UI
"""

import pandas as pd
import streamlit as st

from src.ml_pipeline import train_and_evaluate_models
from src.visualization import (
    ml_feature_importance_plot,
    ml_prediction_vs_actual_plot,
    ml_residuals_plot,
)


def render_ml_lab(laps: pd.DataFrame, default_driver: str = "VER"):
    st.markdown("### 🧠 Machine Learning Lab: Lap Pace Modeling & Evaluation")
    st.markdown("""
    Evaluate predictive models trained strictly under **chronological split** constraints
    (preventing time-series leakage). Includes a **Mean Baseline** comparison, **Linear Regression**,
    and **Random Forest Regressor** with TimeSeriesSplit validation.
    """)

    if laps is None or laps.empty or "Driver" not in laps.columns:
        st.warning("⚠️ No valid lap data found for machine learning.")
        return

    available_drivers = sorted(laps["Driver"].dropna().unique())
    col_d, col_btn = st.columns([2, 1])
    with col_d:
        idx = available_drivers.index(default_driver) if default_driver in available_drivers else 0
        selected_driver = st.selectbox("Select Driver to Train ML Models", available_drivers, index=idx, key="ml_drv")

    drv_laps = laps[laps["Driver"] == selected_driver].copy()

    with st.spinner(f"Training chronological ML models for {selected_driver}..."):
        ml_results = train_and_evaluate_models(drv_laps)

    if not ml_results.get("success", False):
        st.warning(f"⚠️ {ml_results.get('message', 'Training failed.')}")
        return

    # ── Model Comparison Table ──
    st.markdown("#### 🏆 Benchmark Leaderboard (Held-Out Test Set)")

    baseline = ml_results["baseline"]
    models = ml_results["models"]

    leaderboard_rows = [
        {
            "Model": baseline["name"],
            "Test MAE (s)": f"{baseline['mae']:.3f}",
            "Test RMSE (s)": f"{baseline['rmse']:.3f}",
            "Test R²": f"{baseline['r2']:.3f}",
            "CV R² (Train)": "N/A",
        }
    ]

    for m_name, m_data in models.items():
        cv_txt = f"{m_data['cv_r2_mean']:.3f} ± {m_data['cv_r2_std']:.3f}" if m_data.get("cv_r2_mean") is not None else "N/A"
        leaderboard_rows.append({
            "Model": m_name,
            "Test MAE (s)": f"{m_data['test_mae']:.3f}",
            "Test RMSE (s)": f"{m_data['test_rmse']:.3f}",
            "Test R²": f"{m_data['test_r2']:.3f}",
            "CV R² (Train)": cv_txt,
        })

    st.dataframe(pd.DataFrame(leaderboard_rows), use_container_width=True, hide_index=True)

    # ── Inspection Tabs ──
    tab1, tab2, tab3 = st.tabs(["📈 Prediction vs Actual", "🎯 Residual Diagnostics", "⚖️ Feature Importance"])

    with tab1:
        chosen_model = st.selectbox("Select Model for Visual Inspection", list(models.keys()), key="ml_inspect_model")
        m_info = models[chosen_model]
        fig_pred = ml_prediction_vs_actual_plot(
            test_laps=m_info["test_laps"],
            y_test=m_info["y_test"],
            test_preds=m_info["test_predictions"],
            model_name=chosen_model,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        m_info = models[chosen_model]
        fig_res = ml_residuals_plot(
            y_test=m_info["y_test"],
            residuals=m_info["residuals"],
            model_name=chosen_model,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with tab3:
        m_info = models[chosen_model]
        fig_feat = ml_feature_importance_plot(
            importances=m_info["importances"],
            model_name=chosen_model,
        )
        st.plotly_chart(fig_feat, use_container_width=True)
