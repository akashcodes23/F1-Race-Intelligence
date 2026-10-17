"""
F1 Race Intelligence — Machine Learning Lab UI (Phase 2.5)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CHART_ACCENT, CHART_POSITIVE
from src.ml_pipeline import load_trained_model, prepare_ml_features, train_and_evaluate_models
from src.visualization import apply_chart_theme


def render_ml_lab(laps: pd.DataFrame, default_driver: str = "VER"):
    st.markdown("### 🧠 Machine Learning Lab: Lap Pace Modeling & LORO Evaluation")
    st.markdown("""
    Predictive lap-time modeling engine evaluating non-linear tyre wear dynamics, fuel burn-off,
    and track evolution across multi-season Grand Prix datasets. Evaluated strictly via
    **Leave-One-Race-Out (LORO)** cross-validation on laps $\\ge 7$ to guarantee zero target leakage.
    """)

    # ── Dataset Scope & Feature Disclaimer ──
    st.info(
        "ℹ️ **Dataset & Feature Scope (Phase 2.5):** Trained and validated across **13 Grand Prix events** "
        "(2021–2023, 10,637 clean laps evaluated on laps $\\ge 7$). The model relies strictly on causal physical features "
        "(tyre age, compound, laps remaining/fuel proxy, lap number, and strictly lagged previous pace). "
        "**Traffic, gap to car ahead, track/air temperature, driver identity, and team identity are NOT modeled.**"
    )

    # ── Load Pre-Trained Model Artifact ──
    artifact = load_trained_model()

    if artifact and "metadata" in artifact and "cv_results" in artifact["metadata"]:
        meta = artifact["metadata"]
        cv = meta["cv_results"]

        st.markdown("#### 🏆 Benchmark Leaderboard: Leave-One-Race-Out (LORO) Cross-Validation")
        st.caption(
            f"Evaluated on {cv.get('n_samples', 10637):,} clean dry laps (Laps $\\ge 7$) across "
            f"{cv.get('n_races', 13)} Grand Prix events (2021–2023). Target: Relative Lap Time (s) vs Driver Early Reference Pace."
        )

        pers_cv = cv.get("persistence_baseline", {})
        huber_cv = cv.get("huber_per_stint", {})
        s1_cv = cv.get("static_baseline_s1", {})
        s2_cv = cv.get("static_baseline_s2", {})
        hgbr_a = cv.get("gradient_boosting_A", {})
        hgbr_b = cv.get("gradient_boosting_B", {})
        hgbr_b_mono = cv.get("gradient_boosting_B_monotonic", {})

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            label="S1 Physics Linear",
            value=f"{s1_cv.get('mean_mae', 0.695):.3f}s",
            delta=f"RMSE: {s1_cv.get('mean_rmse', 0.957):.3f}s",
            delta_color="off",
        )
        m2.metric(
            label="S2 Scaled Ridge",
            value=f"{s2_cv.get('mean_mae', 0.718):.3f}s",
            delta=f"RMSE: {s2_cv.get('mean_rmse', 0.984):.3f}s",
            delta_color="off",
        )
        m3.metric(
            label="Huber Per-Stint (Phase 1)",
            value=f"{huber_cv.get('mean_mae', 0.424):.3f}s",
            delta=f"RMSE: {huber_cv.get('mean_rmse', 0.664):.3f}s",
        )
        m4.metric(
            label="Persistence Baseline",
            value=f"{pers_cv.get('mean_mae', 0.384):.3f}s",
            delta=f"RMSE: {pers_cv.get('mean_rmse', 0.626):.3f}s",
        )
        m5.metric(
            label="HGBR Variant B (Static)",
            value=f"{hgbr_b.get('mean_mae', 0.776):.3f}s",
            delta=f"Mono: {hgbr_b_mono.get('mean_mae', 0.771):.3f}s",
            delta_color="off",
        )
        m6.metric(
            label="HGBR Variant A (Champion)",
            value=f"{hgbr_a.get('mean_mae', 0.367):.3f}s",
            delta=f"RMSE: {hgbr_a.get('mean_rmse', 0.573):.3f}s (-4.4% MAE)",
            delta_color="normal",
        )

        # Honest comparison verdict & Phase 3 Selection
        st.info(
            f"⚖️ **Model Comparison & Phase 3 Strategy Engine Verdict (Decision Rule A4):**\\n"
            f"- **Single-Lap Autoregressive:** Variant A achieves lowest 1-step MAE (**{hgbr_a.get('mean_mae', 0.367):.3f}s**) beating Persistence ({pers_cv.get('mean_mae', 0.384):.3f}s).\\n"
            f"- **Long-Horizon Multi-Step ($H=10, 20$):** Static physics baseline $S_1$ achieves lower stint-total error than Variant B (9.77s vs 9.90s at $H=10$, and 18.33s vs 19.72s at $H=20$). "
            f"Moreover, Variant B's learned partial dependence severely underestimates physical fuel effect (+0.0094 s/lap vs empirical 0.058 s/lap). "
            f"**Phase 3 Strategy Simulator therefore uses $S_1$ (Physics Linear Baseline with empirical residuals) per Decision Rule A4.**"
        )

        # ── Inspection Tabs ──
        tab_eval, tab_horizons, tab_residuals, tab_live = st.tabs([
            "📊 Per-Race Held-Out Evaluation",
            "⏱️ Multi-Step Forecast Horizon",
            "🎯 Residual Breakdown Diagnostics",
            "⚡ Current Session Live Inference",
        ])

        with tab_eval:
            st.markdown("##### Held-Out Race Cross-Validation Performance (LORO)")
            per_race = cv.get("per_race_eval", [])
            if per_race:
                eval_df = pd.DataFrame(per_race)
                st.dataframe(
                    eval_df.rename(columns={
                        "RaceID": "Grand Prix Event",
                        "Laps": "Held-Out Laps",
                        "Naive_MAE": "Naive MAE",
                        "Persistence_MAE": "Persistence MAE",
                        "Huber_MAE": "Huber MAE",
                        "Ridge_MAE": "Ridge MAE",
                        "HGBR_A_MAE": "HGBR (A) MAE",
                        "HGBR_B_MAE": "HGBR (B) MAE",
                        "Persistence_RMSE": "Persistence RMSE",
                        "Ridge_RMSE": "Ridge RMSE",
                        "HGBR_A_RMSE": "HGBR (A) RMSE",
                        "HGBR_vs_Persistence_Gain": "Gain vs Pers (s)",
                        "HGBR_vs_Ridge_Gain": "Gain vs Ridge (s)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("###### Circuit-Specific Performance Nuances")
            st.caption(
                "On power-dominated circuits with DRS trains and low tyre wear (Monza: 2021 & 2022 Italian GP, "
                "Red Bull Ring: 2021 & 2022 Austrian GP), lap-to-lap variance is dominated by aerodynamic slipstreams "
                "and track position rather than compound degradation curves, so simple persistence or linear fits marginally edge out tree splits. "
                "On all technical, high-degradation multi-stint circuits (Bahrain, Spain, US GP, Silverstone, Paul Ricard), HGBR decisively outperforms all baselines."
            )

        with tab_horizons:
            st.markdown("##### Multi-Step Stint Forecast Horizon Evaluation (Multi-Origin Benchmark)")
            st.markdown(r"""
            Multi-step forecast accuracy when projecting lap times into the future across **multiple stint origins** (Tyre Life ~ 3, 8, and 15)
            across all eligible stints in the 13 held-out Grand Prix events:
            - **Persistence ($\hat{y}_{t+h} = y_0$):** Projects the last observed lap relative pace forward flatly.
            - **Huber (Prior Stint):** Fits a Huber linear model on laps before origin, projecting degradation into the future.
            - **$S_1$ (Physics Linear Baseline):** Reference pace + fuel burn-off + per-compound linear degradation.
            - **$S_2$ (Scaled Ridge on B):** `StandardScaler + Ridge` trained on static stint features without previous pace.
            - **Variant A (Recursive):** Feeds sequential predictions recursively into `PrevLapRelativePace`.
            - **Variant B (Static Unconstrained & Monotonic):** Static origin features (`TyreLife`, `Compound`, `LapsRemaining`, `LapNumber`).
            """)
            h_data = cv.get("horizon_evaluation", {})
            if h_data:
                rows = []
                for h_key, h_info in h_data.items():
                    h_val = h_info.get("horizon_laps", int(h_key.split("_")[1]) if "_" in h_key else 1)
                    models_dict = h_info.get("models", {})
                    for m_name, m_metrics in models_dict.items():
                        rows.append({
                            "Horizon (H)": f"H = {h_val} laps",
                            "Model": m_name,
                            "N": m_metrics.get("N", 0),
                            "MAE at H (s)": round(m_metrics.get("mae_at_h", 0.0), 3),
                            "Signed Cum Error (Bias, s)": f"{m_metrics.get('signed_cum_err', 0.0):+.3f}",
                            "Abs Cum Error (Stint Total, s)": round(m_metrics.get("abs_cum_err", 0.0), 3),
                        })
                if rows:
                    h_table_df = pd.DataFrame(rows)
                    st.dataframe(h_table_df, use_container_width=True, hide_index=True)

            st.markdown("""
            **Why Variant B's Curve Changes with Horizon ($H$) Even Though It Ignores the Origin:**
            Although Variant B excludes the origin pace ($y_0$) and evaluates lap times from static strategy parameters,
            as the horizon $H$ increases, its input features evolve:
            - $\\text{TyreLife}_{t+H} = \\text{TyreLife}_0 + H$ (tyre wear advances)
            - $\\text{LapsRemaining}_{t+H} = \\text{LapsRemaining}_0 - H$ (fuel weight burns off)
            - $\\text{LapNumber}_{t+H} = \\text{LapNumber}_0 + H$ (track grip evolves)
            Thus, Variant B projects a dynamic trajectory even without autoregressive memory.
            """)

            st.warning(
                "⚖️ **Decision Rule A4 Verdict for Phase 3:**\\n"
                "- Variant B fails to beat the static physics baseline $S_1$ on average stint-total absolute error at $H=10$ (9.90s vs 9.77s) and $H=20$ (19.72s vs 18.33s).\\n"
                "- Counterfactual partial-dependence analysis demonstrates that Variant B's learned fuel effect (+0.0094 s/lap) severely underestimates physical reality (0.058 s/lap).\\n"
                "**Decision:** Phase 3 Strategy Simulator will use **$S_1$ (Physics Linear Model with empirical residuals)**."
            )

        with tab_residuals:
            st.markdown("##### Residual Diagnostics with Exact Sample Sizes (N)")
            breakdown = cv.get("residual_breakdown", {})

            c_age, c_comp = st.columns(2)
            with c_age:
                st.markdown("**Residual Error by Tyre Age Band**")
                if "by_tyre_age" in breakdown:
                    age_df = pd.DataFrame(breakdown["by_tyre_age"]).T
                    age_df.index.name = "Tyre Age Band"
                    st.dataframe(age_df.rename(columns={"count": "Sample Count (N)", "mean": "Mean MAE (s)", "median": "Median MAE (s)"}), use_container_width=True)

            with c_comp:
                st.markdown("**Residual Error by Tyre Compound**")
                if "by_compound" in breakdown:
                    comp_df = pd.DataFrame(breakdown["by_compound"]).T
                    comp_df.index.name = "Compound"
                    st.dataframe(comp_df.rename(columns={"count": "Sample Count (N)", "mean": "Mean MAE (s)", "median": "Median MAE (s)"}), use_container_width=True)

            # Residual summary chart
            if "by_tyre_age" in breakdown:
                age_labels = list(breakdown["by_tyre_age"].keys())
                age_maes = [breakdown["by_tyre_age"][k]["mean"] for k in age_labels]
                fig_age = go.Figure()
                fig_age.add_trace(go.Bar(
                    x=age_labels,
                    y=age_maes,
                    marker=dict(color=CHART_ACCENT),
                    hovertemplate="Tyre Age: %{x}<br>Mean Absolute Error: %{y:.3f}s<extra></extra>",
                ))
                fig_age = apply_chart_theme(fig_age, "HGBR Mean Absolute Error by Tyre Age Band")
                fig_age.update_xaxes(title_text="Tyre Age Band")
                fig_age.update_yaxes(title_text="MAE (seconds)")
                st.plotly_chart(fig_age, use_container_width=True)

        with tab_live:
            _render_live_session_inference(laps, default_driver, artifact.get("model_variant_a", artifact["model"]))

    else:
        st.info("No pre-trained model artifact found. You can train on the current loaded session below.")
        _render_session_training(laps, default_driver)


def _render_live_session_inference(laps: pd.DataFrame, default_driver: str, model):
    st.markdown("##### Interactive Model Inference on Loaded Grand Prix Session")
    if laps is None or laps.empty or "Driver" not in laps.columns:
        st.warning("⚠️ No valid lap data loaded for current session.")
        return

    available_drivers = sorted(laps["Driver"].dropna().unique())
    idx = available_drivers.index(default_driver) if default_driver in available_drivers else 0
    selected_driver = st.selectbox("Select Driver for Model Inference", available_drivers, index=idx, key="ml_live_drv")

    drv_laps = laps[laps["Driver"] == selected_driver].copy()
    X, y, feature_names = prepare_ml_features(drv_laps, min_eval_lap=7, include_prev_pace=True)

    if X.empty or len(X) < 10:
        st.warning(f"⚠️ Insufficient clean dry laps ({len(X)}) for {selected_driver} on laps >= 7 to evaluate model.")
        return

    # Clean metadata columns
    X_clean = X[feature_names].copy()
    preds = model.predict(X_clean)

    session_mae = float(np.mean(np.abs(y.values - preds)))
    session_rmse = float(np.sqrt(np.mean((y.values - preds) ** 2)))

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Evaluated Laps (Laps >= 7)", len(y))
    col_m1.caption(f"Ref Pace: {X['_DriverReferencePaceSec'].iloc[0]:.3f}s (Laps 2-6)")
    col_m2.metric("Inference MAE", f"{session_mae:.3f}s")
    col_m3.metric("Inference RMSE", f"{session_rmse:.3f}s")

    # Plot Actual vs Predicted Relative Lap Time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_clean["LapNumber"],
        y=y.values,
        mode="markers+lines",
        name="Actual Relative Pace",
        line=dict(color="#ffffff", width=1.5),
        marker=dict(size=6),
        hovertemplate="Lap %{x}<br>Actual Delta: %{y:+.3f}s<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=X_clean["LapNumber"],
        y=preds,
        mode="markers+lines",
        name="HGBR Predicted Relative Pace",
        line=dict(color=CHART_POSITIVE, width=2, dash="dash"),
        marker=dict(size=6),
        hovertemplate="Lap %{x}<br>Predicted Delta: %{y:+.3f}s<extra></extra>",
    ))
    fig = apply_chart_theme(fig, f"{selected_driver} — Actual vs HGBR Predicted Relative Lap Pace (Laps >= 7)")
    fig.update_xaxes(title_text="Lap Number")
    fig.update_yaxes(title_text="Relative Lap Time Delta vs Reference Pace (s)")
    st.plotly_chart(fig, use_container_width=True)


def _render_session_training(laps: pd.DataFrame, default_driver: str):
    if laps is None or laps.empty or "Driver" not in laps.columns:
        st.warning("⚠️ No valid lap data found.")
        return

    available_drivers = sorted(laps["Driver"].dropna().unique())
    selected_driver = st.selectbox("Select Driver", available_drivers, key="ml_sess_drv")
    drv_laps = laps[laps["Driver"] == selected_driver].copy()

    if st.button("Train Models on Current Session"):
        with st.spinner("Training models..."):
            res = train_and_evaluate_models(drv_laps, save_artifact_path=None)
            if res.get("success"):
                st.success("Training complete!")
                st.json({
                    "Naive MAE": res["naive_baseline"]["mean_mae"],
                    "Persistence MAE": res["persistence_baseline"]["mean_mae"],
                    "Huber MAE": res["huber_per_stint"]["mean_mae"],
                    "Ridge MAE": res["scaled_ridge"]["mean_mae"],
                    "HGBR (Variant A) MAE": res["gradient_boosting_A"]["mean_mae"],
                    "HGBR (Variant B) MAE": res["gradient_boosting_B"]["mean_mae"],
                })
            else:
                st.error(res.get("message", "Training failed."))
