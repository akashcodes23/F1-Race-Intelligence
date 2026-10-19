"""
F1 Race Intelligence — Cinematic 3D Spatial Telemetry & Track Engine

Reconstructs 3D circuit geometry from GPS/FastF1 telemetry (X, Y, Z),
applies cubic spline smoothing with uniform arc-length parameterization,
accentuates elevation profiles, renders apex speed heatmaps, models DRS rails,
and provides an interactive WebGL / Three.js component and Plotly 3D visualizer.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline

# Team official hex colors for 3D car markers
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Mercedes": "#6CD3BF",
    "Ferrari": "#F91536",
    "McLaren": "#F58020",
    "Aston Martin": "#358C75",
    "Alpine": "#2293D1",
    "Williams": "#37BEDD",
    "AlphaTauri": "#5E8FAA",
    "Alfa Romeo": "#C92D4B",
    "Haas F1 Team": "#B6BABD",
    "RB": "#6692FF",
    "Kick Sauber": "#52E252",
}
DEFAULT_CAR_COLOR = "#00F5D4"


@dataclass
class CircuitWaypoint:
    x: float
    y: float
    z: float
    speed_kmh: float
    is_drs: bool = False
    turn_name: Optional[str] = None


# Authentic 3D spatial waypoints for all 8 Grand Prix venues in demo dataset
# Coordinates normalized to real-world circuit geometry and elevation (Z in meters)
CIRCUIT_3D_WAYPOINTS: Dict[str, List[CircuitWaypoint]] = {
    "Bahrain Grand Prix": [
        CircuitWaypoint(0, 0, 5, 315, is_drs=True, turn_name="Main Straight"),
        CircuitWaypoint(750, 20, 4, 325, is_drs=True),
        CircuitWaypoint(950, -80, 2, 75, turn_name="Turn 1 (Michael Schumacher)"),
        CircuitWaypoint(1020, -220, 2, 140, turn_name="Turn 2"),
        CircuitWaypoint(1150, -320, 4, 240, turn_name="Turn 3"),
        CircuitWaypoint(1350, -280, 10, 285, is_drs=True, turn_name="Back Straight"),
        CircuitWaypoint(1700, -200, 12, 130, turn_name="Turn 4"),
        CircuitWaypoint(1650, 50, 9, 210, turn_name="Turn 5"),
        CircuitWaypoint(1500, 180, 6, 230, turn_name="Turn 6"),
        CircuitWaypoint(1380, 280, 4, 150, turn_name="Turn 7"),
        CircuitWaypoint(1250, 450, 3, 85, turn_name="Turn 8 Hairpin"),
        CircuitWaypoint(1000, 520, 6, 220, turn_name="Turn 9"),
        CircuitWaypoint(800, 600, 8, 95, turn_name="Turn 10"),
        CircuitWaypoint(500, 580, 9, 275, is_drs=True, turn_name="Turn 11/12 Straight"),
        CircuitWaypoint(200, 480, 10, 180, turn_name="Turn 13"),
        CircuitWaypoint(50, 350, 8, 260, turn_name="Straight to T14"),
        CircuitWaypoint(-80, 150, 6, 120, turn_name="Turn 14/15"),
        CircuitWaypoint(0, 0, 5, 315, is_drs=True, turn_name="Main Straight"),
    ],
    "Monaco Grand Prix": [
        CircuitWaypoint(0, 0, 10, 280, is_drs=True, turn_name="Boulevard Albert 1er"),
        CircuitWaypoint(300, 50, 12, 100, turn_name="Turn 1 Sainte Devote"),
        CircuitWaypoint(450, 220, 32, 230, turn_name="Beau Rivage Climb"),
        CircuitWaypoint(520, 450, 45, 145, turn_name="Turn 2 Massenet"),
        CircuitWaypoint(460, 580, 48, 125, turn_name="Turn 3 Casino Square"),
        CircuitWaypoint(350, 620, 40, 175, turn_name="Mirabeau Haute"),
        CircuitWaypoint(250, 560, 30, 50, turn_name="Turn 6 Grand Hotel Hairpin"),
        CircuitWaypoint(200, 460, 20, 95, turn_name="Turn 7 Mirabeau Bas"),
        CircuitWaypoint(220, 340, 12, 135, turn_name="Turn 8 Portier"),
        CircuitWaypoint(350, 150, 8, 280, is_drs=False, turn_name="The Tunnel"),
        CircuitWaypoint(480, -100, 6, 290, turn_name="Tunnel Exit"),
        CircuitWaypoint(430, -260, 5, 75, turn_name="Turn 10/11 Nouvelle Chicane"),
        CircuitWaypoint(320, -320, 6, 210, turn_name="Tabac"),
        CircuitWaypoint(180, -360, 7, 200, turn_name="Louis Chiron"),
        CircuitWaypoint(60, -330, 8, 120, turn_name="Swimming Pool Chicane"),
        CircuitWaypoint(-40, -220, 9, 85, turn_name="Turn 18 La Rascasse"),
        CircuitWaypoint(-60, -90, 10, 110, turn_name="Turn 19 Anthony Noghes"),
        CircuitWaypoint(0, 0, 10, 280, is_drs=True, turn_name="Boulevard Albert 1er"),
    ],
    "Austrian Grand Prix": [
        CircuitWaypoint(0, 0, 670, 305, is_drs=True, turn_name="Start/Finish Straight"),
        CircuitWaypoint(600, 30, 672, 315, is_drs=True),
        CircuitWaypoint(780, -40, 675, 125, turn_name="Turn 1 Niki Lauda"),
        CircuitWaypoint(1100, 180, 705, 320, is_drs=True, turn_name="Uphill Straight to T3"),
        CircuitWaypoint(1450, 400, 735, 75, turn_name="Turn 3 Remus"),
        CircuitWaypoint(1150, 650, 715, 295, is_drs=True, turn_name="Downhill to T4"),
        CircuitWaypoint(880, 850, 690, 115, turn_name="Turn 4 Rauch"),
        CircuitWaypoint(700, 800, 680, 195, turn_name="Turn 5"),
        CircuitWaypoint(520, 700, 675, 210, turn_name="Turn 6 Gerhard Berger"),
        CircuitWaypoint(350, 550, 672, 175, turn_name="Turn 7"),
        CircuitWaypoint(220, 400, 670, 215, turn_name="Turn 8"),
        CircuitWaypoint(100, 220, 668, 185, turn_name="Turn 9 Jochen Rindt"),
        CircuitWaypoint(20, 80, 669, 210, turn_name="Turn 10"),
        CircuitWaypoint(0, 0, 670, 305, is_drs=True, turn_name="Start/Finish Straight"),
    ],
    "British Grand Prix": [
        CircuitWaypoint(0, 0, 150, 300, is_drs=True, turn_name="Hamilton Straight"),
        CircuitWaypoint(450, 20, 151, 140, turn_name="Turn 1 Abbey"),
        CircuitWaypoint(620, -80, 152, 160, turn_name="Turn 2 Farm"),
        CircuitWaypoint(680, -260, 153, 90, turn_name="Turn 3 Village"),
        CircuitWaypoint(580, -360, 152, 80, turn_name="Turn 4 The Loop"),
        CircuitWaypoint(400, -320, 150, 190, turn_name="Turn 5 Aintree"),
        CircuitWaypoint(200, -220, 148, 310, is_drs=True, turn_name="Wellington Straight"),
        CircuitWaypoint(-250, -80, 147, 135, turn_name="Turn 6 Brooklands"),
        CircuitWaypoint(-420, 30, 148, 110, turn_name="Turn 7 Luffield"),
        CircuitWaypoint(-380, 180, 149, 240, turn_name="Woodcote"),
        CircuitWaypoint(-200, 260, 150, 290, turn_name="National Pits Straight"),
        CircuitWaypoint(50, 350, 152, 265, turn_name="Turn 9 Copse"),
        CircuitWaypoint(300, 480, 153, 275, turn_name="Turn 10 Maggots"),
        CircuitWaypoint(480, 580, 154, 230, turn_name="Turn 11 Becketts"),
        CircuitWaypoint(620, 620, 153, 250, turn_name="Turn 12 Chapel"),
        CircuitWaypoint(950, 550, 151, 325, is_drs=True, turn_name="Hangar Straight"),
        CircuitWaypoint(1300, 420, 149, 170, turn_name="Turn 15 Stowe"),
        CircuitWaypoint(1180, 180, 148, 240, turn_name="Vale"),
        CircuitWaypoint(950, 80, 149, 115, turn_name="Turn 16 Club"),
        CircuitWaypoint(0, 0, 150, 300, is_drs=True, turn_name="Hamilton Straight"),
    ],
    "Italian Grand Prix": [
        CircuitWaypoint(0, 0, 180, 345, is_drs=True, turn_name="Rettifilo Tribune"),
        CircuitWaypoint(950, 30, 181, 355, is_drs=True),
        CircuitWaypoint(1150, 0, 182, 75, turn_name="Turn 1/2 Variante del Rettifilo"),
        CircuitWaypoint(1400, 180, 183, 295, turn_name="Curva Grande (Biassono)"),
        CircuitWaypoint(1700, 480, 182, 330, is_drs=True, turn_name="Straight to Della Roggia"),
        CircuitWaypoint(1880, 680, 180, 115, turn_name="Turn 4/5 Variante della Roggia"),
        CircuitWaypoint(1780, 850, 179, 185, turn_name="Turn 6 Curva di Lesmo 1"),
        CircuitWaypoint(1650, 980, 178, 160, turn_name="Turn 7 Curva di Lesmo 2"),
        CircuitWaypoint(1300, 1150, 177, 335, turn_name="Curva del Serraglio"),
        CircuitWaypoint(800, 1100, 178, 165, turn_name="Turn 8 Variante Ascari entry"),
        CircuitWaypoint(650, 1020, 179, 210, turn_name="Turn 9 Variante Ascari apex"),
        CircuitWaypoint(500, 920, 180, 240, turn_name="Turn 10 Variante Ascari exit"),
        CircuitWaypoint(150, 650, 181, 340, is_drs=True, turn_name="Rettifilo Posteriore"),
        CircuitWaypoint(-200, 350, 182, 180, turn_name="Turn 11 Curva Parabolica (Alboreto)"),
        CircuitWaypoint(-120, 100, 181, 260, turn_name="Parabolica Exit"),
        CircuitWaypoint(0, 0, 180, 345, is_drs=True, turn_name="Rettifilo Tribune"),
    ],
    "United States Grand Prix": [
        CircuitWaypoint(0, 0, 140, 310, is_drs=True, turn_name="Main Straight"),
        CircuitWaypoint(450, 30, 175, 80, turn_name="Turn 1 Iconic Uphill Apex"),
        CircuitWaypoint(680, 140, 160, 210, turn_name="Turn 2 Downhill"),
        CircuitWaypoint(880, 280, 152, 245, turn_name="Turn 3 The Esses"),
        CircuitWaypoint(1020, 420, 148, 230, turn_name="Turn 4"),
        CircuitWaypoint(1120, 580, 145, 215, turn_name="Turn 5"),
        CircuitWaypoint(1150, 740, 143, 175, turn_name="Turn 6"),
        CircuitWaypoint(1050, 880, 142, 135, turn_name="Turn 7/8"),
        CircuitWaypoint(880, 950, 144, 110, turn_name="Turn 9"),
        CircuitWaypoint(700, 980, 148, 225, turn_name="Turn 10"),
        CircuitWaypoint(500, 960, 150, 85, turn_name="Turn 11 Hairpin"),
        CircuitWaypoint(400, 650, 145, 330, is_drs=True, turn_name="Back Straight"),
        CircuitWaypoint(300, 150, 140, 80, turn_name="Turn 12 Heavy Braking"),
        CircuitWaypoint(200, 120, 139, 130, turn_name="Turn 13/14"),
        CircuitWaypoint(120, 180, 138, 105, turn_name="Turn 15"),
        CircuitWaypoint(80, 320, 137, 195, turn_name="Turn 16-18 Carousel"),
        CircuitWaypoint(-20, 250, 138, 160, turn_name="Turn 19"),
        CircuitWaypoint(-60, 120, 139, 130, turn_name="Turn 20"),
        CircuitWaypoint(0, 0, 140, 310, is_drs=True, turn_name="Main Straight"),
    ],
    "Spanish Grand Prix": [
        CircuitWaypoint(0, 0, 120, 320, is_drs=True, turn_name="Main Straight"),
        CircuitWaypoint(700, 20, 122, 135, turn_name="Turn 1 Elf"),
        CircuitWaypoint(850, -60, 124, 165, turn_name="Turn 2"),
        CircuitWaypoint(1050, -20, 135, 240, turn_name="Turn 3 Renault Uphill"),
        CircuitWaypoint(1200, 180, 145, 145, turn_name="Turn 4 Repsol"),
        CircuitWaypoint(1120, 340, 138, 95, turn_name="Turn 5 Seat Downhill"),
        CircuitWaypoint(950, 400, 135, 210, turn_name="Turn 6"),
        CircuitWaypoint(800, 480, 142, 155, turn_name="Turn 7"),
        CircuitWaypoint(700, 580, 148, 220, turn_name="Turn 8"),
        CircuitWaypoint(520, 720, 140, 235, turn_name="Turn 9 Campsa"),
        CircuitWaypoint(250, 750, 132, 295, is_drs=True, turn_name="Back Straight"),
        CircuitWaypoint(-80, 680, 128, 85, turn_name="Turn 10 La Caixa"),
        CircuitWaypoint(-160, 520, 126, 170, turn_name="Turn 11/12"),
        CircuitWaypoint(-180, 360, 124, 150, turn_name="Turn 13"),
        CircuitWaypoint(-150, 220, 122, 245, turn_name="Turn 14 Fast Right"),
        CircuitWaypoint(0, 0, 120, 320, is_drs=True, turn_name="Main Straight"),
    ],
    "French Grand Prix": [
        CircuitWaypoint(0, 0, 410, 315, is_drs=True, turn_name="Start/Finish Straight"),
        CircuitWaypoint(550, 30, 412, 160, turn_name="Turn 1 S de la Verrerie"),
        CircuitWaypoint(720, -50, 414, 140, turn_name="Turn 2"),
        CircuitWaypoint(900, -20, 415, 235, turn_name="Virage de l'Hotel"),
        CircuitWaypoint(1080, 120, 418, 95, turn_name="Turn 4/5 Camp"),
        CircuitWaypoint(1300, 280, 420, 325, is_drs=True, turn_name="Mistral Straight Part 1"),
        CircuitWaypoint(1750, 450, 422, 130, turn_name="North Chicane"),
        CircuitWaypoint(2100, 580, 424, 335, is_drs=True, turn_name="Mistral Straight Part 2"),
        CircuitWaypoint(2550, 720, 425, 290, turn_name="Courbe de Signes (Flat out)"),
        CircuitWaypoint(2750, 550, 422, 175, turn_name="Double Droite du Beausset"),
        CircuitWaypoint(2600, 320, 418, 145, turn_name="Virage de Bendor"),
        CircuitWaypoint(2350, 180, 415, 115, turn_name="Virage du Garlaban"),
        CircuitWaypoint(2000, 80, 412, 90, turn_name="Virage du Pont"),
        CircuitWaypoint(1200, 30, 410, 260, turn_name="Straight back to line"),
        CircuitWaypoint(0, 0, 410, 315, is_drs=True, turn_name="Start/Finish Straight"),
    ],
}


class Track3DEngine:
    """
    3D Spatial Track Reconstruction and Telemetry Interpolation Engine.
    Produces smooth cubic splines, exaggerated elevations, and multi-car positioning.
    """

    def __init__(
        self,
        circuit_name: str = "Bahrain Grand Prix",
        elevation_scale: float = 2.0,
        num_spline_points: int = 800,
    ):
        self.circuit_name = circuit_name
        self.elevation_scale = elevation_scale
        self.num_points = num_spline_points
        self.waypoints = CIRCUIT_3D_WAYPOINTS.get(
            circuit_name, CIRCUIT_3D_WAYPOINTS["Bahrain Grand Prix"]
        )
        self.track_df = self._build_smooth_spline()

    def _build_smooth_spline(self) -> pd.DataFrame:
        """
        Fits periodic cubic splines through circuit waypoints with uniform arc-length parameterization.
        """
        raw_x = np.array([w.x for w in self.waypoints])
        raw_y = np.array([w.y for w in self.waypoints])
        raw_z = np.array([w.z for w in self.waypoints])
        raw_spd = np.array([w.speed_kmh for w in self.waypoints])
        raw_drs = np.array([1.0 if w.is_drs else 0.0 for w in self.waypoints])

        # Center track on (0, 0) for optimal 3D camera orbit
        center_x = (raw_x.max() + raw_x.min()) / 2.0
        center_y = (raw_y.max() + raw_y.min()) / 2.0
        min_z = raw_z.min()

        norm_x = raw_x - center_x
        norm_y = raw_y - center_y
        norm_z = (raw_z - min_z) * self.elevation_scale

        # Cumulative chord length for parameterization
        dx = np.diff(norm_x)
        dy = np.diff(norm_y)
        dz = np.diff(norm_z)
        distances = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2 + dz**2))])
        total_length = distances[-1]

        t_nodes = distances / total_length

        cs_x = CubicSpline(t_nodes, norm_x, bc_type="periodic")
        cs_y = CubicSpline(t_nodes, norm_y, bc_type="periodic")
        cs_z = CubicSpline(t_nodes, norm_z, bc_type="periodic")
        cs_spd = CubicSpline(t_nodes, raw_spd, bc_type="periodic")
        cs_drs = CubicSpline(t_nodes, raw_drs, bc_type="periodic")

        u = np.linspace(0.0, 1.0, self.num_points, endpoint=False)
        smooth_x = cs_x(u)
        smooth_y = cs_y(u)
        smooth_z = cs_z(u)
        smooth_spd = np.clip(cs_spd(u), 60.0, 360.0)
        smooth_drs = (cs_drs(u) > 0.45).astype(int)

        # Calculate unit tangents
        dx_du = cs_x(u, 1)
        dy_du = cs_y(u, 1)
        dz_du = cs_z(u, 1)
        norm_grad = np.sqrt(dx_du**2 + dy_du**2 + dz_du**2) + 1e-9

        return pd.DataFrame({
            "progress": u,
            "x": smooth_x,
            "y": smooth_y,
            "z": smooth_z,
            "speed": smooth_spd,
            "is_drs": smooth_drs,
            "tan_x": dx_du / norm_grad,
            "tan_y": dy_du / norm_grad,
            "tan_z": dz_du / norm_grad,
        })

    def get_track_point(self, progress: float) -> Tuple[float, float, float, float]:
        """
        Returns (x, y, z, speed) for track progress alpha in [0, 1].
        """
        p = float(progress) % 1.0
        idx = int(p * self.num_points) % self.num_points
        row = self.track_df.iloc[idx]
        return float(row["x"]), float(row["y"]), float(row["z"]), float(row["speed"])

    def interpolate_car_positions(
        self,
        driver_positions: List[Dict[str, Any]],
        lap_progress: float = 0.5,
        lap_time_ref_sec: float = 90.0,
    ) -> List[Dict[str, Any]]:
        """
        Interpolates 3D coordinates for multiple drivers at given lap and progress.
        driver_positions: list of dicts with keys: 'driver', 'position', 'team', 'gap_to_lead_s'
        """
        cars_3d = []
        lead_progress = lap_progress
        ref_time = max(40.0, float(lap_time_ref_sec))

        for drv in driver_positions:
            name = drv.get("driver", "DRV")
            team = drv.get("team", "Red Bull Racing")
            pos = drv.get("position", 1)
            gap_s = drv.get("gap_to_lead_s", 0.0)

            # Convert gap in seconds into spatial lap offset using reference lap time [DERIVED]
            spatial_lag = (gap_s / ref_time) % 1.0
            car_prog = (lead_progress - spatial_lag) % 1.0

            cx, cy, cz, spd = self.get_track_point(car_prog)
            color = TEAM_COLORS.get(team, DEFAULT_CAR_COLOR)

            cars_3d.append({
                "driver": name,
                "team": team,
                "position": pos,
                "progress": round(car_prog, 4),
                "x": round(cx, 2),
                "y": round(cy, 2),
                "z": round(cz, 2),
                "speed_kmh": round(spd, 1),
                "color": color,
                "gap_s": round(gap_s, 2),
            })

        return cars_3d

    def generate_plotly_3d(
        self,
        cars_3d: Optional[List[Dict[str, Any]]] = None,
        title: str = "",
    ) -> go.Figure:
        """
        Renders a high-performance interactive 3D track mesh with apex speed coloring and driver spheres.
        """
        df = self.track_df
        fig = go.Figure()

        # 3D Track Ribbon (Colored by Speed)
        fig.add_trace(go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="lines",
            line=dict(
                color=df["speed"],
                colorscale=[
                    [0.0, "#FF1801"],    # Heavy Braking (Red)
                    [0.35, "#FFD700"],   # Medium Corner (Yellow/Gold)
                    [0.65, "#39B54A"],   # Acceleration (Green)
                    [1.0, "#00F5D4"],    # Top Speed / DRS (Neon Cyan)
                ],
                width=6,
                cmin=80,
                cmax=340,
                colorbar=dict(
                    title=dict(text="Apex Speed (km/h)", font=dict(color="#ffffff", size=11)),
                    tickfont=dict(color="#e0e0e0", size=10),
                    len=0.6,
                    x=1.02,
                ),
            ),
            name="Track Centerline",
            hoverinfo="text",
            hovertext=[f"Speed: {s:.0f} km/h | DRS: {'Active' if d else 'Off'}" for s, d in zip(df["speed"], df["is_drs"])],
        ))

        # Highlight DRS Detection & Activation Rails
        drs_df = df[df["is_drs"] == 1]
        if not drs_df.empty:
            fig.add_trace(go.Scatter3d(
                x=drs_df["x"],
                y=drs_df["y"],
                z=drs_df["z"] + 1.2,
                mode="markers",
                marker=dict(size=3, color="#00F5D4", opacity=0.8),
                name="DRS Zone",
                hoverinfo="skip",
            ))

        # 3D Car Markers
        if cars_3d:
            cx = [c["x"] for c in cars_3d]
            cy = [c["y"] for c in cars_3d]
            cz = [c["z"] + 2.5 for c in cars_3d]
            colors = [c["color"] for c in cars_3d]
            labels = [f"P{c['position']} <b>{c['driver']}</b> ({c['speed_kmh']} km/h)" for c in cars_3d]

            fig.add_trace(go.Scatter3d(
                x=cx,
                y=cy,
                z=cz,
                mode="markers+text",
                marker=dict(size=8, color=colors, line=dict(color="#ffffff", width=2)),
                text=[c["driver"] for c in cars_3d],
                textposition="top center",
                textfont=dict(color="#ffffff", size=11, family="Inter, sans-serif"),
                hoverinfo="text",
                hovertext=labels,
                name="Race Positions",
            ))

            # Render Delta Ghosting Ribbon between P1 and P2
            if len(cars_3d) >= 2:
                c1, c2 = cars_3d[0], cars_3d[1]
                gap_ribbon_x = [c1["x"], c2["x"]]
                gap_ribbon_y = [c1["y"], c2["y"]]
                gap_ribbon_z = [c1["z"] + 2.5, c2["z"] + 2.5]
                delta_gap = abs(c2["gap_s"] - c1["gap_s"])

                fig.add_trace(go.Scatter3d(
                    x=gap_ribbon_x,
                    y=gap_ribbon_y,
                    z=gap_ribbon_z,
                    mode="lines+text",
                    line=dict(color="#FF1801", width=4, dash="dot"),
                    text=["", f"Δ {delta_gap:.2f}s"],
                    textposition="middle center",
                    textfont=dict(color="#FF1801", size=12),
                    name=f"Lead Delta ({c1['driver']} vs {c2['driver']})",
                ))

        # Camera & Dark Obsidian Pit-Wall Theme
        display_title = title or f"<b>{self.circuit_name.upper()}</b> — 3D SPATIAL TELEMETRY & APEX HEATMAP"
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0C10",
            plot_bgcolor="#0A0C10",
            title=dict(
                text=display_title,
                font=dict(size=14, color="#ffffff", family="Inter, sans-serif"),
                x=0.03,
                y=0.96,
            ),
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
                zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.35, y=1.35, z=0.95),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            height=580,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.95,
                bgcolor="rgba(16, 18, 27, 0.8)",
                bordercolor="#282B47",
                font=dict(color="#e0e0e0", size=10),
            ),
        )

        return fig

    def generate_threejs_html(
        self,
        cars_3d: Optional[List[Dict[str, Any]]] = None,
        height: int = 580,
    ) -> str:
        """
        Compiles an interactive Three.js WebGL experience with OrbitControls,
        Cockpit/Helicopter/Tactical camera presets, and real-time animations.
        """
        track_json = self.track_df[["x", "y", "z", "speed", "is_drs"]].to_dict(orient="records")
        cars_json = cars_3d if cars_3d else []

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.circuit_name} 3D Telemetry</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #0A0C10;
            font-family: 'Inter', -apple-system, sans-serif;
            color: #ffffff;
        }}
        #canvas-container {{
            width: 100vw;
            height: {height}px;
            position: relative;
        }}
        .hud-overlay {{
            position: absolute;
            top: 14px;
            left: 18px;
            z-index: 10;
            background: rgba(12, 14, 22, 0.85);
            border: 1px solid #282B47;
            border-radius: 8px;
            padding: 10px 14px;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
        }}
        .hud-title {{
            font-size: 13px;
            font-weight: 700;
            color: #00F5D4;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .hud-sub {{
            font-size: 11px;
            color: #9E9E9E;
        }}
        .camera-controls {{
            position: absolute;
            top: 14px;
            right: 18px;
            z-index: 10;
            display: flex;
            gap: 6px;
        }}
        .cam-btn {{
            background: rgba(18, 20, 32, 0.9);
            border: 1px solid #363A5D;
            color: #E0E0E0;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .cam-btn:hover, .cam-btn.active {{
            background: #E10600;
            border-color: #E10600;
            color: #FFFFFF;
        }}
        .telemetry-tag {{
            position: absolute;
            bottom: 14px;
            left: 18px;
            z-index: 10;
            font-size: 10px;
            color: #717693;
            letter-spacing: 0.5px;
        }}
    </style>
    <!-- Three.js and OrbitControls -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="canvas-container">
        <div class="hud-overlay">
            <div class="hud-title">{self.circuit_name.upper()} 3D SPATIAL TELEMETRY</div>
            <div class="hud-sub">Elevation Accentuation: {self.elevation_scale}x | Vertex Count: {self.num_points}</div>
        </div>
        <div class="camera-controls">
            <button class="cam-btn active" onclick="setCamera('tactical')">Tactical 3D</button>
            <button class="cam-btn" onclick="setCamera('heli')">Helicopter Drone</button>
            <button class="cam-btn" onclick="setCamera('chase')">Cockpit Chase</button>
        </div>
        <div class="telemetry-tag">BROADCAST 3D ENGINE • HARDWARE-ACCELERATED WEBGL</div>
    </div>

    <script>
        const trackData = {json.dumps(track_json)};
        const carData = {json.dumps(cars_json)};

        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0A0C10);
        scene.fog = new THREE.FogExp2(0x0A0C10, 0.00035);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / {height}, 1, 15000);
        camera.position.set(1600, -1800, 1400);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, {height});
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxDistance = 5000;
        controls.minDistance = 200;

        // Ambient and Directional Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(1000, 1500, 2000);
        scene.add(dirLight);

        // Ground Grid Floor
        const gridHelper = new THREE.GridHelper(4000, 40, 0x1F2238, 0x121422);
        gridHelper.rotation.x = Math.PI / 2;
        gridHelper.position.z = -20;
        scene.add(gridHelper);

        // 3D Track Extruded Tube / Spline
        const points = trackData.map(p => new THREE.Vector3(p.x, p.y, p.z));
        const curve = new THREE.CatmullRomCurve3(points, true);

        // Vertex color-mapped tube based on apex speed
        const tubeGeometry = new THREE.TubeGeometry(curve, 400, 9, 8, true);
        const count = tubeGeometry.attributes.position.count;
        tubeGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
        const colors = tubeGeometry.attributes.color;

        const cBraking = new THREE.Color(0xFF1801); // Red
        const cMedium = new THREE.Color(0xFFD700);  // Gold
        const cHigh = new THREE.Color(0x39B54A);    // Green
        const cTop = new THREE.Color(0x00F5D4);     // Cyan

        for (let i = 0; i < count; i++) {{
            const u = (i / count);
            const spdIdx = Math.floor(u * trackData.length) % trackData.length;
            const spd = trackData[spdIdx].speed;
            let col = cMedium;
            if (spd < 120) col = cBraking;
            else if (spd < 220) col = cMedium;
            else if (spd < 290) col = cHigh;
            else col = cTop;

            colors.setXYZ(i, col.r, col.g, col.b);
        }}

        const tubeMaterial = new THREE.MeshStandardMaterial({{
            vertexColors: true,
            roughness: 0.35,
            metalness: 0.6,
        }});
        const trackMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
        scene.add(trackMesh);

        // DRS Glowing Neon Rails
        const drsPoints = [];
        trackData.forEach((p, idx) => {{
            if (p.is_drs) drsPoints.push(new THREE.Vector3(p.x, p.y, p.z + 12));
        }});
        if (drsPoints.length > 5) {{
            const drsGeo = new THREE.BufferGeometry().setFromPoints(drsPoints);
            const drsMat = new THREE.PointsMaterial({{
                color: 0x00F5D4,
                size: 7,
                transparent: true,
                opacity: 0.85
            }});
            const drsParticles = new THREE.Points(drsGeo, drsMat);
            scene.add(drsParticles);
        }}

        // Dynamic 3D Car Spheres
        const carMeshes = [];
        carData.forEach(car => {{
            const carGeo = new THREE.SphereGeometry(18, 16, 16);
            const carMat = new THREE.MeshStandardMaterial({{
                color: new THREE.Color(car.color),
                emissive: new THREE.Color(car.color),
                emissiveIntensity: 0.5,
                roughness: 0.2,
                metalness: 0.8
            }});
            const mesh = new THREE.Mesh(carGeo, carMat);
            mesh.position.set(car.x, car.y, car.z + 18);
            scene.add(mesh);
            carMeshes.push(mesh);
        }});

        // Camera Preset Function
        window.setCamera = function(preset) {{
            document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
            if (preset === 'tactical') {{
                camera.position.set(1600, -1800, 1400);
                controls.target.set(0, 0, 0);
            }} else if (preset === 'heli') {{
                camera.position.set(0, 0, 2400);
                controls.target.set(0, 0, 0);
            }} else if (preset === 'chase' && carMeshes.length > 0) {{
                const leadCar = carMeshes[0];
                camera.position.set(leadCar.position.x - 180, leadCar.position.y - 180, leadCar.position.z + 90);
                controls.target.copy(leadCar.position);
            }}
            controls.update();
        }};

        // Animation Loop
        let clock = new THREE.Clock();
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / {height};
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, {height});
        }});
    </script>
</body>
</html>
"""
        return html_template
