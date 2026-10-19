"""
Unit tests for 3D Spatial Track Reconstruction and Telemetry Interpolation Engine.
"""

import unittest

import plotly.graph_objects as go

from src.visualization.track_3d import CIRCUIT_3D_WAYPOINTS, Track3DEngine


class TestTrack3DEngine(unittest.TestCase):
    def setUp(self):
        self.engine = Track3DEngine("Bahrain Grand Prix", elevation_scale=2.0, num_spline_points=400)

    def test_all_circuits_have_waypoints(self):
        for circuit_name, waypoints in CIRCUIT_3D_WAYPOINTS.items():
            self.assertGreater(len(waypoints), 5, f"Circuit {circuit_name} has insufficient waypoints")
            # First and last waypoints should form a closed loop (same or close start/finish)
            p_start = waypoints[0]
            p_end = waypoints[-1]
            dist = ((p_start.x - p_end.x)**2 + (p_start.y - p_end.y)**2)**0.5
            self.assertLess(dist, 10.0, f"Circuit {circuit_name} is not closed")

    def test_track_3d_spline_generation(self):
        df = self.engine.track_df
        self.assertEqual(len(df), 400)
        self.assertIn("x", df.columns)
        self.assertIn("y", df.columns)
        self.assertIn("z", df.columns)
        self.assertIn("speed", df.columns)
        self.assertIn("is_drs", df.columns)

        # Speeds should be physically bounded in [60, 360] km/h
        self.assertGreaterEqual(df["speed"].min(), 60.0)
        self.assertLessEqual(df["speed"].max(), 360.0)

        # Elevation scale verification: z values should have positive range
        z_range = df["z"].max() - df["z"].min()
        self.assertGreater(z_range, 0.0)

    def test_interpolate_car_positions(self):
        drivers = [
            {"driver": "VER", "position": 1, "team": "Red Bull Racing", "gap_to_lead_s": 0.0},
            {"driver": "HAM", "position": 2, "team": "Mercedes", "gap_to_lead_s": 1.45},
            {"driver": "LEC", "position": 3, "team": "Ferrari", "gap_to_lead_s": 3.80},
        ]
        cars_3d = self.engine.interpolate_car_positions(drivers, lap_progress=0.45)
        self.assertEqual(len(cars_3d), 3)

        ver = cars_3d[0]
        self.assertEqual(ver["driver"], "VER")
        self.assertEqual(ver["team"], "Red Bull Racing")
        self.assertEqual(ver["color"], "#3671C6")
        self.assertEqual(ver["gap_s"], 0.0)

        ham = cars_3d[1]
        self.assertEqual(ham["driver"], "HAM")
        self.assertEqual(ham["color"], "#6CD3BF")
        self.assertEqual(ham["gap_s"], 1.45)
        # HAM is behind VER, so progress should reflect spatial lag
        self.assertNotEqual(ver["progress"], ham["progress"])

    def test_generate_plotly_3d(self):
        drivers = [
            {"driver": "VER", "position": 1, "team": "Red Bull Racing", "gap_to_lead_s": 0.0},
            {"driver": "HAM", "position": 2, "team": "Mercedes", "gap_to_lead_s": 1.45},
        ]
        cars_3d = self.engine.interpolate_car_positions(drivers, lap_progress=0.30)
        fig = self.engine.generate_plotly_3d(cars_3d, title="Test 3D Circuit")

        self.assertIsInstance(fig, go.Figure)
        # Should contain Track Centerline, DRS Zone, Race Positions, and Lead Delta ribbon
        trace_names = [t.name for t in fig.data]
        self.assertIn("Track Centerline", trace_names)
        self.assertIn("Race Positions", trace_names)
        self.assertTrue(any("Lead Delta" in str(n) for n in trace_names))

        # Check dark layout
        self.assertEqual(fig.layout.paper_bgcolor, "#0A0C10")

    def test_generate_threejs_html(self):
        drivers = [
            {"driver": "VER", "position": 1, "team": "Red Bull Racing", "gap_to_lead_s": 0.0},
        ]
        cars_3d = self.engine.interpolate_car_positions(drivers, lap_progress=0.10)
        html = self.engine.generate_threejs_html(cars_3d, height=500)

        self.assertIn("three.min.js", html)
        self.assertIn("OrbitControls.js", html)
        self.assertIn("Tactical 3D", html)
        self.assertIn("Helicopter Drone", html)
        self.assertIn("Cockpit Chase", html)
        self.assertIn("VER", html)


if __name__ == "__main__":
    unittest.main()
