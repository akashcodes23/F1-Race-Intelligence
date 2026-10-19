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

    def test_multiple_circuits_and_lap_times(self):
        circuits_to_test = ["Monaco Grand Prix", "Austrian Grand Prix", "British Grand Prix", "Italian Grand Prix"]
        for c_name in circuits_to_test:
            engine = Track3DEngine(circuit_name=c_name, elevation_scale=1.5, num_spline_points=300)
            df = engine.track_df
            self.assertEqual(len(df), 300)
            self.assertTrue(df["speed"].between(60.0, 360.0).all())

            # Test spatial lag scaling across different reference lap times (70s vs 105s)
            drivers = [
                {"driver": "P1", "position": 1, "team": "Ferrari", "gap_to_lead_s": 0.0},
                {"driver": "P2", "position": 2, "team": "Mercedes", "gap_to_lead_s": 7.0},
            ]
            cars_short_lap = engine.interpolate_car_positions(drivers, lap_progress=0.5, lap_time_ref_sec=70.0)
            cars_long_lap = engine.interpolate_car_positions(drivers, lap_progress=0.5, lap_time_ref_sec=105.0)

            # In shorter lap, 7.0s gap covers larger fraction of the track (10%) than in longer lap (6.67%)
            lag_short = (cars_short_lap[0]["progress"] - cars_short_lap[1]["progress"]) % 1.0
            lag_long = (cars_long_lap[0]["progress"] - cars_long_lap[1]["progress"]) % 1.0
            self.assertGreater(lag_short, lag_long)

    def test_interpolate_car_positions_edge_cases(self):
        engine = self.engine
        # 1. Zero gaps
        zero_gap_drivers = [
            {"driver": "VER", "position": 1, "gap_to_lead_s": 0.0},
            {"driver": "PER", "position": 2, "gap_to_lead_s": 0.0},
        ]
        cars_zero = engine.interpolate_car_positions(zero_gap_drivers, lap_progress=0.25)
        self.assertEqual(cars_zero[0]["progress"], cars_zero[1]["progress"])

        # 2. Large gaps (wrapping around the track)
        large_gap_drivers = [
            {"driver": "VER", "position": 1, "gap_to_lead_s": 0.0},
            {"driver": "LAT", "position": 20, "gap_to_lead_s": 95.5},  # > 90s lap time
        ]
        cars_large = engine.interpolate_car_positions(large_gap_drivers, lap_progress=0.10, lap_time_ref_sec=90.0)
        self.assertEqual(len(cars_large), 2)
        self.assertTrue(0.0 <= cars_large[1]["progress"] <= 1.0)

        # 3. Missing/None gap or missing fields
        missing_data_drivers = [
            {"driver": "VER", "position": 1, "gap_to_lead_s": None},
            {"driver": "HAM"},
        ]
        cars_missing = engine.interpolate_car_positions(missing_data_drivers, lap_progress=0.35)
        self.assertEqual(len(cars_missing), 2)
        self.assertIn("progress", cars_missing[0])
        self.assertIn("progress", cars_missing[1])

        # 4. Empty driver list
        cars_empty = engine.interpolate_car_positions([], lap_progress=0.5)
        self.assertEqual(cars_empty, [])


if __name__ == "__main__":
    unittest.main()
