import pandas as pd

from src.data.cleaning import (
    clean_laps,
    clean_laps_with_stats,
    filter_stint_mad_outliers,
    is_caution_track_status,
)


def test_is_caution_track_status():
    assert is_caution_track_status("1") is False
    assert is_caution_track_status("1.0") is False
    assert is_caution_track_status("") is False

    # Caution codes (4=SC, 5=Red, 6=VSC, 7=VSC Ending)
    assert is_caution_track_status("4") is True
    assert is_caution_track_status("5") is True
    assert is_caution_track_status("6") is True
    assert is_caution_track_status("7") is True

    # Combined codes
    assert is_caution_track_status("124") is True
    assert is_caution_track_status("26") is True
    assert is_caution_track_status("17") is True

    # Yellow flag handling ('2')
    assert is_caution_track_status("2", drop_yellow=False) is False
    assert is_caution_track_status("2", drop_yellow=True) is True
    assert is_caution_track_status("12", drop_yellow=False) is False
    assert is_caution_track_status("12", drop_yellow=True) is True


def test_clean_laps_combined_track_status():
    df = pd.DataFrame({
        "LapNumber": [2, 3, 4, 5, 6, 7, 8],
        "Driver": ["VER"] * 7,
        "LapTimeSec": [80.0, 80.1, 80.2, 105.0, 110.0, 95.0, 80.3],
        "TrackStatus": ["1", "12", "2", "124", "26", "7", "1"],
    })
    # Default (drop_yellow=False): drops '124', '26', '7'
    cleaned = clean_laps(df, drop_caution_laps=True, drop_yellow_laps=False, outlier_low=0.0, outlier_high=1.0)
    assert 5 not in cleaned["LapNumber"].values  # '124'
    assert 6 not in cleaned["LapNumber"].values  # '26'
    assert 7 not in cleaned["LapNumber"].values  # '7'
    assert 3 in cleaned["LapNumber"].values      # '12' kept
    assert 4 in cleaned["LapNumber"].values      # '2' kept

    # With drop_yellow_laps=True: also drops '12' and '2'
    cleaned_no_yellow = clean_laps(df, drop_caution_laps=True, drop_yellow_laps=True, outlier_low=0.0, outlier_high=1.0)
    assert 3 not in cleaned_no_yellow["LapNumber"].values
    assert 4 not in cleaned_no_yellow["LapNumber"].values


def test_clean_laps_driver_breakdown_stats():
    df = pd.DataFrame({
        "LapNumber": [1, 2, 3, 1, 2, 3],
        "Driver": ["VER", "VER", "VER", "HAM", "HAM", "HAM"],
        "LapTimeSec": [95.0, 80.0, 80.2, 96.0, 110.0, 80.5],
        "TrackStatus": ["1", "1", "1", "1", "4", "1"],  # HAM Lap 2 is SC
    })
    cleaned, stats = clean_laps_with_stats(df, drop_lap1=True, drop_caution_laps=True, outlier_low=0.0, outlier_high=1.0)
    assert stats["filter_breakdown"]["lap1"] == 2
    assert stats["filter_breakdown"]["caution_laps"] == 1
    assert "VER" in stats["driver_breakdown"]
    assert "HAM" in stats["driver_breakdown"]
    assert stats["driver_breakdown"]["HAM"]["caution_laps"] == 1
    assert stats["driver_breakdown"]["VER"]["lap1"] == 1


# ─── Restored Phase 0 Tests ──────────────────────────────────────────

def test_clean_laps_drops_lap1():
    df = pd.DataFrame({
        "LapNumber": [1, 2, 3, 4, 5],
        "Driver": ["VER"] * 5,
        "LapTimeSec": [95.0, 80.0, 80.2, 80.1, 80.3],
    })
    cleaned = clean_laps(df, drop_lap1=True, outlier_low=0.0, outlier_high=1.0)
    assert 1 not in cleaned["LapNumber"].values
    assert len(cleaned) == 4


def test_clean_laps_drops_pit_laps():
    df = pd.DataFrame({
        "LapNumber": [2, 3, 4, 5, 6],
        "Driver": ["HAM"] * 5,
        "LapTimeSec": [81.0, 81.2, 105.0, 81.1, 81.3],
        "PitInTime": [pd.NaT, pd.NaT, pd.Timestamp("2023-05-28 15:30:00"), pd.NaT, pd.NaT],
        "PitOutTime": [pd.NaT] * 5,
    })
    cleaned = clean_laps(df, drop_pit_laps=True, outlier_low=0.0, outlier_high=1.0)
    assert 4 not in cleaned["LapNumber"].values
    assert len(cleaned) == 4


def test_clean_laps_drops_caution_and_vsc():
    df = pd.DataFrame({
        "LapNumber": [2, 3, 4, 5, 6, 7],
        "Driver": ["ALO"] * 6,
        "LapTimeSec": [80.0, 80.1, 110.0, 115.0, 80.2, 80.3],
        "TrackStatus": ["1", "1", "4", "6", "1", "1"],  # '4'=SC, '6'=VSC
    })
    cleaned = clean_laps(df, drop_caution_laps=True, outlier_low=0.0, outlier_high=1.0)
    assert 4 not in cleaned["LapNumber"].values
    assert 5 not in cleaned["LapNumber"].values
    assert len(cleaned) == 4


def test_clean_laps_drops_inaccurate():
    df = pd.DataFrame({
        "LapNumber": [2, 3, 4, 5, 6],
        "Driver": ["LEC"] * 5,
        "LapTimeSec": [79.0, 79.1, 99.0, 79.2, 79.3],
        "IsAccurate": [True, True, False, True, True],
    })
    cleaned = clean_laps(df, drop_inaccurate=True, outlier_low=0.0, outlier_high=1.0)
    assert 4 not in cleaned["LapNumber"].values
    assert len(cleaned) == 4


def test_clean_laps_with_stats_telemetry():
    df = pd.DataFrame({
        "LapNumber": [1, 2, 3, 4, 5, 6],
        "Driver": ["RUS"] * 6,
        "LapTimeSec": [95.0, 80.0, 80.2, 110.0, 80.1, 80.3],
        "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.Timestamp("2023-05-28 15:30:00"), pd.NaT, pd.NaT],
    })
    cleaned, stats = clean_laps_with_stats(df, drop_lap1=True, drop_pit_laps=True, outlier_low=0.0, outlier_high=1.0)
    assert stats["original_count"] == 6
    assert stats["final_count"] == 4
    assert stats["removed_count"] == 2
    assert stats["sufficient"] is True


def test_robust_stint_mad_outlier_filtering():
    # Stint with 10 normal laps and 1 severe rogue outlier (e.g. spun, 15 seconds off)
    normal_times = [80.0 + i * 0.05 for i in range(10)]
    times = normal_times + [98.5]  # Lap 11 is an outlier
    df = pd.DataFrame({
        "LapNumber": list(range(2, 13)),
        "Driver": ["VER"] * 11,
        "Stint": [1] * 11,
        "Compound": ["MEDIUM"] * 11,
        "LapTimeSec": times,
    })
    mask = filter_stint_mad_outliers(df, mad_k=3.5)
    assert mask.iloc[-1] is True or mask.iloc[-1] == 1
    # Check that normal laps are NOT flagged
    assert mask.iloc[:10].sum() == 0

    # Test through clean_laps
    cleaned, stats = clean_laps_with_stats(df, outlier_mad_k=3.5, filter_outliers=True)
    assert stats["filter_breakdown"]["outliers"] == 1
    assert 12 not in cleaned["LapNumber"].values
