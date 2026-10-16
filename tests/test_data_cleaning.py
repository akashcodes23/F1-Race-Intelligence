import pandas as pd

from src.data.cleaning import clean_laps, clean_laps_with_stats, is_caution_track_status


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
