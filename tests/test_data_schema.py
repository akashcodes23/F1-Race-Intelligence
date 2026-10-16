import pandas as pd

from src.data.schema import EXPECTED_LAP_COLUMNS, LapRecord, validate_laps_df


def test_lap_record_pydantic_valid():
    record = LapRecord(
        LapNumber=12,
        Driver="VER",
        LapTimeSec=78.432,
        Stint=1,
        Compound="MEDIUM",
        TyreLife=12.0,
        TrackStatus="1",
        IsAccurate=True,
    )
    assert record.LapNumber == 12
    assert record.Driver == "VER"
    assert record.LapTimeSec == 78.432
    assert record.Compound == "MEDIUM"


def test_validate_laps_df_empty():
    df = validate_laps_df(pd.DataFrame())
    assert df.empty
    assert list(df.columns) == EXPECTED_LAP_COLUMNS


def test_validate_laps_df_coercion():
    raw_df = pd.DataFrame({
        "LapNumber": ["1", "2", "3"],
        "Driver": [1, 2, 3],
        "LapTimeSec": ["80.5", "81.0", "80.2"],
        "Stint": ["1", "1", "1"],
        "Compound": ["soft", "medium", None],
        "TireAge": ["1", "2", "3"],
        "TrackStatus": [1, 1, 1],
        "IsAccurate": [1, 0, 1],
    })
    validated = validate_laps_df(raw_df)
    assert validated["LapNumber"].dtype == int
    assert validated["Driver"].dtype == object
    assert validated["LapTimeSec"].dtype == float
    assert validated["Compound"].tolist() == ["SOFT", "MEDIUM", "UNKNOWN"]
    assert "TyreLife" in validated.columns
    assert validated["TrackStatus"].tolist() == ["1", "1", "1"]
    assert validated["IsAccurate"].tolist() == [True, False, True]
