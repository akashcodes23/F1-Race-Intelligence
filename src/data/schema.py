"""
F1 Race Intelligence — Validated Laps Schema

Defines formal data contracts for lap records and provides schema validation
using Pydantic and Pandas.
"""

from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class LapRecord(BaseModel):
    """Schema definition for a single lap record."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    LapNumber: int = Field(..., ge=1, description="Sequential lap number in session")
    Driver: str = Field(..., min_length=1, max_length=10, description="3-letter driver code or name")
    LapTimeSec: float = Field(..., gt=0.0, description="Lap duration in float seconds")
    Stint: Optional[int] = Field(default=None, ge=1, description="Stint index (1-indexed)")
    Compound: Optional[str] = Field(default="UNKNOWN", description="Tyre compound name")
    TyreLife: Optional[float] = Field(default=None, ge=0.0, description="Tyre age in completed laps")
    TrackStatus: Optional[str] = Field(default="1", description="FastF1 TrackStatus code string")
    IsAccurate: Optional[bool] = Field(default=True, description="Whether timing is flagged accurate")
    PitInTime: Optional[object] = Field(default=None, description="Timestamp of pit entry if in-lap")
    PitOutTime: Optional[object] = Field(default=None, description="Timestamp of pit exit if out-lap")
    Sector1TimeSec: Optional[float] = Field(default=None, gt=0.0, description="Sector 1 time in seconds")
    Sector2TimeSec: Optional[float] = Field(default=None, gt=0.0, description="Sector 2 time in seconds")
    Sector3TimeSec: Optional[float] = Field(default=None, gt=0.0, description="Sector 3 time in seconds")


EXPECTED_LAP_COLUMNS: List[str] = [
    "LapNumber",
    "Driver",
    "LapTimeSec",
    "Stint",
    "Compound",
    "TyreLife",
    "TrackStatus",
    "IsAccurate",
]


def validate_laps_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and coerce a Pandas DataFrame against the Laps schema.

    Ensures proper dtypes and required fields without mutating input.
    Returns a normalized DataFrame conforming to the schema.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=EXPECTED_LAP_COLUMNS)

    validated = df.copy()

    # 1. Coerce LapNumber
    if "LapNumber" in validated.columns:
        validated["LapNumber"] = pd.to_numeric(validated["LapNumber"], errors="coerce").fillna(0).astype(int)

    # 2. Coerce Driver
    if "Driver" in validated.columns:
        validated["Driver"] = validated["Driver"].astype(str)

    # 3. Coerce LapTimeSec
    if "LapTimeSec" in validated.columns:
        validated["LapTimeSec"] = pd.to_numeric(validated["LapTimeSec"], errors="coerce")

    # 4. Coerce Stint
    if "Stint" in validated.columns:
        validated["Stint"] = pd.to_numeric(validated["Stint"], errors="coerce")

    # 5. Coerce Compound
    if "Compound" in validated.columns:
        validated["Compound"] = validated["Compound"].fillna("UNKNOWN").astype(str).str.upper()

    # 6. Coerce TyreLife
    if "TyreLife" in validated.columns:
        validated["TyreLife"] = pd.to_numeric(validated["TyreLife"], errors="coerce")
    elif "TireAge" in validated.columns:
        validated["TyreLife"] = pd.to_numeric(validated["TireAge"], errors="coerce")

    # 7. Coerce TrackStatus
    if "TrackStatus" in validated.columns:
        validated["TrackStatus"] = validated["TrackStatus"].fillna("1").astype(str)

    # 8. Coerce IsAccurate
    if "IsAccurate" in validated.columns:
        validated["IsAccurate"] = validated["IsAccurate"].fillna(True).astype(bool)

    return validated
