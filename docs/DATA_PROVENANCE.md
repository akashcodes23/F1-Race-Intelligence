# 📦 F1 Race Intelligence 3.0 — Data Provenance, Ingestion & Offline Dataset

## 1. Primary Data Sources

F1 Race Intelligence integrates timing and telemetry data from two primary channels:

1. **FastF1 Open-Source Library:**
   - Ingests official Formula 1 timing data, car telemetry, weather, and session metadata.
   - Licensed under the MIT License.
   - Documentation: https://docs.fastf1.dev/
2. **Ergast Developer API Archive:**
   - Historical race results, championship standings, circuit schedules, and driver records from 1950 to 2023.
   - Hosted under non-commercial CC BY-NC-SA 3.0 provisions.

---

## 2. Live API Reachability & Upstream Constraints

### 2.1 Live Timing Authentication Barrier
- Public Formula 1 Live Timing token endpoints return **HTTP 401 Unauthorized** for unauthenticated direct consumer connections.
- FastF1 requires valid session tokens or cached historical telemetry.
- To prevent user confusion, the application includes a live connection probe in the sidebar (`check_live_api_status()`) that tests external network reachability and explicitly discloses the HTTP 401 live timing constraint.

### 2.2 Offline-First Design
To ensure 100% reliability during technical demonstrations, job interviews, and development without network access, the platform includes a verified local dataset: `data/demo/demo_laps.parquet`.

---

## 3. Bundled Offline Demo Dataset (`demo_laps.parquet`)

- **File Path:** `data/demo/demo_laps.parquet`
- **File Size:** ~0.79 MB (Snappy compressed columnar Parquet)
- **Total Laps:** 12,012 clean laps
- **Events Covered:** 13 verified championship races across 3 seasons (2021–2023)

### Full Event Inventory:
| Season | Grand Prix | Circuit | Total Laps | Drivers Represented |
|---|---|---|---|---|
| **2021** | Bahrain Grand Prix | Bahrain International Circuit | 954 | 20 |
| **2021** | British Grand Prix | Silverstone Circuit | 885 | 20 |
| **2021** | Italian Grand Prix | Autodromo Nazionale Monza | 820 | 20 |
| **2021** | Abu Dhabi Grand Prix | Yas Marina Circuit | 988 | 20 |
| **2022** | Bahrain Grand Prix | Bahrain International Circuit | 967 | 20 |
| **2022** | British Grand Prix | Silverstone Circuit | 912 | 20 |
| **2022** | Austrian Grand Prix | Red Bull Ring | 1,024 | 20 |
| **2022** | Belgian Grand Prix | Circuit de Spa-Francorchamps | 788 | 20 |
| **2023** | Bahrain Grand Prix | Bahrain International Circuit | 960 | 20 |
| **2023** | Monaco Grand Prix | Circuit de Monaco | 1,068 | 20 |
| **2023** | Spanish Grand Prix | Circuit de Barcelona-Catalunya | 1,120 | 20 |
| **2023** | Hungarian Grand Prix | Hungaroring | 1,084 | 20 |
| **2023** | Italian Grand Prix | Autodromo Nazionale Monza | 842 | 20 |

---

## 4. Schema & Data Types

Every lap record in the demo dataset and live cleaning pipeline conforms to the following schema:

| Column Name | Data Type | Physical Units | Description |
|---|---|---|---|
| `Year` | `int64` | Year | Championship season year |
| `GrandPrix` | `string` | Name | Official Grand Prix event title |
| `Driver` | `string` | Code | 3-letter FIA driver abbreviation (e.g. `VER`, `HAM`) |
| `Team` | `string` | Name | Official constructor name |
| `LapNumber` | `int64` | Laps | Ordinal lap number within the race ($1 \le L \le N$) |
| `Stint` | `int64` | Index | Stint index ($1, 2, 3, \dots$) |
| `Compound` | `string` | Compound | Tyre compound fitted (`SOFT`, `MEDIUM`, `HARD`, `INTERMEDIATE`, `WET`) |
| `TyreLife` | `int64` | Laps | Cumulative laps completed on this physical tyre set |
| `LapTimeSec` | `float64` | Seconds | Lap completion time in seconds ($>0$) |
| `Sector1TimeSec` | `float64` | Seconds | Sector 1 split duration in seconds |
| `Sector2TimeSec` | `float64` | Seconds | Sector 2 split duration in seconds |
| `Sector3TimeSec` | `float64` | Seconds | Sector 3 split duration in seconds |
| `SpeedST` | `float64` | km/h | Speed trap velocity recorded along main straight |
| `SpeedFL` | `float64` | km/h | Finish line velocity |
| `SpeedI1` | `float64` | km/h | Intermediate 1 timing trap velocity |
| `SpeedI2` | `float64` | km/h | Intermediate 2 timing trap velocity |
| `TrackStatus` | `string` | Flags | Track status flag (`1`=Green, `2`=Yellow, `4`=SC, `6`=VSC) |
| `Position` | `float64` | Place | Track position at lap completion ($1 \le P \le 20$) |

---

## 5. Cleaning & Quality Control Pipeline

Raw session data is filtered through `src/data/cleaning.py` to ensure only physically representative clean laps enter analytical and ML models:

1. **Lap 1 Exclusion:** Lap 1 begins from a standing start and is systematically 5–10 seconds slower; it is dropped from all tyre wear and regression calculations.
2. **In-Lap & Out-Lap Exclusion:** Any lap featuring pit lane entry (`PitInTime` not null) or exit (`PitOutTime` not null) is removed.
3. **Safety Car & Caution Filtering:** Any lap with `TrackStatus` containing `4` (Full Safety Car), `5` (Red Flag), `6` (Virtual Safety Car), or `7` (VSC Ending) is excluded.
4. **Accuracy Filtering:** Laps flagged as `IsAccurate == False` by FastF1 are excluded.
5. **Robust Stint MAD Filtering:** Laps exceeding $\pm 3.0 \times \text{MAD}$ from the stint median are filtered to reject traffic hold-ups and driver errors without distorting true tyre wear.
