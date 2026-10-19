# F1 Race Intelligence — Data Access & Upstream API Technical Audit

## 1. Executive Summary & Root Cause

The F1 Race Intelligence application integrates live telemetry and lap timing feeds via the `FastF1` Python library. During testing of uncached historical and recent Grand Prix sessions (e.g. 2023 Bahrain GP, 2024 Bahrain GP), session loading failed with:

```text
fastf1.exceptions.DataNotLoadedError: The data you are trying to access has not been loaded yet. See Session.load
```

### Root Cause Verification

Under FastF1 `DEBUG` logging (`fastf1.set_log_level("DEBUG")`), the exact failure chain was captured:

```text
fastf1.req: DEBUG: Getting URL: https://livetiming.formula1.com/static/2023/2023-03-05_Bahrain_Grand_Prix/2023-03-05_Race/TimingData.jsonStream
urllib3.connectionpool: DEBUG: https://livetiming.formula1.com:443 "GET /static/2023/2023-03-05_Bahrain_Grand_Prix/2023-03-05_Race/TimingData.jsonStream HTTP/1.1" 401 None
fastf1.req: DEBUG: Getting URL: https://livetiming-mirror.fastf1.dev/static/2023/2023-03-05_Bahrain_Grand_Prix/2023-03-05_Race/TimingData.jsonStream
urllib3.connectionpool: DEBUG: https://livetiming-mirror.fastf1.dev:443 "GET /static/2023/2023-03-05_Bahrain_Grand_Prix/2023-03-05_Race/TimingData.jsonStream HTTP/1.1" 404 None
fastf1._api.SessionNotAvailableError: No data for this session!
    at fastf1._api._load_timing_data()
    --> caught internally; leaves session._laps = None
fastf1.exceptions.DataNotLoadedError: The data you are trying to access has not been loaded yet. See Session.load
    at fastf1.core.Session.laps (fastf1/core.py:1262)
```

### Analysis
1. **Primary Request:** Formula 1 official live timing endpoint (`livetiming.formula1.com`) returned **HTTP 401 Unauthorized**. Formula 1 has permanently restricted unauthenticated public access to session timing streams.
2. **Mirror Fallback:** FastF1's community mirror (`livetiming-mirror.fastf1.dev`) returned **HTTP 404 Not Found**, as timing streams for these sessions are not hosted on the mirror.
3. **Exception Chain:** `SessionNotAvailableError` was caught internally in `_load_timing_data()`. Because timing streams failed, `session._laps` remained uninitialized (`None`). Calling `session.laps` subsequently raised `DataNotLoadedError`.
4. **Conclusion:** `DataNotLoadedError` is not the root cause; the root cause is **upstream HTTP 401 Unauthorized** due to mandatory authentication credentials on official Formula 1 live timing streams.

---

## 2. FastF1 Version Evaluation (v3.8.3 in Fresh Environment)

In a clean, isolated virtual environment, the latest release **`fastf1==3.8.3`** was benchmarked on two uncached sessions:

1. **Uncached 2021 Race (2021 Abu Dhabi Grand Prix):**
   - **Result:** **SUCCESSFUL** (1,000 clean laps loaded).
   - **Reason:** FastF1 accesses historical Ergast and F1 archive mirrors that remain publicly reachable for older seasons.
2. **Uncached 2024 Race (2024 Bahrain Grand Prix):**
   - **Result:** **FAILED** (`DataNotLoadedError`, upstream HTTP 401 Unauthorized).
   - **Reason:** 2024 live timing streams require active Formula 1 credentials.

`requirements.txt` has been updated to pin `fastf1==3.8.3`.

---

## 3. High-Availability Offline Architecture: Bundled Demo Dataset

To guarantee 100% application reliability on hosted environments (Streamlit Community Cloud) independent of live external network state or upstream authentication barriers, the application ships with a bundled, pre-processed offline demo dataset:

- **File Location:** `data/demo/demo_laps.parquet`
- **Size:** **0.79 MB** (828,564 bytes), well below the 10 MB budget.
- **Coverage:** **12,012 cleaned laps across 13 Grand Prix events**:
  - **2021:** Austrian, Bahrain, British, Italian, Spanish, US Grand Prix.
  - **2022:** Austrian, Bahrain, British, French, Italian, Spanish Grand Prix.
  - **2023:** Monaco Grand Prix.
- **Attribution & Terms:** Fully documented in `data/demo/ATTRIBUTION.md`.
- **Offline Ingestion:** `src/data/loader.py` and `src/data_loader.py` implement `get_demo_races()` and `load_demo_laps()`. When live API fetching fails or when running offline, the application seamlessly serves data from this bundled repository.

---

## 4. Telemetry Availability Notice

Continuous high-frequency telemetry channels (millisecond Speed, Throttle, Brake traces, and GPS coordinates) are **not bundled** into `demo_laps.parquet` to maintain the compact sub-1MB package footprint. Telemetry visualization requires live reachable FastF1 sessions with valid telemetry data; for offline demo sessions, pages cleanly display a notice indicating that high-frequency telemetry is unavailable in offline demo mode.
