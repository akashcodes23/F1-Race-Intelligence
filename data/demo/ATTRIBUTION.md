# F1 Race Intelligence — Offline Demo Dataset Attribution & Terms

## Data Source & Attribution Note

This directory contains offline fallback timing data (`demo_laps.parquet`, 6.2 MB) covering **12,012 cleaned race laps** across **13 Grand Prix events** (2021–2023) to enable full application functionality offline, with empty cache directories, and independent of upstream live telemetry feeds.

### Data Origin & Sources
1. **Formula 1 Timing Data:** Sourced via the open-source [FastF1](https://github.com/theOehrly/Fast-F1) Python library, which interfaces with Formula 1 live timing and archive feeds.
2. **Historical Timing Archive:** Ergast Developer API (historical driver, constructor, and results data).
3. **Usage & Disclaimer:** Formula 1, F1, and related marks are trademarks of Formula One Licensing B.V. This project is a non-commercial, open-source educational and research analytics platform. It is not associated with, sponsored by, or affiliated with Formula 1, the FIA, or any Formula 1 team.

### Legal Notice on Redistribution of Derived Timing Data
> [!IMPORTANT]
> **Redistribution Notice:** Redistributing derived Formula 1 timing and telemetry data may be restricted by the upstream provider's terms of service (Formula One Licensing B.V. / Formula One Management and FastF1 license terms).
> 
> To comply with terms and maintain a minimal repository footprint:
> - The demo dataset is kept strictly minimal (~6.2 MB compressed parquet containing only essential timing channels without high-frequency raw telemetry arrays or GPS coordinates).
> - Users and downstream developers can rebuild or regenerate the demo dataset directly from authorized FastF1 caches using the provided rebuild script.

### Rebuilding the Demo Dataset
To rebuild or update `data/demo/demo_laps.parquet` locally from your own FastF1 cache:
```bash
PYTHONPATH=. python scripts/export_demo_dataset.py --output data/demo/demo_laps.parquet
```

### Included Grand Prix Events (13 Races)
- **2021 Season:**
  - 2021 Bahrain Grand Prix (Sakhir)
  - 2021 Spanish Grand Prix (Barcelona)
  - 2021 Austrian Grand Prix (Red Bull Ring)
  - 2021 British Grand Prix (Silverstone)
  - 2021 Italian Grand Prix (Monza)
  - 2021 United States Grand Prix (COTA)
- **2022 Season:**
  - 2022 Bahrain Grand Prix (Sakhir)
  - 2022 Spanish Grand Prix (Barcelona)
  - 2022 British Grand Prix (Silverstone)
  - 2022 Austrian Grand Prix (Red Bull Ring)
  - 2022 French Grand Prix (Paul Ricard)
  - 2022 Italian Grand Prix (Monza)
- **2023 Season:**
  - 2023 Monaco Grand Prix (Circuit de Monaco)
