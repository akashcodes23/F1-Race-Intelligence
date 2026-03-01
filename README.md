# 🏎️ F1 Race Intelligence Platform
[![Live Demo](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge)](YOUR_STREAMLIT_LINK)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)]()

![Screenshot 2026-03-01 at 7 50 04 PM](https://github.com/user-attachments/assets/7b74f2ec-62f2-4c55-8b68-38d06606bdcf)

AI-Powered Motorsport Analytics & Strategic Performance Engine.Simulating real-world Formula 1 race engineering decision workflows using structured data analytics.

https://github.com/user-attachments/assets/ee20de16-98d4-466a-ba7b-0041e7992fb4

>F1 Race Intelligence Platform is a modular, production-oriented motorsport analytics system that transforms raw Formula 1 race session data into actionable performance intelligence.
Built on top of the FastF1 API and engineered using scalable data pipelines, this platform bridges sports analytics, machine learning foundations, and interactive decision-support visualization.


## 🎯 Executive Summary

Modern Formula 1 strategy is driven by data — lap times, tire degradation, stint dynamics, pace evolution, and race momentum.

This platform replicates that analytical framework by:
	
	•	Ingesting official F1 race session data
	•	Cleaning and validating lap-level telemetry
	•	Performing structured performance modeling
	•	Generating interactive strategic insights

It is not just a visualization tool.
It is a race intelligence system.


## 🧠 Core Capabilities

1️⃣ Driver Delta Intelligence Engine
	
	•	Lap-by-lap time delta modeling
	•	Cumulative advantage tracking
	•	Zero-baseline race momentum visualization
	•	Identification of strategic turning points

Enables micro-level pace analysis across the race lifecycle.


2️⃣ Tire Degradation Modeling System
	
	•	Quantile-based outlier filtering (1%–99%)
	•	LapTime normalization to seconds
	•	Linear regression slope extraction
	•	Side-by-side degradation comparison

Quantifies tire performance decay and driver consistency under load.


3️⃣ Defensive Data Pipeline Architecture
	
	•	FastF1 cache optimization
	•	Null-safe session loading
	•	Minimum lap threshold enforcement
	•	Defensive driver input validation
	•	Streamlit-level caching (data + resources)

Ensures analytical integrity and production-level robustness.


## 🏗 System Architecture

```shell
User Interface (Streamlit)
        ↓
Session Loader (FastF1 + Cache Layer)
        ↓
Data Validation & Cleaning Engine
        ↓
Feature Engineering Pipeline
        ↓
Analytics Core (Delta + Degradation Modeling)
        ↓
Interactive Plotly Intelligence Dashboard
```
The architecture follows clean separation of concerns:

	•	Data Layer
	•	Analytics Layer
	•	Visualization Layer
	•	Interface Layer

This makes the system extensible for advanced modeling modules.



## 📊 Technical Stack
```
•Python 3.10+
•FastF1 (Official F1 Data API)
•Pandas / NumPy (Data Engineering)
•Scikit-learn (Statistical Modeling)
•Plotly (Interactive Visualization)
•Streamlit (Analytics Interface Layer)
```
## ⚡ Performance Engineering

	•	Cached session loading to reduce API overhead
	•	Vectorized data operations
	•	Minimal recomputation via Streamlit cache decorators
	•	Structured modular imports for scalability
	•	Clean codebase separation under src/

This enables low-latency analytics even on full-race datasets (1000+ laps).


## 📂 Project Structure

```shell
F1-Race-Intelligence
│
├── app.py                     # Streamlit Controller (UI + orchestration)
├── src/
│   ├── data_loader.py         # FastF1 session ingestion & caching
│   ├── metrics.py             # Cleaning, modeling, analytics engine
│   ├── visualization.py       # Plotly visualization layer
│
├── config.py                  # Thresholds & model parameters
├── requirements.txt
└── README.md
```


## 🔬 Analytical Methodology

Lap Cleaning Strategy
	
	•	Convert timedelta → seconds
	•	Drop missing lap times
	•	Remove extreme outliers via quantile filtering
	•	Enforce minimum lap threshold

Degradation Metric
	
	•	Fit linear regression on LapNumber vs LapTimeSec
	•	Extract slope coefficient
	•	Interpret slope as tire decay rate

Delta Computation
	
	•	Merge lap times by lap number
	•	Compute lap delta
	•	Compute cumulative delta
	•	Plot race momentum curve

## 🎮 How to Run

```
git clone https://github.com/akashcodes23/F1-Race-Intelligence.git
cd F1-Race-Intelligence
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Why This Project Stands Out

This project demonstrates:
	
	•	Applied sports analytics
	•	Structured data engineering
	•	Statistical modeling foundations
	•	Interactive data storytelling
	•	Clean modular architecture
	•	Production-aware defensive programming

## 🛣 Roadmap: Phase II (Strategic Expansion)

Planned extensions:
	
	•	Pit Strategy Simulation Engine
	•	Lap Time Prediction Model (ML)
	•	Driver Consistency Index
	•	Stint-level segmentation modeling
	•	Strategy recommendation system
	•	AI-generated race summary module
	•	Multi-race comparative intelligence dashboard

The architecture already supports these extensions.

## 🤝 Contributing

We welcome contributions in:
	
	•	Advanced modeling (time-series ML)
	•	Feature engineering optimization
	•	Visualization enhancements
	•	Strategy simulation frameworks
	•	Telemetry-level integration

Open an issue or submit a pull request.

## License

Licensed under the MIT License, allowing full flexibility to reuse, modify, distribute, and integrate this project into personal or commercial applications. Attribution is required.


## Contact

If you have questions, suggestions, or collaboration ideas, feel free to reach out at:
📩 akashgpatil23.05@gmail.com
