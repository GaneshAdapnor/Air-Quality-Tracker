# 🌬️ AtmosShield — Air Quality Intelligence Platform

> Real-time indoor & outdoor air quality monitoring powered by OpenWeatherMap, covering 120+ cities across 50+ countries.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenWeatherMap](https://img.shields.io/badge/OpenWeatherMap-Live_API-EB6E4B)](https://openweathermap.org/api/air-pollution)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)

---

## Overview

**AtmosShield** is a full-stack air quality dashboard that fetches **live data** from the OpenWeatherMap Air Pollution API and displays four key air quality metrics:

| Metric | Description |
|---|---|
| 🌍 **Outdoor AQI** | Real-time ambient air quality index |
| 🏠 **Indoor AQI** | Estimated indoor air quality (67% infiltration ratio) |
| ✨ **Outdoor Purified** | Post-purification outdoor AQI (~80% reduction) |
| 💨 **Indoor Purified** | Post-purification indoor AQI (~82% reduction) |

Indoor and purified values are derived from empirical HEPA filter efficiency and typical building infiltration ratios.

---

## Features

- **Live OpenWeatherMap data** — real-time pollutants (PM2.5, PM10, CO, NO₂, O₃, SO₂) + weather
- **Global coverage** — 120+ cities, 50+ countries, from Delhi to Wellington
- **Dark / Light theme** — single-click toggle, sidebar always stays dark
- **Sidebar controls** — country → city cascade dropdown + API key input
- **Current conditions** — AQI KPI cards, weather metrics, full pollutant breakdown
- **Purification analysis** — before/after comparison chart with efficiency percentages
- **5-Day Forecast** — chart + colour-coded table from OWM weather forecast
- **City Comparison** — live multi-city AQI bar chart (up to 6 cities)
- **AQI legend** — inline colour scale for quick reference

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.32+ |
| Data Source | OpenWeatherMap Air Pollution & Weather APIs |
| Data Processing | Pandas, NumPy |
| Charts | Plotly |
| Styling | Custom CSS via `st.markdown` |

---

## Project Structure

```
Air-Quality-Tracker/
├── app.py              # Main application (live fetch, AQI calc, UI)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- A free [OpenWeatherMap API key](https://openweathermap.org/appid)

### Installation

```bash
# Clone the repository
git clone https://github.com/GaneshAdapnor/Air-Quality-Tracker.git
cd Air-Quality-Tracker

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. Enter your OpenWeatherMap API key in the sidebar.

---

## How It Works

### 1. Live Data Fetching

Two OpenWeatherMap endpoints are called per city selection (cached 30 min):

| Endpoint | Data |
|---|---|
| `/data/2.5/air_pollution` | PM2.5, PM10, CO, NO₂, O₃, SO₂ |
| `/data/2.5/weather` | Temperature, humidity, wind speed, pressure, visibility |
| `/data/2.5/forecast` | 5-day / 3-hour forecast, grouped to daily |

### 2. AQI Calculation

AQI is computed using the **EPA breakpoint sub-index formula** across all available pollutants. The displayed AQI is the maximum sub-index across all pollutants.

### 3. Derived Metrics

Indoor and purified AQI values are estimated using real-world empirical ratios:

| Metric | Formula |
|---|---|
| Indoor AQI | `outdoor_aqi × 0.67` (typical building infiltration) |
| Outdoor Purified | `outdoor_aqi × 0.20` (HEPA ~80% efficiency) |
| Indoor Purified | `indoor_aqi × 0.18` (combined infiltration + filtration) |

---

## AQI Scale Reference

| Range | Category | Color |
|---|---|---|
| 0 – 50 | Good | 🟢 |
| 51 – 100 | Moderate | 🟡 |
| 101 – 150 | Unhealthy for Sensitive Groups | 🟠 |
| 151 – 200 | Unhealthy | 🔴 |
| 201 – 300 | Very Unhealthy | 🟣 |
| 301 – 500 | Hazardous | ⚫ |

---

## Configuration

All city data lives in the `WORLD_CITIES` constant in `app.py`. To add a new city:

```python
"Country Name": [("City Name", pollution_factor, "N", lat, lon)],  # "N" or "S" hemisphere
```

`pollution_factor` is informational only (not used for live data — real values come from the API).

---

## Deployment

### Streamlit Cloud (recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select `GaneshAdapnor/Air-Quality-Tracker`, branch `main`, file `app.py`
4. Click **Deploy**
5. Enter your OpenWeatherMap API key in the sidebar on first load

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Ganesh Adapnor** — [GitHub](https://github.com/GaneshAdapnor)

---

*AtmosShield uses the OpenWeatherMap Air Pollution API for live data. A free API key provides up to 60 calls/minute — more than sufficient for this dashboard's 30-minute cache.*
