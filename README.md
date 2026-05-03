# 🌬️ AtmosShield — Air Quality Intelligence Platform

> Real-time indoor & outdoor air quality prediction powered by XGBoost, covering 120+ cities across 50+ countries.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-FF6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)

---

## Overview

**AtmosShield** is a full-stack air quality dashboard that trains four XGBoost regression models to predict:

| Metric | Description |
|---|---|
| 🌍 **Outdoor AQI** | Raw ambient air quality index |
| 🏠 **Indoor AQI** | Estimated indoor air quality |
| ✨ **Outdoor Purified** | Post-purification outdoor AQI |
| 💨 **Indoor Purified** | Post-purification indoor AQI |

The app ships a synthetic dataset of 365 daily readings per city (120+ cities), uses hemisphere-aware seasonal multipliers for realistic pollution cycles, and maps any selected date range year-agnostically onto the underlying 2024 dataset.

---

## Features

- **XGBoost ML** — four independent `XGBRegressor` models, one per target metric
- **Global coverage** — 120+ cities, 50+ countries, from Delhi to Wellington
- **Dark / Light theme** — single-click toggle, sidebar always stays dark
- **Sidebar controls** — country → city cascade dropdown + manual DD/MM/YYYY date range
- **Current conditions** — AQI KPI cards, weather metrics, full pollutant breakdown
- **Purification analysis** — before/after comparison chart with efficiency percentages
- **Period comparison** — delta metrics vs. start of selected range
- **7-day forecast** — chart + colour-coded table
- **Global rankings** — top-30 city bar chart + full sortable table
- **AQI legend** — inline colour scale for quick reference

---

## Screenshots

| Dark Mode | Light Mode |
|---|---|
| ![dark](https://via.placeholder.com/500x280/0D1117/58A6FF?text=Dark+Mode) | ![light](https://via.placeholder.com/500x280/F0F4F8/2563EB?text=Light+Mode) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.32+ |
| ML Models | XGBoost 2.0+, scikit-learn |
| Data | Pandas, NumPy |
| Charts | Plotly |
| Styling | Custom CSS via `st.markdown` |

---

## Project Structure

```
Air-Quality-Tracker/
├── app.py              # Main application (data, models, UI)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

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

The app will open at `http://localhost:8501`.

---

## How It Works

### 1. Dataset Generation

A synthetic dataset is generated once and cached (`@st.cache_data`). Each city gets 365 daily rows with:

- **Pollutants** — PM2.5, PM10, CO, NO₂, O₃, SO₂ scaled by a city-specific pollution factor
- **Seasonality** — hemisphere-aware multipliers (Northern/Southern winter peaks)
- **Weather** — temperature, humidity, wind speed
- **AQI** — computed from EPA breakpoint sub-indices

### 2. Model Training

Four `XGBRegressor` models are trained in a single pass (`@st.cache_resource`) with an 80/20 train/test split:

```
Features: pm25, pm10, co, no2, o3, so2,
          temperature, humidity, wind_speed,
          month, day_of_year, city_enc
```

Typical accuracy: **R² > 0.97**, **RMSE < 5 AQI points**.

### 3. Year-Agnostic Date Filtering

The dataset is 2024-only, but users can enter any year. Dates are mapped via a `month×100 + day` integer key, so selecting `01/06/2026 → 30/11/2026` returns the same seasonal slice as `01/06/2024 → 30/11/2024`.

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

All city data, pollution factors, and hemisphere assignments live in the `WORLD_CITIES` constant in `app.py`. To add a new city:

```python
"Country Name": [("City Name", pollution_factor, "N")],  # "N" or "S"
```

`pollution_factor` is a multiplier relative to a baseline of 1.0. Clean cities like Helsinki use ~0.38; heavily polluted cities like Dhaka use ~1.72.

---

## Deployment

### Streamlit Cloud (recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select `GaneshAdapnor/Air-Quality-Tracker`, branch `main`, file `app.py`
4. Click **Deploy**

> First load trains the models (~60s). Subsequent loads use Streamlit's cache and are near-instant.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Ganesh Adapnor** — [GitHub](https://github.com/GaneshAdapnor)

---

*AtmosShield uses synthetic data for demonstration purposes. For real-world air quality data, integrate with APIs such as OpenAQ, IQAir, or the EPA AirNow service.*
