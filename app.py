import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AtmosShield — Air Quality Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme palettes ───────────────────────────────────────────────────────────

LIGHT = dict(
    bg="#F0F4F8",
    card="#FFFFFF",
    card2="#F8FAFC",
    border="#E2E8F0",
    border2="#CBD5E1",
    text1="#0F172A",
    text2="#475569",
    text3="#94A3B8",
    accent="#2563EB",
    accent2="#1D4ED8",
    accent_soft="#EFF6FF",
    sidebar="#0F1C38",
    sidebar_card="#1A2D50",
    sidebar_border="#243B5E",
    metric_label="#64748B",
    metric_value="#0F172A",
    expander_hover="#F8FAFF",
    chart_template="plotly_white",
    chart_bg="#FFFFFF",
    chart_grid="#F1F5F9",
    chart_line="#E2E8F0",
    chart_tick="#64748B",
    section_head="#0F172A",
    progress_bg="#E2E8F0",
    banner_bg="#1E3A8A",
)

DARK = dict(
    bg="#0D1117",
    card="#161B22",
    card2="#1C2128",
    border="#21262D",
    border2="#30363D",
    text1="#F0F6FC",
    text2="#8D96A0",
    text3="#6E7681",
    accent="#3B82F6",
    accent2="#2563EB",
    accent_soft="#162032",
    sidebar="#090E1A",
    sidebar_card="#111827",
    sidebar_border="#1E2D45",
    metric_label="#8D96A0",
    metric_value="#F0F6FC",
    expander_hover="#1C2128",
    chart_template="plotly_dark",
    chart_bg="#161B22",
    chart_grid="#1C2128",
    chart_line="#21262D",
    chart_tick="#8D96A0",
    section_head="#F0F6FC",
    progress_bg="#1C2128",
    banner_bg="#0D1B35",
)

LIGHT_CELLS = {
    "#10B981": "background:#ECFDF5; color:#065F46; font-weight:700;",
    "#F59E0B": "background:#FFFBEB; color:#92400E; font-weight:700;",
    "#F97316": "background:#FFF7ED; color:#9A3412; font-weight:700;",
    "#EF4444": "background:#FEF2F2; color:#991B1B; font-weight:700;",
    "#8B5CF6": "background:#F5F3FF; color:#5B21B6; font-weight:700;",
    "#475569": "background:#F8FAFC; color:#334155; font-weight:700;",
}
DARK_CELLS = {
    "#10B981": "background:#052E16; color:#6EE7B7; font-weight:700;",
    "#F59E0B": "background:#451A03; color:#FDE68A; font-weight:700;",
    "#F97316": "background:#431407; color:#FDBA74; font-weight:700;",
    "#EF4444": "background:#450A0A; color:#FCA5A5; font-weight:700;",
    "#8B5CF6": "background:#2E1065; color:#C4B5FD; font-weight:700;",
    "#475569": "background:#1E293B; color:#CBD5E1; font-weight:700;",
}
DARK_CAT = {
    "Good":                  "background:#052E16; color:#6EE7B7; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Moderate":              "background:#451A03; color:#FDE68A; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Unhealthy (Sensitive)": "background:#431407; color:#FDBA74; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Unhealthy":             "background:#450A0A; color:#FCA5A5; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Very Unhealthy":        "background:#2E1065; color:#C4B5FD; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Hazardous":             "background:#1E293B; color:#CBD5E1; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
}
LIGHT_CAT = {
    "Good":                  "background:#ECFDF5; color:#065F46; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Moderate":              "background:#FFFBEB; color:#92400E; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Unhealthy (Sensitive)": "background:#FFF7ED; color:#9A3412; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Unhealthy":             "background:#FEF2F2; color:#991B1B; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Very Unhealthy":        "background:#F5F3FF; color:#5B21B6; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
    "Hazardous":             "background:#F8FAFC; color:#334155; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:20px;",
}

AQI_COLORS = {
    "Good":                  ("#10B981", "#ECFDF5", "#065F46"),
    "Moderate":              ("#F59E0B", "#FFFBEB", "#92400E"),
    "Unhealthy (Sensitive)": ("#F97316", "#FFF7ED", "#9A3412"),
    "Unhealthy":             ("#EF4444", "#FEF2F2", "#991B1B"),
    "Very Unhealthy":        ("#8B5CF6", "#F5F3FF", "#5B21B6"),
    "Hazardous":             ("#64748B", "#F8FAFC", "#334155"),
}


# ─── CSS injection ────────────────────────────────────────────────────────────

def inject_css(T):
    bg, card, card2 = T["bg"], T["card"], T["card2"]
    bd, bd2 = T["border"], T["border2"]
    t1, t2, t3 = T["text1"], T["text2"], T["text3"]
    acc, acc2, acc_s = T["accent"], T["accent2"], T["accent_soft"]
    sb, sb_c, sb_b = T["sidebar"], T["sidebar_card"], T["sidebar_border"]
    p_bg = T["progress_bg"]

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800;0,14..32,900&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; }}

    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: {bg} !important;
        -webkit-font-smoothing: antialiased;
    }}
    [data-testid="stAppViewContainer"] {{ background: {bg} !important; }}
    [data-testid="stMain"] {{ background: {bg} !important; }}
    .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }}
    [data-testid="stMainBlockContainer"] {{
        max-width: 100% !important;
        padding: 1.5rem 2rem 3rem !important;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ══ SIDEBAR — always visible, never collapsible ══ */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[aria-label="Close sidebar"],
    button[aria-label="Collapse sidebar"] {{
        display: none !important;
    }}
    [data-testid="stSidebar"] {{
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        width: 290px !important;
        min-width: 290px !important;
        max-width: 290px !important;
        background: {sb} !important;
        border-right: 1px solid {sb_b} !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: {sb} !important;
        padding: 0 !important;
    }}
    [data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div > div,
    [data-testid="stSidebar"] [data-baseweb="select"] button {{
        background: {sb_c} !important;
        background-color: {sb_c} !important;
        border: 1px solid {sb_b} !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        transition: border-color 0.2s ease !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {{
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] p {{
        color: #E2E8F0 !important;
        background: transparent !important;
    }}
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] {{
        background: #1A2D50 !important;
        color: #E2E8F0 !important;
        border: 1px solid #243B5E !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
        padding: 4px !important;
    }}
    [data-baseweb="popover"] [role="option"] {{
        border-radius: 6px !important;
        margin: 1px 4px !important;
        font-size: 0.84rem !important;
    }}
    [data-baseweb="popover"] [role="option"]:hover {{
        background: #243B5E !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] input[type="text"] {{
        background: {sb_c} !important;
        border: 1px solid {sb_b} !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="input"] input:focus {{
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
        outline: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMetric"] {{
        background: {sb_c} !important;
        border: 1px solid {sb_b} !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] > div {{
        color: #7DD3FC !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    [data-testid="stSidebar"] [data-testid="stMetricValue"] > div {{
        color: #F0F9FF !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] > div {{
        font-size: 0.72rem !important;
        color: #94A3B8 !important;
    }}

    /* ══ METRIC CARDS ══ */
    [data-testid="stMetric"] {{
        background: {card} !important;
        border: 1px solid {bd} !important;
        border-radius: 12px !important;
        padding: 20px 20px 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04) !important;
        transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    [data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 20px rgba(0,0,0,0.10) !important;
        transform: translateY(-2px) !important;
        border-color: {acc} !important;
    }}
    [data-testid="stMetricLabel"] > div {{
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: {T["metric_label"]} !important;
        margin-bottom: 2px !important;
    }}
    [data-testid="stMetricValue"] > div {{
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: {T["metric_value"]} !important;
        letter-spacing: -1px;
        line-height: 1.1 !important;
    }}
    [data-testid="stMetricDelta"] > div {{
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }}

    /* ══ PROGRESS BAR ══ */
    [data-testid="stProgress"] > div {{
        background: {p_bg} !important;
        border-radius: 99px !important;
        height: 8px !important;
    }}
    [data-testid="stProgress"] > div > div {{
        background: linear-gradient(90deg, #3B82F6, #2563EB) !important;
        border-radius: 99px !important;
    }}

    /* ══ EXPANDER ══ */
    [data-testid="stExpander"] {{
        background: {card} !important;
        border: 1px solid {bd} !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
        overflow: hidden !important;
        transition: box-shadow 0.2s, border-color 0.2s !important;
    }}
    [data-testid="stExpander"]:hover {{
        box-shadow: 0 4px 16px rgba(37,99,235,0.12) !important;
        border-color: {acc} !important;
    }}
    [data-testid="stExpander"] details summary {{
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: {t1} !important;
        padding: 16px 20px !important;
        cursor: pointer;
    }}
    [data-testid="stExpander"] details summary span {{
        color: {t1} !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }}
    [data-testid="stExpander"] details summary svg {{
        color: {acc} !important;
    }}

    /* ══ DATAFRAME ══ */
    [data-testid="stDataFrame"] {{ border-radius: 10px !important; overflow: hidden; border: 1px solid {bd} !important; }}
    [data-testid="stDataFrame"] th {{
        background: {card2} !important;
        color: {t2} !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.68rem !important;
        letter-spacing: 0.07em;
        padding: 10px 14px !important;
    }}
    [data-testid="stDataFrame"] td {{
        color: {t1} !important;
        font-size: 0.85rem !important;
        padding: 9px 14px !important;
    }}

    /* ══ BUTTON — theme toggle ══ */
    [data-testid="stButton"] > button {{
        background: {card} !important;
        border: 1px solid {bd2} !important;
        border-radius: 8px !important;
        padding: 7px 16px !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: {t1} !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
        transition: all 0.18s ease !important;
        letter-spacing: 0.01em !important;
        cursor: pointer !important;
    }}
    [data-testid="stButton"] > button:hover {{
        background: {acc_s} !important;
        border-color: {acc} !important;
        color: {acc} !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.18) !important;
        transform: translateY(-1px) !important;
    }}

    /* ══ MAIN TEXT ══ */
    .main .block-container p,
    .main .block-container span,
    .main .block-container li,
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container label,
    .main .block-container small,
    .main .block-container strong {{
        color: {t1} !important;
    }}
    .main [data-testid="stMarkdownContainer"],
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] span {{
        color: {t1} !important;
    }}
    .main [data-testid="stCaptionContainer"] * {{ color: {t2} !important; }}
    .main [data-baseweb="select"] span,
    .main [data-baseweb="select"] div {{ color: {t1} !important; }}
    .main [data-baseweb="input"] input {{ color: {t1} !important; }}
    .main [data-testid="stExpanderDetails"] * {{ color: {t1} !important; }}
    [data-testid="stSpinner"] p {{ color: {t2} !important; }}

    /* ══ CUSTOM COMPONENTS ══ */
    .kpi-card {{
        background: {card};
        border: 1px solid {bd};
        border-radius: 14px;
        padding: 22px 22px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 14px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s, transform 0.2s, border-color 0.2s;
        height: 100%;
    }}
    .kpi-card:hover {{
        box-shadow: 0 6px 24px rgba(0,0,0,0.10);
        transform: translateY(-2px);
        border-color: {acc};
    }}
    .kpi-label {{
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: {t2};
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1;
        color: {t1};
        margin-bottom: 10px;
    }}
    .kpi-badge {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }}
    .chart-wrap {{
        background: {card};
        border: 1px solid {bd};
        border-radius: 14px;
        padding: 8px 6px 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .section-title {{
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {t3};
        margin: 32px 0 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-title::after {{
        content: '';
        flex: 1;
        height: 1px;
        background: {bd};
    }}
    .info-card {{
        background: {card};
        border: 1px solid {bd};
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        height: 100%;
    }}
    .date-chip {{
        background: {acc_s};
        border: 1px solid {bd};
        border-radius: 8px;
        padding: 9px 14px;
        font-size: 0.75rem;
        font-weight: 600;
        color: {acc} !important;
        text-align: center;
        margin-top: 8px;
        letter-spacing: 0.01em;
    }}
    .stat-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid {bd};
    }}
    .stat-row:last-child {{ border-bottom: none; }}
    .stat-label {{ font-size: 0.78rem; font-weight: 500; color: {t2}; }}
    .stat-value {{ font-size: 0.88rem; font-weight: 700; color: {t1}; }}
    </style>
    """, unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────

WORLD_CITIES = {
    "India":          [("Delhi",1.65,"N"),("Mumbai",1.25,"N"),("Kolkata",1.40,"N"),("Chennai",0.88,"N"),
                       ("Bangalore",0.75,"N"),("Hyderabad",1.02,"N"),("Pune",0.90,"N"),("Ahmedabad",1.18,"N"),
                       ("Jaipur",1.22,"N"),("Lucknow",1.35,"N"),("Patna",1.45,"N"),("Bhopal",1.10,"N")],
    "Bangladesh":     [("Dhaka",1.72,"N"),("Chittagong",1.40,"N")],
    "Pakistan":       [("Karachi",1.55,"N"),("Lahore",1.68,"N"),("Islamabad",1.20,"N")],
    "Nepal":          [("Kathmandu",1.30,"N")],
    "Sri Lanka":      [("Colombo",0.85,"N")],
    "China":          [("Beijing",1.70,"N"),("Shanghai",1.45,"N"),("Guangzhou",1.35,"N"),
                       ("Shenzhen",1.25,"N"),("Chengdu",1.40,"N"),("Wuhan",1.38,"N"),
                       ("Xian",1.50,"N"),("Chongqing",1.42,"N"),("Tianjin",1.55,"N")],
    "Japan":          [("Tokyo",0.65,"N"),("Osaka",0.68,"N"),("Kyoto",0.60,"N"),("Nagoya",0.67,"N")],
    "South Korea":    [("Seoul",0.85,"N"),("Busan",0.80,"N"),("Incheon",0.82,"N")],
    "Taiwan":         [("Taipei",0.75,"N"),("Kaohsiung",0.80,"N")],
    "Indonesia":      [("Jakarta",1.30,"S"),("Surabaya",1.15,"S"),("Bandung",1.10,"S")],
    "Philippines":    [("Manila",1.25,"N"),("Cebu",1.10,"N")],
    "Vietnam":        [("Hanoi",1.20,"N"),("Ho Chi Minh City",1.15,"N")],
    "Thailand":       [("Bangkok",1.15,"N"),("Chiang Mai",0.90,"N")],
    "Malaysia":       [("Kuala Lumpur",0.95,"N"),("Penang",0.85,"N")],
    "Singapore":      [("Singapore",0.70,"N")],
    "Myanmar":        [("Yangon",1.20,"N")],
    "Cambodia":       [("Phnom Penh",1.10,"N")],
    "Saudi Arabia":   [("Riyadh",1.10,"N"),("Jeddah",1.05,"N"),("Mecca",1.08,"N")],
    "UAE":            [("Dubai",0.95,"N"),("Abu Dhabi",0.90,"N")],
    "Iran":           [("Tehran",1.30,"N"),("Isfahan",1.10,"N")],
    "Iraq":           [("Baghdad",1.25,"N"),("Basra",1.15,"N")],
    "Turkey":         [("Istanbul",0.90,"N"),("Ankara",0.85,"N"),("Izmir",0.78,"N")],
    "Israel":         [("Tel Aviv",0.75,"N"),("Jerusalem",0.70,"N")],
    "Kazakhstan":     [("Almaty",1.10,"N"),("Astana",1.05,"N")],
    "Uzbekistan":     [("Tashkent",1.15,"N")],
    "United Kingdom": [("London",0.65,"N"),("Birmingham",0.70,"N"),("Manchester",0.68,"N"),("Glasgow",0.60,"N")],
    "France":         [("Paris",0.70,"N"),("Lyon",0.62,"N"),("Marseille",0.65,"N")],
    "Germany":        [("Berlin",0.60,"N"),("Munich",0.55,"N"),("Hamburg",0.58,"N"),("Frankfurt",0.62,"N")],
    "Italy":          [("Rome",0.72,"N"),("Milan",0.78,"N"),("Naples",0.75,"N")],
    "Spain":          [("Madrid",0.68,"N"),("Barcelona",0.70,"N"),("Valencia",0.65,"N")],
    "Netherlands":    [("Amsterdam",0.58,"N"),("Rotterdam",0.60,"N")],
    "Belgium":        [("Brussels",0.65,"N")],
    "Switzerland":    [("Zurich",0.45,"N"),("Geneva",0.42,"N")],
    "Sweden":         [("Stockholm",0.42,"N"),("Gothenburg",0.44,"N")],
    "Norway":         [("Oslo",0.40,"N")],
    "Denmark":        [("Copenhagen",0.43,"N")],
    "Finland":        [("Helsinki",0.38,"N")],
    "Poland":         [("Warsaw",0.75,"N"),("Krakow",0.80,"N")],
    "Czech Republic": [("Prague",0.68,"N")],
    "Austria":        [("Vienna",0.55,"N")],
    "Portugal":       [("Lisbon",0.62,"N"),("Porto",0.58,"N")],
    "Greece":         [("Athens",0.72,"N")],
    "Hungary":        [("Budapest",0.70,"N")],
    "Romania":        [("Bucharest",0.82,"N")],
    "Ukraine":        [("Kyiv",0.80,"N"),("Kharkiv",0.82,"N")],
    "Russia":         [("Moscow",0.85,"N"),("Saint Petersburg",0.78,"N"),("Novosibirsk",0.88,"N")],
    "United States":  [("New York",0.72,"N"),("Los Angeles",0.82,"N"),("Chicago",0.75,"N"),
                       ("Houston",0.80,"N"),("Phoenix",0.78,"N"),("Dallas",0.76,"N"),
                       ("Seattle",0.55,"N"),("Miami",0.65,"N"),("Denver",0.60,"N"),
                       ("Boston",0.65,"N"),("Atlanta",0.72,"N"),("San Francisco",0.58,"N")],
    "Canada":         [("Toronto",0.60,"N"),("Vancouver",0.48,"N"),("Montreal",0.58,"N"),("Calgary",0.52,"N")],
    "Mexico":         [("Mexico City",1.15,"N"),("Guadalajara",0.95,"N"),("Monterrey",1.00,"N")],
    "Brazil":         [("Sao Paulo",0.95,"S"),("Rio de Janeiro",0.88,"S"),("Brasilia",0.72,"S"),("Belo Horizonte",0.85,"S")],
    "Argentina":      [("Buenos Aires",0.82,"S"),("Cordoba",0.75,"S")],
    "Colombia":       [("Bogota",0.90,"N"),("Medellin",0.85,"N")],
    "Chile":          [("Santiago",0.88,"S")],
    "Peru":           [("Lima",0.92,"S")],
    "Venezuela":      [("Caracas",0.95,"N")],
    "Ecuador":        [("Quito",0.70,"S")],
    "Egypt":          [("Cairo",1.25,"N"),("Alexandria",1.10,"N")],
    "Nigeria":        [("Lagos",1.30,"N"),("Abuja",1.00,"N"),("Kano",1.20,"N")],
    "South Africa":   [("Johannesburg",0.92,"S"),("Cape Town",0.72,"S"),("Durban",0.85,"S")],
    "Kenya":          [("Nairobi",0.88,"S")],
    "Ethiopia":       [("Addis Ababa",0.95,"N")],
    "Ghana":          [("Accra",0.92,"N")],
    "Morocco":        [("Casablanca",0.90,"N"),("Rabat",0.82,"N")],
    "Tanzania":       [("Dar es Salaam",0.88,"S")],
    "Algeria":        [("Algiers",0.95,"N")],
    "Australia":      [("Sydney",0.48,"S"),("Melbourne",0.45,"S"),("Brisbane",0.50,"S"),
                       ("Perth",0.42,"S"),("Adelaide",0.44,"S")],
    "New Zealand":    [("Auckland",0.38,"S"),("Wellington",0.35,"S")],
}

CITIES_COUNTRIES = {city: country for country, cities in WORLD_CITIES.items() for city,_,_ in cities}
CITY_FACTOR      = {city: factor  for cities in WORLD_CITIES.values() for city, factor, _ in cities}
CITY_HEMI        = {city: hemi    for cities in WORLD_CITIES.values() for city, _, hemi in cities}
FEATURES = ["pm25","pm10","co","no2","o3","so2","temperature","humidity",
            "wind_speed","month","day_of_year","city_enc"]
TARGETS  = ["outdoor_aqi","indoor_aqi","outdoor_purified","indoor_purified"]
CHART_COLORS = {
    "outdoor_aqi":      "#EF4444",
    "indoor_aqi":       "#3B82F6",
    "outdoor_purified": "#10B981",
    "indoor_purified":  "#A78BFA",
}
CHART_NAMES = {
    "outdoor_aqi":      "Outdoor AQI",
    "indoor_aqi":       "Indoor AQI",
    "outdoor_purified": "Outdoor Purified",
    "indoor_purified":  "Indoor Purified",
}

PM25_BP = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
           (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500,301,500)]
PM10_BP = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
           (255,354,151,200),(355,424,201,300),(425,604,301,500)]


# ─── AQI helpers ──────────────────────────────────────────────────────────────

def _sub(c, bp):
    for c_lo, c_hi, i_lo, i_hi in bp:
        if c_lo <= c <= c_hi:
            return (i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo
    return 500.0

def calc_aqi(pm25, pm10, co, no2, o3, so2):
    return min(500.0, max(0.0, max(
        _sub(max(0,pm25), PM25_BP), _sub(max(0,pm10), PM10_BP),
        min(500, co*14), min(500, no2*2.6), min(500, o3*2.1), min(500, so2*4.2),
    )))

def aqi_meta(aqi):
    if aqi <= 50:  return "Good",                  "#10B981", "good"
    if aqi <= 100: return "Moderate",              "#F59E0B", "moderate"
    if aqi <= 150: return "Unhealthy (Sensitive)", "#F97316", "usg"
    if aqi <= 200: return "Unhealthy",             "#EF4444", "unhealthy"
    if aqi <= 300: return "Very Unhealthy",        "#8B5CF6", "very"
    return               "Hazardous",              "#64748B", "hazardous"


# ─── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_dataset():
    rng  = np.random.default_rng(42)
    base = datetime(2024, 1, 1)
    rows = []
    for city, country in CITIES_COUNTRIES.items():
        f = CITY_FACTOR[city]
        for day in range(365):
            date = base + timedelta(days=day)
            m    = date.month
            hemi = CITY_HEMI[city]
            s    = (1.55 if m in (6,7,8) else (0.62 if m in (12,1,2) else 1.0)) if hemi=="S" \
                   else (1.55 if m in (11,12,1,2) else (0.62 if m in (6,7,8,9) else 1.0))
            pm25 = rng.uniform(12, 130)*f*s; pm10 = rng.uniform(25, 220)*f*s
            co   = rng.uniform(0.3, 5.0)*f;  no2  = rng.uniform(6,  75)*f*s
            o3   = rng.uniform(12, 95);       so2  = rng.uniform(2,  38)*f*s
            temp = rng.uniform(8, 46);        hum  = rng.uniform(22, 96)
            wind = rng.uniform(0.4, 26)
            out  = calc_aqi(pm25, pm10, co, no2, o3, so2)
            inn  = out * rng.uniform(0.52, 0.82)
            rows.append({
                "city": city, "state": country, "date": date, "month": m, "day_of_year": day+1,
                "pm25": pm25, "pm10": pm10, "co": co, "no2": no2, "o3": o3, "so2": so2,
                "temperature": temp, "humidity": hum, "wind_speed": wind,
                "outdoor_aqi": out, "indoor_aqi": inn,
                "outdoor_purified": out * rng.uniform(0.13, 0.32),
                "indoor_purified":  inn * rng.uniform(0.09, 0.24),
            })
    return pd.DataFrame(rows)


# ─── XGBoost ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_models(_df):
    le = LabelEncoder()
    df = _df.copy()
    df["city_enc"] = le.fit_transform(df["city"])
    X = df[FEATURES]; trained, metrics = {}, {}
    for target in TARGETS:
        y = df[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        mdl = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                gamma=0.1, random_state=42, n_jobs=-1, verbosity=0)
        mdl.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        preds = mdl.predict(Xte)
        metrics[target] = {"r2": round(r2_score(yte, preds), 4),
                           "rmse": round(np.sqrt(mean_squared_error(yte, preds)), 2)}
        trained[target] = mdl
    return trained, le, metrics

def predict_city(models, le, city, df):
    cdf = df[df["city"] == city].copy()
    cdf["city_enc"] = le.transform(cdf["city"])
    X = cdf[FEATURES]
    for t, m in models.items():
        cdf[f"pred_{t}"] = m.predict(X)
    return cdf.sort_values("day_of_year")

def forecast_7days(models, le, city, df):
    cdf  = predict_city(models, le, city, df)
    last = cdf.iloc[-1]
    rng  = np.random.default_rng(7)
    rows = []
    for i in range(1, 8):
        row = last.copy()
        row["day_of_year"] = int((last["day_of_year"]+i-1) % 365 + 1)
        row["month"] = ((int(last["month"])-1+i//30) % 12)+1
        noise = rng.normal(1.0, 0.06)
        for p in ("pm25","pm10","co","no2","o3","so2"):
            row[p] = max(0.0, row[p]*noise)
        row["city_enc"] = le.transform([city])[0]
        Xp = pd.DataFrame([row[FEATURES]])
        entry = {"date": last["date"] + timedelta(days=i)}
        for t, m in models.items():
            entry[f"pred_{t}"] = float(m.predict(Xp)[0])
        rows.append(entry)
    return pd.DataFrame(rows)


# ─── Chart builders ───────────────────────────────────────────────────────────

def base_layout(T, height):
    return dict(
        template=T["chart_template"],
        font=dict(family="Inter, sans-serif", size=12, color=T["text2"]),
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor=T["chart_bg"],
        plot_bgcolor=T["chart_bg"],
        legend=dict(
            orientation="h", y=1.12, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=T["text2"]),
            itemwidth=30,
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showline=True,
            linecolor=T["chart_line"],
            tickfont=dict(size=11, color=T["chart_tick"]),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=T["chart_grid"], gridwidth=1,
            zeroline=False, showline=False,
            tickfont=dict(size=11, color=T["chart_tick"]),
        ),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=T["card2"],
            font_color=T["text1"],
            font_size=12,
            bordercolor=T["border"],
        ),
    )

def make_trend_chart(trend_df, T):
    fig = go.Figure()
    for y0, y1, fc in [(0,50,"#ECFDF5"),(50,100,"#FFFBEB"),(100,150,"#FFF7ED"),(150,200,"#FEF2F2")]:
        op = 0.18 if T == DARK else 0.50
        fig.add_hrect(y0=y0, y1=y1, fillcolor=fc, opacity=op, line_width=0, layer="below")
    dashes = {"outdoor_aqi":"solid","indoor_aqi":"solid",
              "outdoor_purified":"dot","indoor_purified":"dot"}
    widths = {"outdoor_aqi":2.5,"indoor_aqi":2.5,"outdoor_purified":1.8,"indoor_purified":1.8}
    for t in TARGETS:
        fig.add_trace(go.Scatter(
            x=trend_df["date"], y=trend_df[f"pred_{t}"],
            name=CHART_NAMES[t], mode="lines",
            line=dict(color=CHART_COLORS[t], width=widths[t], dash=dashes[t]),
            hovertemplate=f"<b>{CHART_NAMES[t]}</b>: %{{y:.0f}}<extra></extra>",
        ))
    fig.update_layout(**{**base_layout(T, 340), "yaxis_title": "AQI Index"})
    return fig

def make_forecast_chart(fdf, T):
    fig = go.Figure()
    for t in TARGETS:
        fig.add_trace(go.Scatter(
            x=fdf["date"], y=fdf[f"pred_{t}"],
            name=CHART_NAMES[t], mode="lines+markers",
            line=dict(color=CHART_COLORS[t], width=2.5),
            marker=dict(size=8, color=CHART_COLORS[t],
                        line=dict(width=2, color=T["chart_bg"])),
            hovertemplate=f"<b>{CHART_NAMES[t]}</b>: %{{y:.0f}}<extra></extra>",
        ))
    fig.update_layout(**{**base_layout(T, 320), "yaxis_title": "Predicted AQI"})
    return fig

def make_purif_chart(b_out, a_out, b_in, a_in, T):
    fig = go.Figure([
        go.Bar(
            name="Before Purification",
            x=["Outdoor","Indoor"], y=[b_out, b_in],
            marker=dict(color=["#FCA5A5","#93C5FD"],
                        line=dict(width=0), cornerradius=6),
            text=[f"{b_out:.0f}",f"{b_in:.0f}"], textposition="outside",
            textfont=dict(color=T["text1"], size=13, family="Inter"),
            hovertemplate="<b>%{x}</b> Before: %{y:.0f}<extra></extra>",
        ),
        go.Bar(
            name="After Purification",
            x=["Outdoor","Indoor"], y=[a_out, a_in],
            marker=dict(color=["#6EE7B7","#C4B5FD"],
                        line=dict(width=0), cornerradius=6),
            text=[f"{a_out:.0f}",f"{a_in:.0f}"], textposition="outside",
            textfont=dict(color=T["text1"], size=13, family="Inter"),
            hovertemplate="<b>%{x}</b> After: %{y:.0f}<extra></extra>",
        ),
    ])
    layout = {
        **base_layout(T, 280),
        "barmode": "group", "bargap": 0.28, "bargroupgap": 0.06,
        "yaxis": dict(showgrid=True, gridcolor=T["chart_grid"], zeroline=False,
                      range=[0, max(b_out,b_in)*1.30],
                      tickfont=dict(size=11, color=T["chart_tick"])),
        "xaxis": dict(showgrid=False, showline=False,
                      tickfont=dict(size=13, color=T["text1"], family="Inter")),
    }
    fig.update_layout(**layout)
    return fig

def make_city_chart(cmp_df, T):
    top = cmp_df.head(30)
    colors = [aqi_meta(v)[1] for v in top["Outdoor AQI"]]
    fig = go.Figure(go.Bar(
        x=top["City"], y=top["Outdoor AQI"],
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=[f"{v:.0f}" for v in top["Outdoor AQI"]],
        textposition="outside",
        textfont=dict(color=T["text2"], size=10),
        hovertemplate="<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>",
    ))
    layout = {
        **base_layout(T, 320),
        "yaxis_title": "Outdoor AQI",
        "yaxis": dict(showgrid=True, gridcolor=T["chart_grid"], zeroline=False,
                      range=[0, top["Outdoor AQI"].max()*1.22],
                      tickfont=dict(size=10, color=T["chart_tick"])),
        "xaxis": dict(showgrid=False, showline=False, tickangle=-35,
                      tickfont=dict(size=10, color=T["text2"])),
    }
    fig.update_layout(**layout)
    return fig

def make_pollutant_chart(row, T):
    names = ["PM2.5","PM10","CO","NO₂","O₃","SO₂"]
    vals  = [row.pm25, row.pm10, row.co, row.no2, row.o3, row.so2]
    clrs  = ["#EF4444","#F97316","#F59E0B","#EAB308","#22C55E","#3B82F6"]
    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker=dict(color=clrs, line=dict(width=0), cornerradius=6),
        text=[f"{v:.1f}" for v in vals], textposition="outside",
        textfont=dict(color=T["text2"], size=11),
        hovertemplate="<b>%{x}</b>: %{y:.2f}<extra></extra>",
    ))
    layout = {
        **base_layout(T, 240),
        "yaxis": dict(showgrid=True, gridcolor=T["chart_grid"], zeroline=False,
                      range=[0, max(vals)*1.28],
                      tickfont=dict(size=10, color=T["chart_tick"])),
    }
    fig.update_layout(**layout)
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(df, metrics):
    import datetime as dt

    # Brand header
    st.sidebar.markdown("""
    <div style='padding:24px 20px 20px; border-bottom:1px solid #1E2D45;'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>
            <div style='background:linear-gradient(135deg,#1D4ED8,#0EA5E9);
                 border-radius:10px;width:36px;height:36px;display:flex;
                 align-items:center;justify-content:center;font-size:1.1rem;
                 flex-shrink:0;'>🌬️</div>
            <div>
                <div style='font-size:1.05rem;font-weight:800;color:#F0F9FF;
                     letter-spacing:-0.3px;line-height:1.2;'>AtmosShield</div>
                <div style='font-size:0.65rem;color:#64748B;font-weight:500;
                     letter-spacing:0.04em;text-transform:uppercase;'>
                     Air Quality Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Location section
    st.sidebar.markdown("""
    <div style='padding:16px 20px 8px;'>
        <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
             letter-spacing:0.12em;color:#475569;margin-bottom:10px;'>
             📍 Location</div>
    </div>
    """, unsafe_allow_html=True)

    states = sorted(df["state"].unique())
    sel_country = st.sidebar.selectbox("State / Country", states,
                                       label_visibility="visible", key="dd_state")
    if st.session_state.get("_last_state") != sel_country:
        st.session_state["_last_state"] = sel_country
        st.session_state.pop("dd_city", None)

    cities = sorted(df[df["state"] == sel_country]["city"].unique())
    sel_city = st.sidebar.selectbox("City", cities,
                                    label_visibility="visible", key="dd_city")

    # Date section
    st.sidebar.markdown("""
    <div style='padding:16px 20px 8px; border-top:1px solid #1E2D45; margin-top:8px;'>
        <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
             letter-spacing:0.12em;color:#475569;margin-bottom:10px;'>
             📅 Date Range</div>
    </div>
    """, unsafe_allow_html=True)

    from_str = st.sidebar.text_input("From", value="01/01/2024",
                                     placeholder="DD/MM/YYYY",
                                     label_visibility="visible", key="from_date_txt")
    to_str   = st.sidebar.text_input("To", value="31/12/2024",
                                     placeholder="DD/MM/YYYY",
                                     label_visibility="visible", key="to_date_txt")

    def _parse(s, fallback):
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y"):
            try:
                return dt.datetime.strptime(s.strip(), fmt).date()
            except Exception:
                pass
        return fallback

    from_date = _parse(from_str, dt.date(2024, 1, 1))
    to_date   = _parse(to_str,   dt.date(2024, 12, 31))
    if from_date > to_date:
        from_date, to_date = to_date, from_date

    days_selected = (to_date - from_date).days + 1
    st.sidebar.markdown(
        f"<div class='date-chip'>"
        f"<span style='font-weight:800;color:#7DD3FC;font-size:1rem;'>{days_selected}</span> days &nbsp;·&nbsp; "
        f"{from_date.strftime('%d %b')} → {to_date.strftime('%d %b %Y')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Model accuracy
    st.sidebar.markdown("""
    <div style='padding:16px 20px 8px; border-top:1px solid #1E2D45; margin-top:12px;'>
        <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
             letter-spacing:0.12em;color:#475569;margin-bottom:10px;'>
             🤖 Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    labels = {"outdoor_aqi":"Outdoor AQI","indoor_aqi":"Indoor AQI",
              "outdoor_purified":"Outdoor Purified","indoor_purified":"Indoor Purified"}
    for t, m in metrics.items():
        st.sidebar.metric(labels[t], f"R² {m['r2']}", f"RMSE {m['rmse']}")

    # Footer
    st.sidebar.markdown("""
    <div style='position:absolute;bottom:0;left:0;right:0;padding:14px 20px;
         border-top:1px solid #1E2D45;'>
        <div style='font-size:0.65rem;color:#334155;text-align:center;font-weight:500;'>
            Powered by XGBoost · 120+ Cities
        </div>
    </div>
    """, unsafe_allow_html=True)

    return sel_city, sel_country, from_date, to_date


# ─── KPI card helper ──────────────────────────────────────────────────────────

def kpi_card(col, icon, label, value, _T):
    lbl, color, _ = aqi_meta(value)
    ac_info = AQI_COLORS.get(lbl, ("#64748B","#F8FAFC","#334155"))
    bg_badge, fg_badge = ac_info[1], ac_info[2]
    col.markdown(
        f"""<div class="kpi-card">
            <div style="width:36px;height:3px;background:{color};
                 border-radius:99px;margin-bottom:14px;"></div>
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value" style="color:{color};">{value:.0f}</div>
            <div class="kpi-badge" style="background:{bg_badge};color:{fg_badge};">{lbl}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def weather_card(col, icon, label, value_str, T):
    col.markdown(
        f"""<div style="background:{T['card']};border:1px solid {T['border']};
             border-radius:12px;padding:16px 18px;
             box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:0.10em;color:{T['text3']};margin-bottom:6px;">{icon} {label}</div>
            <div style="font-size:1.55rem;font-weight:800;letter-spacing:-0.5px;
                 color:{T['text1']};">{value_str}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    with st.spinner("Initialising models…"):
        df = generate_dataset()
        models, le, metrics = train_models(df)

    is_dark = st.session_state.get("dark_mode", False)
    T = DARK if is_dark else LIGHT
    inject_css(T)

    sel_city, sel_country, from_date, to_date = render_sidebar(df, metrics)

    cell_styles = DARK_CELLS if is_dark else LIGHT_CELLS
    cat_styles  = DARK_CAT   if is_dark else LIGHT_CAT

    city_df = predict_city(models, le, sel_city, df)

    from_key = from_date.month * 100 + from_date.day
    to_key   = to_date.month   * 100 + to_date.day
    md_key   = city_df["date"].dt.month * 100 + city_df["date"].dt.day

    if from_key <= to_key:
        mask = (md_key >= from_key) & (md_key <= to_key)
    else:
        mask = (md_key >= from_key) | (md_key <= to_key)

    filtered_df = city_df[mask]
    if filtered_df.empty:
        filtered_df = city_df

    cur  = filtered_df.iloc[-1]
    prev = filtered_df.iloc[0]

    lbl_cur, color_cur, _ = aqi_meta(cur["pred_outdoor_aqi"])

    # ── Top bar ───────────────────────────────────────────────────────────
    top_left, top_right = st.columns([0.75, 0.25])
    with top_left:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;padding:4px 0 18px;'>"
            f"<div style='font-size:1.35rem;font-weight:800;color:{T['text1']};letter-spacing:-0.5px;'>"
            f"Air Quality Dashboard</div>"
            f"<div style='background:{T['accent_soft']};border:1px solid {T['border']};border-radius:20px;"
            f"padding:4px 14px;font-size:0.75rem;font-weight:600;color:{T['accent']};'>"
            f"📍 {sel_city}, {sel_country}</div>"
            f"<div style='background:{color_cur};border-radius:20px;padding:4px 12px;"
            f"font-size:0.72rem;font-weight:700;color:white;'>{lbl_cur}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with top_right:
        btn_label = "☀️ Light mode" if is_dark else "🌙 Dark mode"
        if st.button(btn_label, key="theme_btn"):
            st.session_state.dark_mode = not is_dark
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — CURRENT CONDITIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<div class='section-title'>Current Conditions &nbsp;·&nbsp; "
        f"{from_date.strftime('%d %b')} – {to_date.strftime('%d %b %Y')}</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4, gap="small")
    kpi_card(k1, "🌍", "Outdoor AQI",      cur["pred_outdoor_aqi"],      T)
    kpi_card(k2, "🏠", "Indoor AQI",       cur["pred_indoor_aqi"],       T)
    kpi_card(k3, "✨", "Outdoor Purified", cur["pred_outdoor_purified"], T)
    kpi_card(k4, "💨", "Indoor Purified",  cur["pred_indoor_purified"],  T)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    w1, w2, w3, w4 = st.columns(4, gap="small")
    weather_card(w1, "🌡️", "Temperature",  f"{cur['temperature']:.1f} °C", T)
    weather_card(w2, "💧", "Humidity",     f"{cur['humidity']:.0f} %",     T)
    weather_card(w3, "💨", "Wind Speed",   f"{cur['wind_speed']:.1f} km/h",T)
    weather_card(w4, "🔬", "PM2.5",        f"{cur['pm25']:.1f} µg/m³",    T)

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    with st.expander("🧪 Full Pollutant Breakdown", expanded=False):
        st.plotly_chart(make_pollutant_chart(cur, T), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — AQI TREND
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>AQI Trend</div>", unsafe_allow_html=True)

    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.plotly_chart(make_trend_chart(filtered_df, T), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — PURIFICATION & COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>Before vs After Purification</div>", unsafe_allow_html=True)

    out_red = (1 - cur["pred_outdoor_purified"] / cur["pred_outdoor_aqi"]) * 100
    in_red  = (1 - cur["pred_indoor_purified"]  / cur["pred_indoor_aqi"])  * 100
    avg_red = (out_red + in_red) / 2

    pur_left, pur_right = st.columns([3, 2], gap="large")
    with pur_left:
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(
            make_purif_chart(cur["pred_outdoor_aqi"], cur["pred_outdoor_purified"],
                             cur["pred_indoor_aqi"],  cur["pred_indoor_purified"], T),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with pur_right:
        st.markdown(f"<p style='font-size:0.62rem;font-weight:700;text-transform:uppercase;"
                    f"letter-spacing:0.12em;color:{T['text3']};margin:4px 0 18px;'>"
                    f"Purification Efficiency</p>", unsafe_allow_html=True)

        # Outdoor reduction row
        r1, v1 = st.columns([4, 1])
        r1.markdown(f"<p style='font-size:0.82rem;font-weight:600;color:{T['text1']};margin:0;'>"
                    f"🌍 Outdoor Reduction</p>", unsafe_allow_html=True)
        v1.markdown(f"<p style='font-size:1rem;font-weight:800;color:#10B981;"
                    f"text-align:right;margin:0;'>{out_red:.0f}%</p>", unsafe_allow_html=True)
        st.markdown(f"""<div style='background:{T['progress_bg']};border-radius:99px;
            height:7px;margin-bottom:16px;'>
            <div style='background:linear-gradient(90deg,#10B981,#34D399);border-radius:99px;
            height:7px;width:{min(out_red,100):.0f}%;'></div></div>""", unsafe_allow_html=True)

        # Indoor reduction row
        r2, v2 = st.columns([4, 1])
        r2.markdown(f"<p style='font-size:0.82rem;font-weight:600;color:{T['text1']};margin:0;'>"
                    f"🏠 Indoor Reduction</p>", unsafe_allow_html=True)
        v2.markdown(f"<p style='font-size:1rem;font-weight:800;color:#A78BFA;"
                    f"text-align:right;margin:0;'>{in_red:.0f}%</p>", unsafe_allow_html=True)
        st.markdown(f"""<div style='background:{T['progress_bg']};border-radius:99px;
            height:7px;margin-bottom:20px;'>
            <div style='background:linear-gradient(90deg,#A78BFA,#8B5CF6);border-radius:99px;
            height:7px;width:{min(in_red,100):.0f}%;'></div></div>""", unsafe_allow_html=True)

        # Average stat box
        st.markdown(f"""<div style='background:{T['accent_soft']};border:1px solid {T['border']};
            border-radius:10px;padding:16px;text-align:center;'>
            <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.12em;color:{T['text3']};margin-bottom:6px;'>Average Reduction</div>
            <div style='font-size:2.4rem;font-weight:800;color:{T['accent']};
            letter-spacing:-1px;line-height:1;'>{avg_red:.0f}%</div>
            <div style='font-size:0.72rem;color:{T['text2']};margin-top:6px;'>
            significant air quality improvement</div></div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — PREVIOUS VS CURRENT
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>Period Comparison</div>", unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4, gap="small")
    for col, icon, label, field in [
        (d1,"🌍","Outdoor AQI",      "pred_outdoor_aqi"),
        (d2,"🏠","Indoor AQI",       "pred_indoor_aqi"),
        (d3,"✨","Outdoor Purified", "pred_outdoor_purified"),
        (d4,"💨","Indoor Purified",  "pred_indoor_purified"),
    ]:
        delta = cur[field] - prev[field]
        col.metric(
            f"{icon} {label}", f"{cur[field]:.0f}",
            f"{delta:+.0f} vs {prev['date'].strftime('%d %b')}",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — 7-DAY FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>7-Day Forecast</div>", unsafe_allow_html=True)

    fdf = forecast_7days(models, le, sel_city, df)

    fc_chart, fc_table = st.columns([3, 2], gap="large")
    with fc_chart:
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(make_forecast_chart(fdf, T), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with fc_table:
        table = fdf[["date"] + [f"pred_{t}" for t in TARGETS]].copy()
        table.columns = ["Date","Outdoor AQI","Indoor AQI","Outdoor Purified","Indoor Purified"]
        table["Date"] = table["Date"].dt.strftime("%a %d %b")

        def color_cell(val):
            if not isinstance(val, float): return ""
            return cell_styles.get(aqi_meta(val)[1],
                   f"background:{T['card2']};color:{T['text1']};font-weight:700;")

        st.dataframe(
            table.round(1).style
                 .map(color_cell, subset=table.columns[1:])
                 .format({c: "{:.1f}" for c in table.columns[1:]}),
            use_container_width=True, hide_index=True, height=270,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — GLOBAL RANKINGS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>Global City Rankings</div>", unsafe_allow_html=True)

    city_rows = []
    for city in CITIES_COUNTRIES:
        cdf = predict_city(models, le, city, df)
        row = cdf.iloc[-1]
        lbl2, clr, _ = aqi_meta(row["pred_outdoor_aqi"])
        city_rows.append({
            "City": city, "Country": CITIES_COUNTRIES[city],
            "Outdoor AQI":  round(row["pred_outdoor_aqi"],     1),
            "Indoor AQI":   round(row["pred_indoor_aqi"],      1),
            "Outdoor ✨":   round(row["pred_outdoor_purified"],1),
            "Indoor 💨":    round(row["pred_indoor_purified"], 1),
            "Category": lbl2, "_color": clr,
        })

    cmp = pd.DataFrame(city_rows).sort_values("Outdoor AQI", ascending=False)

    gc_chart, gc_table = st.columns([3, 2], gap="large")
    with gc_chart:
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(make_city_chart(cmp, T), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with gc_table:
        display = cmp[["City","Country","Outdoor AQI","Category"]].reset_index(drop=True)

        def aqiv_style(val):
            if isinstance(val, float):
                return cell_styles.get(aqi_meta(val)[1],
                       f"background:{T['card2']};color:{T['text1']};font-weight:700;")
            return ""

        def cat_style(val):
            return cat_styles.get(val,
                   f"background:{T['card2']};color:{T['text1']};font-weight:600;font-size:0.75rem;")

        st.dataframe(
            display.style.map(aqiv_style, subset=["Outdoor AQI"])
                         .map(cat_style,  subset=["Category"]),
            use_container_width=True, height=360, hide_index=True,
        )

    # AQI scale legend
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{T['card']};border:1px solid {T['border']};border-radius:10px;
         padding:12px 20px;display:flex;flex-wrap:wrap;gap:18px;align-items:center;'>
        <span style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
              letter-spacing:0.10em;color:{T['text3']};'>AQI Index:</span>
        <span style='display:flex;align-items:center;gap:6px;font-size:0.75rem;
              font-weight:600;color:{T['text2']};'>
            <span style='width:10px;height:10px;border-radius:50%;background:#10B981;
                  display:inline-block;'></span>0–50 Good</span>
        <span style='display:flex;align-items:center;gap:6px;font-size:0.75rem;
              font-weight:600;color:{T['text2']};'>
            <span style='width:10px;height:10px;border-radius:50%;background:#F59E0B;
                  display:inline-block;'></span>51–100 Moderate</span>
        <span style='display:flex;align-items:center;gap:6px;font-size:0.75rem;
              font-weight:600;color:{T['text2']};'>
            <span style='width:10px;height:10px;border-radius:50%;background:#F97316;
                  display:inline-block;'></span>101–150 Sensitive</span>
        <span style='display:flex;align-items:center;gap:6px;font-size:0.75rem;
              font-weight:600;color:{T['text2']};'>
            <span style='width:10px;height:10px;border-radius:50%;background:#EF4444;
                  display:inline-block;'></span>151–200 Unhealthy</span>
        <span style='display:flex;align-items:center;gap:6px;font-size:0.75rem;
              font-weight:600;color:{T['text2']};'>
            <span style='width:10px;height:10px;border-radius:50%;background:#8B5CF6;
                  display:inline-block;'></span>201–300 Very Unhealthy</span>
        <span style='display:flex;align-items:center;gap:6px;font-size:0.75rem;
              font-weight:600;color:{T['text2']};'>
            <span style='width:10px;height:10px;border-radius:50%;background:#64748B;
                  display:inline-block;'></span>301+ Hazardous</span>
    </div>
    <div style='height:32px;'></div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
