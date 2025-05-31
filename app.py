import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np
import re

# --- App Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("⚽ Football Manager 2024 Squad & Transfer Analyzer")
st.markdown(
    """
Upload your FM24 exported squad and transfer market HTML files to analyze your squad and available players.
Ask AI questions about your squad or transfer targets, and view detailed player stats with radar charts!
"""
)

# --- API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- Position Normalization ---
position_aliases = {
    # Goalkeepers
    "GK": "Goalkeeper",

    # Sweepers / Central defenders
    "D (C)": "Centre Back",

    # Fullbacks / Wingbacks
    "D (L)": "Fullback",
    "D (R)": "Fullback",
    "WB (L)": "Wingback",
    "WB (R)": "Wingback",

    # Defensive Midfielders
    "DM": "Defensive Midfielder",

    # Central Midfielders
    "M (C)": "Central Midfielder",
    "MC": "Central Midfielder",

    # Attacking Midfielders
    "AM (C)": "Attacking Midfielder",

    # Wide Midfielders / Wingers
    "M (L)": "Wide Midfielder",
    "M (R)": "Wide Midfielder",
    "AM (L)": "Winger",
    "AM (R)": "Winger",

    # Inside Forward
    "IF (L)": "Inside Forward",
    "IF (R)": "Inside Forward",

    # Forwards
    "ST (C)": "Striker",
    "CF": "Complete Forward",
    "FW": "Forward",
    "WF (L)": "Wide Forward",
    "WF (R)": "Wide Forward",
}

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    positions = [p.strip() for p in pos_str.split(",")]
    for pos in positions:
        # Remove brackets content and uppercase
        pos_clean = re.sub(r"\s*\(.*?\)", "", pos).strip().upper()
        if pos in position_aliases:
            return position_aliases[pos]
        elif pos_clean in position_aliases:
            return position_aliases[pos_clean]
    return "Unknown"

# --- Position-based metrics for radar charts ---
position_metrics = {
    "Goalkeeper": [
        "Pass Completion Ratio", "Save Ratio", "Clean Sheets",
        "Saves Held", "Saves Parried", "Saves Tipped"
    ],
    "Centre Back": [
        "Assists", "Goals", "Headers Won", "Tackle Completion Ratio",
        "Interceptions", "Pass Completion Ratio"
    ],
    "Fullback": [
        "Assists", "Goals", "Dribbles Made", "Tackle Completion Ratio",
        "Interceptions", "Pass Completion Ratio"
    ],
    "Wingback": [
        "Assists", "Goals", "Dribbles Made", "Tackle Completion Ratio",
        "Interceptions", "Pass Completion Ratio"
    ],
    "Defensive Midfielder": [
        "Assists", "Goals", "Tackle Completion Ratio", "Interceptions",
        "Pass Completion Ratio", "Key Passes"
    ],
    "Central Midfielder": [
        "Assists", "Goals", "Key Passes", "Dribbles Made",
        "Pass Completion Ratio", "Interceptions"
    ],
    "Attacking Midfielder": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Expected Assists", "Key Passes"
    ],
    "Wide Midfielder": [
        "Assists", "Goals", "Dribbles Made", "Key Passes",
        "Pass Completion Ratio", "Expected Goals per 90 Minutes"
    ],
    "Winger": [
        "Assists", "Goals", "Dribbles Made", "Key Passes",
        "Pass Completion Ratio", "Expected Goals per 90 Minutes"
    ],
    "Inside Forward": [
        "Assists", "Goals", "Dribbles Made", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Key Passes"
    ],
    "Complete Forward": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Conversion %", "Key Passes"
    ],
    "Striker": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Conversion %", "Key Passes"
    ],
    "Forward": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Conversion %", "Key Passes"
    ],
    "Wide Forward": [
        "Assists", "Goals", "Dribbles Made", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Key Passes"
    ],
    "Unknown": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Expected Assists", "Key Passes"
    ]
}

# --- HTML Parsing to DataFrame ---
def parse_html_to_df(file):
    soup = BeautifulSoup(file, "html.parser")
    table = soup.find("table")
    if not table:
        st.error("No table found in HTML.")
        return pd.DataFrame()
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    # Make headers unique
    seen = {}
    unique_headers = []
    for col in headers:
        if col in seen:
            seen[col] += 1
            unique_headers.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_headers.append(col)

    rows = []
    for row in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True).replace("-", "") for td in row.find_all("td")]
        if len(cols) == len(unique_headers):
            rows.append(cols)
        else:
            # Skip rows that don't match column count
            continue

    df = pd.DataFrame(rows, columns=unique_headers)

    # Try to convert numeric columns to float
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col].str.replace(",",""), errors="coerce")
        except:
            pass
    return df

# --- Normalize metric values for radar chart ---
def normalize_metrics(df, metrics):
    normalized = {}
    for m in metrics:
        if m not in df.columns:
            normalized[m] = 0.0
            continue
        col = df[m]
        if col.isnull().all():
            normalized[m] = 0.0
            continue
        # Normalize by max in the whole dataframe for scaling (avoid dividing by zero)
        max_val = df[m].max()
        if max_val == 0 or pd.isna(max_val):
            normalized[m] = 0.0
        else:
            normalized[m] = min(col.max() / max_val, 1.0)
    return normalized

# --- Radar Chart Plotting ---
def plot_radar_chart(player_name, player_data, metrics):
    categories = metrics
    values = []

    # Build values list, normalized between 0-1 based on dataset max for each metric
    for metric in categories:
        val = player_data.get(metric)
        if val is None or pd.isna(val):
            values.append(0)
        else:
            # Normalize value to 0-1 scale (using max of that metric across all players)
            # To be done outside or with a helper arg? 
            values.append(float(val))
    # We'll scale the values dynamically in Streamlit below

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='black', size=9)
    ax.tick_params(pad=10)

    # Y axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot line and fill
    ax.plot(angles, values, color='tab:blue', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='tab:blue', alpha=0.25)

    plt.title(f"Performance Radar for {player_name}", size=14, y=1.1)

    st.pyplot(fig)

# --- Main App Logic ---

# Upload squad HTML
st.sidebar.header("Upload Files")
squad_file = st.sidebar.file_uploader("Upload FM24 Squad HTML", type=["html"])
transfer_file = st.sidebar.file_uploader("Upload FM24 Transfer Market HTML", type=["html"])

if squad_file is None and transfer_file is None:
    st.info("Please upload your FM24 squad and/or transfer market HTML files from the sidebar to get started.")
    st.stop()

squad_df = None
