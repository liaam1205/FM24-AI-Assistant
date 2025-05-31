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
Upload your FM24 exported **squad** and **transfer market** HTML files to analyze your squad and transfer targets.  
Ask AI questions, get detailed player stats with radar charts, and search transfer market players easily!
"""
)

# --- OpenAI API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- Position Normalization with FM24 roles & positions ---
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
    "AM": "Attacking Midfielder",
    "AM (C)": "Attacking Midfielder",

    # Wide Midfielders / Wingers
    "M (L)": "Wide Midfielder",
    "M (R)": "Wide Midfielder",
    "AM (L)": "Winger",
    "AM (R)": "Winger",

    # Inside Forward
    "AM (L)": "Inside Forward",
    "AM (R)": "Inside Forward",
    "IF": "Inside Forward",

    # Forwards
    "ST (C)": "Striker",
    "FW": "Forward",
    "CF": "Complete Forward",
    "WF": "Wide Forward",
}

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    positions = [p.strip() for p in pos_str.split(",")]
    for pos in positions:
        pos_clean = pos.upper().replace(" ", "")
        # handle special cases
        for alias_key in position_aliases:
            if alias_key.replace(" ", "") == pos_clean:
                return position_aliases[alias_key]
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

# --- Parsing HTML export to DataFrame ---
def parse_html_to_df(html_file):
    try:
        soup = BeautifulSoup(html_file, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("No table found in HTML file.")
            return None

        # Extract headers
        headers = [th.get_text(strip=True) for th in table.find_all("th")]

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) == 0:
                continue
            row = [td.get_text(strip=True) for td in cells]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Convert numeric columns to proper dtype where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace("%","").str.replace(",",""), errors='coerce')

        # Normalize position column if exists
        if "Position" in df.columns:
            df["Normalized Position"] = df["Position"].apply(normalize_position)
        elif "Pos" in df.columns:
            df["Normalized Position"] = df["Pos"].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"Failed to parse HTML file: {e}")
        return None

# --- Radar chart (pizza chart) for player metrics ---
def plot_player_radar(player_data, metrics, title="Player Radar Chart"):
    labels = metrics
    values = []

    for m in metrics:
        val = player_data.get(m)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        values.append(float(val))

    # Close the radar plot
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))

    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_title(title, y=1.1, fontsize=15)

    plt.tight_layout()
    st.pyplot(fig)

# --- Display player details ---
def display_player_details(df, player_name):
    player_row = df[df["Name"] == player_name]
    if player_row.empty:
        st.warning("Player not found in dataset.")
        return

    player_data = player_row.iloc[0].to_dict()
    st.subheader(f"Player Details: {player_name} ({player_data.get('Nationality', '')}, {player_data.get('Position', '')})")

    # Show key player stats
    st.write(player_data)

    # Radar chart metrics by position
    position = player_data.get("Normalized Position", "Unknown")
    metrics = position_metrics.get(position, position_metrics["Unknown"])

    # Check if player_data has enough valid metrics
    valid_metrics = [m for m in metrics if m in player_data and not (player_data[m] is None or (isinstance(player_data[m], float) and np.isnan(player_data[m])))]
    if len(valid_metrics) < 3:
        st.warning("Not enough data for radar chart.")
        return

    plot_player_radar(player_data, metrics, title=f"{player_name} - {position}")

# --- Upload Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Upload Squad Export (.html)")
    squad_file = st.file_uploader("Upload your FM24 squad HTML export", type=["html"], key="squad")

with col2:
    st.subheader("📁 Upload Transfer Market Export (.html)")
    transfer_file = st.file_uploader("Upload your FM24 transfer market HTML export", type=["html"], key="transfer")

squad_df = None
transfer_df = None

if squad_file is not None:
    squad_df = parse_html_to_df(squad_file)
    if squad_df is not None:
        st.success(f"✅ Squad file uploaded and parsed! {len(squad_df)} players loaded.")
        st.subheader("Your Squad Data")
        st.dataframe(squad_df, use_container_width=True)

if transfer_file is not None:
    transfer_df = parse_html_to_df(transfer_file)
    if transfer_df is not None:
        st.success(f"✅ Transfer market file uploaded and parsed! {len(transfer_df)} players loaded.")
        st.subheader("Transfer Market Data")
        st.dataframe(transfer
