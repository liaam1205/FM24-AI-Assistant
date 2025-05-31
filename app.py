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
Get detailed player stats, position-aware radar charts, and AI scouting reports powered by OpenAI!  
"""
)

# --- OpenAI API Key ---
try:
    api_key = st.secrets["API_KEY"]
    client = openai.OpenAI(api_key=api_key)
except Exception:
    api_key = None
    client = None

# --- Position Normalization with FM24 roles & positions ---
position_aliases = {
    "GK": "Goalkeeper",
    "D (C)": "Centre Back",
    "D (L)": "Fullback",
    "D (R)": "Fullback",
    "WB (L)": "Wingback",
    "WB (R)": "Wingback",
    "DM": "Defensive Midfielder",
    "M (C)": "Central Midfielder",
    "MC": "Central Midfielder",
    "AM": "Attacking Midfielder",
    "AM (C)": "Attacking Midfielder",
    "M (L)": "Wide Midfielder",
    "M (R)": "Wide Midfielder",
    "AM (L)": "Inside Forward",   # Added as Inside Forward
    "AM (R)": "Inside Forward",   # Added as Inside Forward
    "IF": "Inside Forward",
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
        for alias_key in position_aliases:
            if alias_key.replace(" ", "") == pos_clean:
                return position_aliases[alias_key]
    return "Unknown"

# --- Position-based metrics for radar charts ---
position_metrics = {
    "Goalkeeper": [
        "Pas %", "Sv %", "Clean Sheets", "Svh", "Svp", "Sv"
    ],
    "Centre Back": [
        "Ast", "Gls", "Hdrs", "Tck R", "Itc", "Pas %"
    ],
    "Fullback": [
        "Ast", "Gls", "Drb", "Tck R", "Itc", "Pas %"
    ],
    "Wingback": [
        "Ast", "Gls", "Drb", "Tck R", "Itc", "Pas %"
    ],
    "Defensive Midfielder": [
        "Ast", "Gls", "Tck R", "Itc", "Pas %", "K Pas"
    ],
    "Central Midfielder": [
        "Ast", "Gls", "K Pas", "Drb", "Pas %", "Itc"
    ],
    "Attacking Midfielder": [
        "Ast", "Gls", "xG/90", "xG-OP", "xA", "K Pas"
    ],
    "Wide Midfielder": [
        "Ast", "Gls", "Drb", "K Pas", "Pas %", "xG/90"
    ],
    "Winger": [
        "Ast", "Gls", "Drb", "K Pas", "Pas %", "xG/90"
    ],
    "Inside Forward": [
        "Ast", "Gls", "Drb", "xG/90", "xG-OP", "K Pas"
    ],
    "Complete Forward": [
        "Ast", "Gls", "xG/90", "xG-OP", "K Pas"
    ],
    "Striker": [
        "Ast", "Gls", "xG/90", "xG-OP", "K Pas"
    ],
    "Forward": [
        "Ast", "Gls", "xG/90", "xG-OP", "K Pas"
    ],
    "Wide Forward": [
        "Ast", "Gls", "Drb", "xG/90", "xG-OP", "K Pas"
    ],
    "Unknown": [
        "Ast", "Gls", "xG/90", "xG-OP", "xA", "K Pas"
    ]
}

# --- Parse HTML to DataFrame ---
def parse_html(file) -> pd.DataFrame | None:
    try:
        soup = BeautifulSoup(file, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

        # Find header row (some tables have thead, some don't)
        header_row = table.find("thead")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
        else:
            first_tr = table.find("tr")
            headers = [th.get_text(strip=True) for th in first_tr.find_all("th")]
            if not headers:
                # Sometimes headers are in first td row
                headers = [td.get_text(strip=True) for td in first_tr.find_all("td")]

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            row = [td.get_text(strip=True) for td in cells]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Convert numeric columns, cleaning commas and %
        for col in df.columns:
            df[col] = df[col].str.replace(",", "").str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Normalize position columns (Position or Pos)
        if "Position" in df.columns:
            df["Normalized Position"] = df["Position"].apply(normalize_position)
        elif "Pos" in df.columns:
            df["Normalized Position"] = df["Pos"].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df
    except Exception as e:
        st.error(f"Error parsing HTML file: {e}")
        return None

# --- Radar chart ---
def plot_player_radar(player_data: dict, metrics: list[str], title="Player Radar Chart"):
    labels = metrics
    values = []
    for m in metrics:
        val = player_data.get(m)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        values.append(float(val))

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#1f77b4', alpha=0.25)
    ax.plot(angles, values, color='#1f77b4', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color='#333333')

    ax.spines['polar'].set_visible(False)
    ax.grid(color='#cccccc', linestyle='--', linewidth=0.5)
    ax.set_title(title, y=1.1, fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# --- Display player details ---
def display_player_details(df: pd.DataFrame, player_name: str):
    player_rows = df[df["Name"] == player_name]
    if player_rows.empty:
        st.warning("Player not found in dataset.")
        return
    player_data = player_rows.iloc[0].to_dict()

    st.subheader(f"{player_name} - {player_data.get('Normalized Position', 'Unknown')}")

    # Prepare a clean table of key stats for display (exclude non-stat columns)
    exclude_cols = ["Normalized Position", "Rec", "Potential", "Ability", "Name", "Position", "Pos"]
    stat_cols = [col for col in df.columns if col not in exclude_cols]

    stats = {col: player_data.get(col, "") for col in stat_cols}

    stats_df = pd.DataFrame(stats.items(), columns=["Stat", "Value"])
    stats_df["Value"] = stats_df["Value"].astype(str)

    st.table(stats_df)

    # Plot radar chart for player's normalized position
    position = player_data.get("Normalized Position", "Unknown")
    metrics = position_metrics.get(position, position_metrics["Unknown"])

    # Check for at least 3 metrics with valid numeric data
    valid_metrics = [m for m in metrics if m in player_data and pd.notna(player_data[m])]
    if len(valid_metrics) < 3:
        st.info("Not enough data to display radar chart.")
        return

    plot_player_radar(player_data, valid_metrics, title=f"{player_name} - {position}")

# --- AI scouting report ---
def generate_ai_scouting_report(player_data: dict, api_key: str) -> str:
    if not api_key:
        return "OpenAI API key not configured. Cannot generate AI scouting report."

    prompt = (
        f"Provide a detailed football scouting report for the following player based on these stats:\n\n"
        f"Name: {player_data.get('Name', 'Unknown')}\n"
        f"Position: {player_data.get('Normalized Position', 'Unknown')}\n"
        f"Stats:\n"
    )
    # Include numeric stats for report
    for stat, val in player_data.items():
        if stat in ["Name", "Normalized Position"] or val == "" or pd.isna(val):
            continue
        prompt += f
