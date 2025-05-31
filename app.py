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
st.markdown("""
Upload your FM24 exported **squad** and **transfer market** HTML files to analyze your squad and transfer targets.  
Ask AI questions, get detailed player stats with radar charts, and search transfer market players easily!
""")

# --- OpenAI API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

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
    "AM (L)": "Inside Forward",
    "AM (R)": "Inside Forward",
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

# --- Metric mapping ---
header_mapping = {
    "Inf": "Info",
    "Name": "Name",
    "Club": "Club",
    "Position": "Position",
    "Age": "Age",
    "CA": "Current Ability",
    "PA": "Potential Ability",
    "Transfer Value": "Transfer Value",
    "Wage": "Wage",
    "Ast": "Assists",
    "Gls": "Goals",
    "xG/90": "Expected Goals per 90 Minutes",
    "xG-OP": "Expected Goals Overperformance",
    "xA": "Expected Assists",
    "K Pas": "Key Passes",
    "Drb": "Dribbles Made",
    "Pas %": "Pass Completion Ratio",
    "Itc": "Interceptions",
    "Hdrs": "Headers Won",
    "Tck R": "Tackle Completion Ratio",
    "Sv %": "Save Ratio",
    "Clean Sheets": "Clean Sheets",
    "Svh": "Saves Held",
    "Svp": "Saves Parried",
    "Sv": "Saves Tipped"
}

position_metrics = {
    "Goalkeeper": ["Save Ratio", "Clean Sheets", "Saves Held", "Saves Parried", "Saves Tipped", "Pass Completion Ratio"],
    "Centre Back": ["Headers Won", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio", "Goals", "Assists"],
    "Fullback": ["Dribbles Made", "Key Passes", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio", "Assists"],
    "Wingback": ["Dribbles Made", "Key Passes", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio", "Assists"],
    "Defensive Midfielder": ["Interceptions", "Tackle Completion Ratio", "Pass Completion Ratio", "Key Passes", "Assists", "Goals"],
    "Central Midfielder": ["Key Passes", "Dribbles Made", "Pass Completion Ratio", "Interceptions", "Assists", "Goals"],
    "Attacking Midfielder": ["Expected Assists", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes", "Assists", "Goals"],
    "Wide Midfielder": ["Dribbles Made", "Expected Goals per 90 Minutes", "Key Passes", "Pass Completion Ratio", "Assists", "Goals"],
    "Winger": ["Dribbles Made", "Expected Goals per 90 Minutes", "Key Passes", "Pass Completion Ratio", "Assists", "Goals"],
    "Inside Forward": ["Dribbles Made", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes", "Assists", "Goals"],
    "Striker": ["Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes", "Assists", "Goals"],
    "Complete Forward": ["Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes", "Assists", "Goals"],
    "Forward": ["Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes", "Assists", "Goals"],
    "Wide Forward": ["Dribbles Made", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes", "Assists", "Goals"],
    "Unknown": ["Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Expected Assists", "Key Passes", "Assists", "Goals"]
}

# --- HTML Parser ---
def parse_html(html_file):
    soup = BeautifulSoup(html_file, "html.parser")
    table = soup.find("table")
    if not table:
        return None
    
    header_row = table.find("thead").find_all("th")
    headers = [header_mapping.get(th.text.strip(), th.text.strip()) for th in header_row]
    
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        row = [td.get_text(strip=True) for td in cells]
        if row:
            rows.append(row)

    df = pd.DataFrame(rows, columns=headers)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace("%", "").str.replace(",", ""), errors='ignore')

    if "Position" in df.columns:
        df["Normalized Position"] = df["Position"].apply(normalize_position)
    return df

# --- Radar Chart ---
def plot_radar(player_data, metrics, title):
    labels = metrics
    values = [float(player_data.get(m, 0) or 0) for m in metrics]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='orange', alpha=0.3)
    ax.plot(angles, values, color='orange', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, y=1.1, fontsize=10)
    st.pyplot(fig)

# --- Display Details ---
def show_player_details(df, player_name):
    row = df[df["Name"] == player_name]
    if row.empty:
        st.warning("Player not found.")
        return

    data = row.iloc[0].to_dict()
    st.subheader(f"{player_name} ({data.get('Club', '')}, {data.get('Position', '')})")
    st.json(data)

    position = data.get("Normalized Position", "Unknown")
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    valid_metrics = [m for m in metrics if m in data and data[m] not in [None, '', 'NaN']]

    if len(valid_metrics) < 3:
        st.warning("Not enough data for radar chart.")
        return

    plot_radar(data, valid_metrics, f"{player_name} - {position}")

# --- File Upload UI ---
squad_file = st.file_uploader("Upload Squad Export (.html)", type="html")
transfer_file = st.file_uploader("Upload Transfer Export (.html)", type="html")

if squad_file:
    st.subheader("📊 Squad Analysis")
    squad_df = parse_html(squad_file)
    if squad_df is not None:
        st.dataframe(squad_df)
        player_name = st.selectbox("Select Player", squad_df["Name"].unique())
        show_player_details(squad_df, player_name)

if transfer_file:
    st.subheader("💼 Transfer Market Analysis")
    transfer_df = parse_html(transfer_file)
    if transfer_df is not None:
        st.dataframe(transfer_df)
        player_name = st.selectbox("Select Transfer Player", transfer_df["Name"].unique(), key="transfer")
        show_player_details(transfer_df, player_name)
