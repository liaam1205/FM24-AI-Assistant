import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np

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

# --- Position Normalization ---
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

# --- Metrics by Position ---
position_metrics = {
    "Goalkeeper": ["Pass Completion Ratio", "Save Ratio", "Clean Sheets", "Saves Held", "Saves Parried", "Saves Tipped"],
    "Centre Back": ["Assists", "Goals", "Headers Won", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio"],
    "Fullback": ["Assists", "Goals", "Dribbles Made", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio"],
    "Wingback": ["Assists", "Goals", "Dribbles Made", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio"],
    "Defensive Midfielder": ["Assists", "Goals", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio", "Key Passes"],
    "Central Midfielder": ["Assists", "Goals", "Key Passes", "Dribbles Made", "Pass Completion Ratio", "Interceptions"],
    "Attacking Midfielder": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Expected Assists", "Key Passes"],
    "Wide Midfielder": ["Assists", "Goals", "Dribbles Made", "Key Passes", "Pass Completion Ratio", "Expected Goals per 90 Minutes"],
    "Winger": ["Assists", "Goals", "Dribbles Made", "Key Passes", "Pass Completion Ratio", "Expected Goals per 90 Minutes"],
    "Inside Forward": ["Assists", "Goals", "Dribbles Made", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes"],
    "Complete Forward": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Conversion %", "Key Passes"],
    "Striker": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Conversion %", "Key Passes"],
    "Forward": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Conversion %", "Key Passes"],
    "Wide Forward": ["Assists", "Goals", "Dribbles Made", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes"],
    "Unknown": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Expected Assists", "Key Passes"]
}

# --- HTML Parser ---
def parse_html_to_df(html_file):
    try:
        soup = BeautifulSoup(html_file, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            row = [cell.get_text(strip=True) for cell in cells]
            rows.append(row)

        if len(rows) < 2:
            st.error("Not enough data rows found in the HTML file.")
            return None

        raw_headers = rows[0]
        data_rows = rows[1:]

        header_mapping = {
            "Inf": "Player Information",
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
            "Sv": "Saves Tipped",
        }

        mapped_headers = [header_mapping.get(h, h) for h in raw_headers]
        df = pd.DataFrame(data_rows, columns=mapped_headers)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(",", "").str.replace("%", ""), errors="ignore")

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

# --- Radar Chart ---
def plot_player_radar(player_data, metrics, title="Player Radar Chart"):
    labels = metrics
    values = [float(player_data.get(m, 0) or 0) for m in labels]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, y=1.1, fontsize=12)
    st.pyplot(fig)

# --- Player Details ---
def display_player_details(df, player_name):
    player_row = df[df["Name"] == player_name]
    if player_row.empty:
        st.warning("Player not found in dataset.")
        return

    player_data = player_row.iloc[0].to_dict()
    st.subheader(f"Player: {player_name} ({player_data.get('Position', '')})")
    st.json({k: v for k, v in player_data.items() if k not in ["Normalized Position"]})

    position = player_data.get("Normalized Position", "Unknown")
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    valid_metrics = [m for m in metrics if m in player_data and pd.notnull(player_data[m])]

    if len(valid_metrics) < 3:
        st.warning("Not enough data for radar chart.")
        return

    plot_player_radar(player_data, valid_metrics, title=f"{player_name} - {position}")

# --- UI Layout ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Upload Squad Export (.html)")
    squad_file = st.file_uploader("Upload FM24 squad HTML", type="html", key="squad")

with col2:
    st.subheader("📁 Upload Transfer Market Export (.html)")
    transfer_file = st.file_uploader("Upload FM24 transfer market HTML", type="html", key="transfer")

squad_df, transfer_df = None, None

if squad_file:
    squad_df = parse_html_to_df(squad_file)
    if squad_df is not None:
        st.subheader("📊 Squad Data Preview")
        st.dataframe(squad_df)

if transfer_file:
    transfer_df = parse_html_to_df(transfer_file)
    if transfer_df is not None:
        st.subheader("🔍 Transfer Market Preview")
        st.dataframe(transfer_df)

if squad_df is not None:
    player_name = st.selectbox("Select a Player to View Details", options=squad_df["Name"].dropna().unique())
    if player_name:
        display_player_details(squad_df, player_name)
