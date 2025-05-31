import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np

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
    "AM (L)": "Inside Forward",     # Your requested change here
    "AM (R)": "Inside Forward",     # Your requested change here
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

# --- Parse HTML file to DataFrame with tbody fallback ---
def parse_html_to_df(html_str):
    try:
        soup = BeautifulSoup(html_str, "html.parser")
        table = soup.find("table")
        if not table:
            st.error("No table found in HTML.")
            return None

        # Extract headers
        headers = []
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        else:
            # No thead - get headers from first row
            first_row = table.find("tr")
            if first_row:
                headers = [th.get_text(strip=True) for th in first_row.find_all(["th", "td"])]

        # Extract rows from tbody or fallback to table rows
        tbody = table.find("tbody")
        if tbody:
            rows_source = tbody.find_all("tr")
        else:
            all_rows = table.find_all("tr")
            if thead:
                rows_source = all_rows  # all rows since thead is separate
            else:
                rows_source = all_rows[1:]  # skip first row as header

        rows = []
        for tr in rows_source:
            row = [td.get_text(separator=" ", strip=True) for td in tr.find_all("td")]
            if len(row) == len(headers):
                rows.append(row)

        if not rows:
            st.error("No rows found in table.")
            return None

        df = pd.DataFrame(rows, columns=headers)

        # Clean numeric columns (% and commas), convert to numeric where possible
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace("%", "").str.replace(",", "")
                df[col] = pd.to_numeric(df[col], errors="ignore")

        # Normalize positions if possible
        pos_cols = [c for c in df.columns if "pos" in c.lower()]
        if pos_cols:
            df["Normalized Position"] = df[pos_cols[0]].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return None

# --- Plot radar (pizza) chart for player ---
def plot_player_radar(player_data, metrics, title="Player Radar Chart"):
    labels = metrics
    values = []
    for m in metrics:
        val = player_data.get(m)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        values.append(float(val))

    values += values[:1]  # close the loop

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_title(title, y=1.1, fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

# --- Display player details and radar ---
def display_player_details(df, player_name):
    player_row = df[df["Name"] == player_name]
    if player_row.empty:
        st.warning("Player not found in dataset.")
        return
    player_data = player_row.iloc[0].to_dict()

    st.subheader(f"Player Details: {player_name} ({player_data.get('Nationality', '')}, {player_data.get('Position', '')})")

    # Show player stats as JSON
    details_to_show = {k: v for k, v in player_data.items() if k != 'Normalized Position'}
    st.json(details_to_show)

    # Radar chart by normalized position
    position = player_data.get("Normalized Position", "Unknown")
    metrics = position_metrics.get(position, position_metrics["Unknown"])

    # Filter metrics with data
    valid_metrics = [m for m in metrics if m in player_data and not (player_data[m] is None or (isinstance(player_data[m], float) and np.isnan(player_data[m])))]
    if len(valid_metrics) < 3:
        st.warning("Not enough data for radar chart.")
        return

    plot_player_radar(player_data, valid_metrics, title=f"{player_name} - {position}")

# --- Main UI ---

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
    squad_html = squad_file.read().decode("utf-8")
    squad_df = parse_html_to_df(squad_html)
    if squad_df is not None:
        st.subheader("📊 Squad Data")
        st.dataframe(squad_df)

if transfer_file is not None:
    transfer_html = transfer_file.read().decode("utf-8")
    transfer_df = parse_html_to_df(transfer_html)
    if transfer_df is not None:
        st.subheader("📊 Transfer Market Data")
        st.dataframe(transfer_df)

# Player details section
if squad_df is not None:
    st.subheader("🔍 Search Squad Player")
    player_names = squad_df["Name"].dropna().unique()
    selected_player = st.selectbox("Select player from your squad", player_names)
    if selected_player:
        display_player_details(squad_df, selected_player)

# Transfer market player search
if transfer_df is not None:
    st.subheader("🔍 Search Transfer Market Player")
    transfer_names = transfer_df["Name"].dropna().unique()
    selected_transfer_player = st.selectbox("Select player from transfer market", transfer_names)
    if selected_transfer_player:
        display_player_details(transfer_df, selected_transfer_player)

# You can add AI question feature or other components below
