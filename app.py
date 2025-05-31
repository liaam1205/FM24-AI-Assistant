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

# --- API Key ---
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

    # Forwards
    "ST (C)": "Striker",
    "FW": "Forward",
    "WF": "Wide Forward"
}

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    positions = [p.strip() for p in pos_str.split(",")]
    for pos in positions:
        # remove parentheses and uppercase
        main_pos = re.sub(r"\s*\(.*?\)", "", pos).strip().upper()
        if main_pos in position_aliases:
            return position_aliases[main_pos]
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

# --- Parsing HTML to DataFrame ---
def parse_html_to_df(file):
    soup = BeautifulSoup(file, "html.parser")
    table = soup.find("table")
    if table is None:
        st.error("No table found in uploaded file.")
        return None
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    # Make column headers unique
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

    df = pd.DataFrame(rows, columns=unique_headers)
    # Convert numeric columns where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

# --- Normalize metrics for radar ---
def normalize_metrics(df, metrics):
    norm = {}
    for m in metrics:
        if m in df.columns:
            max_val = df[m].max()
            # Avoid division by zero or NaN
            if pd.isna(max_val) or max_val == 0:
                norm[m] = None
            else:
                norm[m] = max_val
        else:
            norm[m] = None
    return norm

# --- Draw radar (pizza) chart ---
def plot_radar_chart(player_stats, metrics, metric_maxes, player_name):
    labels = metrics
    stats = []
    for m in metrics:
        val = player_stats.get(m, 0)
        if pd.isna(val) or val is None:
            val = 0
        max_val = metric_maxes.get(m, 1)
        if max_val is None or max_val == 0:
            scaled_val = 0
        else:
            scaled_val = val / max_val
        stats.append(scaled_val)

    # Radar requires stats to be a closed loop
    stats += stats[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels)

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.yaxis.labelpad = 20
    ax.grid(True)
    ax.plot(angles, stats, color="red", linewidth=2)
    ax.fill(angles, stats, color="red", alpha=0.25)
    ax.set_ylim(0, 1)

    plt.title(f"{player_name} Performance Radar", size=15, y=1.1)
    st.pyplot(fig)

# --- Display player details + radar chart ---
def display_player_details(df, player_name):
    player_row = df[df["Name"] == player_name]
    if player_row.empty:
        st.warning(f"No data found for player: {player_name}")
        return
    player = player_row.iloc[0]
    pos_raw = player.get("Position", "Unknown")
    position = normalize_position(pos_raw)

    st.markdown(f"### Player: {player_name} | Position: {position}")

    # Show all player stats in table format
    stats_display = player.drop(labels=["Name"]).to_dict()
    stats_display = {k: v if pd.notna(v) else "N/A" for k, v in stats_display.items()}
    stats_df = pd.DataFrame.from_dict(stats_display, orient='index', columns=["Value"])
    st.table(stats_df)

    # Radar chart metrics & normalization
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    metric_maxes = normalize_metrics(df, metrics)
    # Show radar chart only if at least 3 metrics have data
    available_metrics = [m for m in metrics if metric_maxes.get(m) not in [None, 0]]
    if len(available_metrics) < 3:
        st.info("Not enough metric data to show radar chart.")
        return
    plot_radar_chart(player, available_metrics, metric_maxes, player_name)

# --- Main App ---

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
        st.dataframe(transfer_df, use_container_width=True)

# --- AI Query about Squad ---
if squad_df is not None:
    st.subheader("🤖 Ask AI about your Squad")
    user_query = st.text_area("Enter a question (e.g., 'Who should I sell?', 'Top 3 midfielders?')")
    if st.button("Analyze Squad with AI"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Asking AI..."):
                try:
                    # Prepare
