import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np
import re

# --- App Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("\u26bd Football Manager 2024 Squad & Transfer Analyzer")
st.markdown("""
Upload your FM24 exported squad and transfer market HTML files to analyze your squad and the available players.
Ask AI questions about your squad or transfer targets, and view detailed player stats with radar charts!
""")

# --- API Key ---
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
    "M (L)": "Wide Midfielder",
    "M (R)": "Wide Midfielder",
    "AM (C)": "Attacking Midfielder",
    "AM (L)": "Winger",
    "AM (R)": "Winger",
    "ST (C)": "Striker"
}

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    for pos in pos_str.split(","):
        pos_cleaned = pos.strip().upper()
        if pos_cleaned in position_aliases:
            return position_aliases[pos_cleaned]
    return "Unknown"

# --- Position-based radar metrics ---
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
    "Striker": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Conversion %", "Key Passes"],
    "Unknown": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Expected Assists", "Key Passes"]
}

# --- HTML Parsing Function ---
def parse_html_to_df(file):
    soup = BeautifulSoup(file, "html.parser")
    table = soup.find("table")
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for row in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True).replace("-", "") for td in row.find_all("td")]
        if len(cols) == len(headers):
            rows.append(cols)
    df = pd.DataFrame(rows, columns=headers)
    return df

# --- Radar Chart ---
def plot_radar(player_data, position):
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    values = []
    for metric in metrics:
        val = player_data.get(metric)
        try:
            val = float(val)
            values.append(val)
        except:
            st.warning(f"Missing or invalid value for metric: {metric}")
            return

    if len(values) < 3:
        st.warning("Not enough metrics to plot radar chart.")
        return

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(values))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    st.pyplot(fig)

# --- File Uploads ---
squad_file = st.file_uploader("Upload Squad HTML", type="html", key="squad")
transfer_file = st.file_uploader("Upload Transfer Market HTML", type="html", key="transfer")

if squad_file:
    st.subheader("\ud83d\udcc4 Squad Data")
    squad_df = parse_html_to_df(squad_file)
    st.dataframe(squad_df)

if transfer_file:
    st.subheader("\ud83d\udcbc Transfer Market Data")
    transfer_df = parse_html_to_df(transfer_file)
    st.dataframe(transfer_df)

# --- Player Selector ---
combined_df = pd.concat([squad_df, transfer_df], ignore_index=True) if squad_file and transfer_file else squad_df if squad_file else transfer_df if transfer_file else None

if combined_df is not None:
    combined_df['Position'] = combined_df['Position'].apply(normalize_position)
    player_names = combined_df['Name'].dropna().unique()
    selected_player = st.selectbox("Select a player to view detailed stats:", player_names)
    player_row = combined_df[combined_df['Name'] == selected_player].iloc[0]

    st.markdown(f"### \ud83d\udd0d Player Details: {selected_player}")
    st.json(player_row.to_dict())

    plot_radar(player_row, player_row['Position'])

# --- AI Squad Question ---
if combined_df is not None:
    st.subheader("\ud83e\udde0 Ask AI About Your Squad or Transfers")
    user_query = st.text_area("Ask a question (e.g. 'Who should I sell?', 'Best midfielders?', 'Top transfer targets?')")
    if st.button("Analyze with AI") and user_query:
        with st.spinner("Analyzing..."):
            try:
                prompt = f"""
You are an assistant analyzing a Football Manager 2024 squad and transfer market.

Player data:
{combined_df.to_markdown(index=False)}

User question: {user_query}
                """
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a tactical football analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800
                )
                st.markdown("### \ud83d\udcda AI Insights")
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"AI API error: {e}")
