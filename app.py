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
openai.api_key = api_key

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

# --- Position-based metrics for radar charts ---
position_metrics = {
    "Goalkeeper": ["Pas %", "Sv %", "Clean Sheets", "Svh", "Svp", "Sv"],
    "Centre Back": ["Ast", "Gls", "Hdrs", "Tck R", "Itc", "Pas %"],
    "Fullback": ["Ast", "Gls", "Drb", "Tck R", "Itc", "Pas %"],
    "Wingback": ["Ast", "Gls", "Drb", "Tck R", "Itc", "Pas %"],
    "Defensive Midfielder": ["Ast", "Gls", "Tck R", "Itc", "Pas %", "K Pas"],
    "Central Midfielder": ["Ast", "Gls", "K Pas", "Drb", "Pas %", "Itc"],
    "Attacking Midfielder": ["Ast", "Gls", "xG/90", "xG-OP", "xA", "K Pas"],
    "Wide Midfielder": ["Ast", "Gls", "Drb", "K Pas", "Pas %", "xG/90"],
    "Winger": ["Ast", "Gls", "Drb", "K Pas", "Pas %", "xG/90"],
    "Inside Forward": ["Ast", "Gls", "Drb", "xG/90", "xG-OP", "K Pas"],
    "Complete Forward": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas"],
    "Striker": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas"],
    "Forward": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas"],
    "Wide Forward": ["Ast", "Gls", "Drb", "xG/90", "xG-OP", "K Pas"],
    "Unknown": ["Ast", "Gls", "xG/90", "xG-OP", "xA", "K Pas"]
}

# --- Parse HTML file to DataFrame ---
def parse_html_to_df(html_file):
    try:
        soup = BeautifulSoup(html_file, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

        tbody = table.find("tbody") or table
        headers = [th.get_text(strip=True) for th in table.find_all("th")]

        rows = []
        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            row = [td.get_text(strip=True) for td in cells]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Convert numeric columns where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace("%", "").str.replace(",", ""), errors='coerce')

        # Normalize positions
        pos_col = next((col for col in df.columns if col.lower() in ["position", "pos"]), None)
        if pos_col:
            df["Normalized Position"] = df[pos_col].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df
    except Exception as e:
        st.error(f"Error parsing HTML file: {e}")
        return None

# --- Radar chart ---
def plot_player_radar(player_data, metrics, title="Radar Chart"):
    labels = metrics
    values = []
    for m in metrics:
        val = player_data.get(m)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        values.append(float(val))

    if sum(values) == 0:
        st.warning("Not enough data to generate radar chart.")
        return

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
    plt.tight_layout()
    st.pyplot(fig)

# --- Display player ---
def display_player_details(df, player_name):
    player_row = df[df["Name"] == player_name]
    if player_row.empty:
        st.warning("Player not found.")
        return

    player_data = player_row.iloc[0].to_dict()
    st.subheader(f"Player: {player_name}")
    st.json({k: v for k, v in player_data.items() if k != "Normalized Position"})

    metrics = position_metrics.get(player_data.get("Normalized Position", "Unknown"), [])
    if len(metrics) >= 3:
        plot_player_radar(player_data, metrics, f"{player_name} - {player_data.get('Normalized Position')}")
    else:
        st.warning("Not enough metrics to display radar chart.")

    # AI Report
    if st.button(f"Generate AI Scouting Report for {player_name}"):
        prompt = f"Write a short scouting report for {player_name} based on: {player_data}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            report = response.choices[0].message.content.strip()
            st.success("Scouting Report:")
            st.markdown(report)
        except Exception as e:
            st.error(f"Error generating report: {e}")

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Upload Squad HTML")
    squad_file = st.file_uploader("Upload squad file", type="html", key="squad")
with col2:
    st.subheader("📁 Upload Transfer Market HTML")
    transfer_file = st.file_uploader("Upload transfer file", type="html", key="transfer")

squad_df, transfer_df = None, None

if squad_file:
    squad_df = parse_html_to_df(squad_file.getvalue().decode("utf-8"))
    if squad_df is not None:
        st.subheader("📊 Squad Players")
        st.dataframe(squad_df)
        player_name = st.selectbox("Select a player to analyze", squad_df["Name"])
        if player_name:
            display_player_details(squad_df, player_name)

if transfer_file:
    transfer_df = parse_html_to_df(transfer_file.getvalue().decode("utf-8"))
    if transfer_df is not None:
        st.subheader("💸 Transfer Market Players")
        st.dataframe(transfer_df)
        player_name = st.selectbox("Select a transfer target", transfer_df["Name"])
        if player_name:
            display_player_details(transfer_df, player_name)
