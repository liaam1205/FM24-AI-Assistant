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

# --- OpenAI API Key (replace or set in secrets) ---
api_key = st.secrets.get("API_KEY", "")
client = openai.OpenAI(api_key=api_key) if api_key else None

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
    "AM (L)": "Inside Forward",    # First alias for Inside Forward
    "AM (R)": "Inside Forward",    # Second alias for Inside Forward
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

# --- Position-based metrics ---
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

# --- Parse HTML ---
def parse_html_to_df(html_str):
    try:
        soup = BeautifulSoup(html_str, "html.parser")
        table = soup.find("table")
        if not table:
            st.error("No table found in HTML.")
            return None

        # Extract headers from thead
        headers = []
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]

        # Extract rows from tbody
        tbody = table.find("tbody")
        if not tbody:
            st.error("No tbody found in table.")
            return None

        rows = []
        for tr in tbody.find_all("tr"):
            row = []
            for td in tr.find_all("td"):
                text = td.get_text(separator=" ", strip=True)
                row.append(text)
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Clean numeric columns (remove % and commas), convert where possible
        for col in df.columns:
            df[col] = df[col].str.replace("%", "").str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # Normalize positions
        pos_cols = [c for c in df.columns if "pos" in c.lower()]
        if pos_cols:
            df["Normalized Position"] = df[pos_cols[0]].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df
    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return None

# --- Radar Chart ---
def plot_player_radar(player_data, metrics, title="Player Radar Chart"):
    labels = metrics
    values = []
    for m in metrics:
        val = player_data.get(m, 0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        values.append(float(val))

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='orange', alpha=0.25)
    ax.plot(angles, values, color='orange', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title, y=1.1, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)

# --- Display Player Details and Radar ---
def display_player_details(df, player_name):
    player_row = df[df["Name"] == player_name]
    if player_row.empty:
        st.warning("Player not found.")
        return
    player_data = player_row.iloc[0].to_dict()

    st.subheader(f"Player Details: {player_name} ({player_data.get('Nationality','')}, {player_data.get('Position','')})")

    # Show key player stats in JSON format (excluding normalized position)
    details_to_show = {k:v for k,v in player_data.items() if k != "Normalized Position"}
    st.json(details_to_show)

    # Radar chart
    position = player_data.get("Normalized Position", "Unknown")
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    valid_metrics = [m for m in metrics if m in player_data and player_data[m] not in [None, ""] and not (isinstance(player_data[m], float) and np.isnan(player_data[m]))]

    if len(valid_metrics) < 3:
        st.warning("Not enough data for radar chart.")
        return
    plot_player_radar(player_data, valid_metrics, title=f"{player_name} - {position}")

# --- Main UI ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Upload Squad Export (.html)")
    squad_file = st.file_uploader("Upload your FM24 squad HTML export", type=["html"])

with col2:
    st.subheader("📁 Upload Transfer Market Export (.html)")
    transfer_file = st.file_uploader("Upload your FM24 transfer market HTML export", type=["html"])

squad_df = None
transfer_df = None

if squad_file:
    html_bytes = squad_file.read()
    squad_df = parse_html_to_df(html_bytes.decode("utf-8"))
    if squad_df is not None:
        st.markdown("### Squad Data Preview")
        st.dataframe(squad_df)

if transfer_file:
    html_bytes = transfer_file.read()
    transfer_df = parse_html_to_df(html_bytes.decode("utf-8"))
    if transfer_df is not None:
        st.markdown("### Transfer Market Data Preview")
        st.dataframe(transfer_df)

# Select player from squad to display details & radar
if squad_df is not None:
    st.markdown("---")
    st.subheader("🔎 Analyze Squad Player")
    player_list = squad_df["Name"].tolist()
    selected_player = st.selectbox("Select a player to view details", player_list)
    if selected_player:
        display_player_details(squad_df, selected_player)

# Select player from transfer market to display details & radar
if transfer_df is not None:
    st.markdown("---")
    st.subheader("🔎 Analyze Transfer Market Player")
    transfer_player_list = transfer_df["Name"].tolist()
    selected_transfer_player = st.selectbox("Select a transfer target player", transfer_player_list, key="transfer_select")
    if selected_transfer_player:
        display_player_details(transfer_df, selected_transfer_player)

# --- Optional: AI Q&A Section (stub for your OpenAI implementation) ---
st.markdown("---")
st.subheader("🤖 Ask AI about your squad or transfers")
question = st.text_input("Enter your question here:")

if question and client:
    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an FM24 assistant helping analyze football data."},
                    {"role": "user", "content": question}
                ]
            )
            answer = response.choices[0].message.content
            st.markdown(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"Error communicating with OpenAI API: {e}")
elif question and not client:
    st.warning("OpenAI API key not set. Please configure your API key.")
