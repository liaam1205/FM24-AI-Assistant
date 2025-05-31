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
Upload your FM24 exported squad and transfer market HTML files to analyze your squad and the available players.
Ask AI questions about your squad or transfer targets, and view detailed player stats with radar charts!
"""
)

# --- API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- Position Normalization with FM24 roles & positions ---
position_aliases = {
    "GK": "Goalkeeper",
    "SW": "Sweeper",
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
    "IF (L)": "Inside Forward",
    "IF (R)": "Inside Forward",
    "ST (C)": "Striker",
    "FW": "Forward",
}

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    positions = [p.strip() for p in pos_str.split(",")]
    for pos in positions:
        # Remove role modifiers like (C), (L), (R)
        base_pos = re.sub(r"\s*\(.*?\)", "", pos).upper()
        if base_pos in position_aliases:
            return position_aliases[base_pos]
    return "Unknown"

# --- Position-based metrics for radar charts ---
position_metrics = {
    "Goalkeeper": [
        "Pass Completion Ratio", "Save Ratio", "Clean Sheets",
        "Saves Held", "Saves Parried", "Saves Tipped"
    ],
    "Sweeper": [
        "Assists", "Goals", "Headers Won", "Tackle Completion Ratio",
        "Interceptions", "Pass Completion Ratio"
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
    "Unknown": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Expected Assists", "Key Passes"
    ]
}

# --- Parsing HTML export to DataFrame ---
def parse_html_to_df(file):
    soup = BeautifulSoup(file, "html.parser")
    table = soup.find("table")
    if table is None:
        st.error("No table found in the uploaded HTML file.")
        return pd.DataFrame()
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    # Make column headers unique if needed
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
            # Skipping rows with column mismatch
            pass

    df = pd.DataFrame(rows, columns=unique_headers)
    return df

# --- Function to clean and convert numeric columns ---
def clean_numeric_columns(df):
    for col in df.columns:
        # Try to convert columns that look numeric
        try:
            df[col] = pd.to_numeric(df[col].str.replace("%", "").str.replace(",", "."), errors='coerce')
        except Exception:
            pass
    return df

# --- Radar chart plotting ---
def plot_radar_chart(player_stats, metrics, player_name):
    # Normalize metric values between 0 and 1 for the radar chart
    values = []
    for metric in metrics:
        val = player_stats.get(metric, None)
        if val is None or pd.isna(val):
            val = 0
        values.append(val)

    max_val = max(values) if values else 1
    # Avoid division by zero
    if max_val == 0:
        max_val = 1
    values = [v / max_val for v in values]

    # Radar plot setup
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the plot
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    ax.plot(angles, values, color="tab:blue", linewidth=2)
    ax.fill(angles, values, color="tab:blue", alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)

    ax.set_title(f"{player_name} - Key Metrics Radar", size=12, y=1.1)
    plt.tight_layout()
    st.pyplot(fig)

# --- Main UI ---
# Upload squad file
uploaded_squad_file = st.file_uploader("Upload your FM24 Squad HTML export", type=["html"])
# Upload transfer market file
uploaded_transfer_file = st.file_uploader("Upload your FM24 Transfer Market HTML export", type=["html"])

# Initialize dataframes
squad_df = pd.DataFrame()
transfer_df = pd.DataFrame()

if uploaded_squad_file:
    squad_df = parse_html_to_df(uploaded_squad_file)
    squad_df = clean_numeric_columns(squad_df)
    if squad_df.empty:
        st.error("Failed to parse squad file or file is empty.")
    else:
        # Normalize position column if exists
        if "Position" in squad_df.columns:
            squad_df["Normalized Position"] = squad_df["Position"].apply(normalize_position)
        else:
            squad_df["Normalized Position"] = "Unknown"

        st.subheader("📋 Squad Player Stats")
        st.dataframe(squad_df, use_container_width=True)

        st.subheader("🤖 Ask AI about your Squad")
        squad_question = st.text_area("Ask a question about your squad (e.g., 'Who should I sell?', 'Top midfielders?')")
        if st.button("Analyze Squad with AI") and squad_question.strip():
            with st.spinner("Analyzing squad with AI..."):
                try:
                    prompt = f"""
                    You are a football analyst. Here are player stats from a Football Manager 2024 squad:

                    {squad_df.to_markdown(index=False)}

                    Answer the user question based on these stats:

                    Question: {squad_question}
                    """
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a tactical football analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 🧠 AI Insights on Squad")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"API call failed: {e}")

if uploaded_transfer_file:
    transfer_df = parse_html_to_df(uploaded_transfer_file)
    transfer_df = clean_numeric_columns(transfer_df)
    if transfer_df.empty:
        st.error("Failed to parse transfer market file or file is empty.")
    else:
        # Normalize position column if exists
        if "Position" in transfer_df.columns:
            transfer_df["Normalized Position"] = transfer_df["Position"].apply(normalize_position)
        else:
            transfer_df["Normalized Position"] = "Unknown"

        st.subheader("📋 Transfer Market Players")
        st.dataframe(transfer_df, use_container_width=True)

        st.subheader("🤖 Ask AI about Transfer Market")
        transfer_question = st.text_area("Ask a question about transfer targets (e.g., 'Best young strikers under 23?')")
        if st.button("Analyze Transfers with AI") and transfer_question.strip():
            with st.spinner("Analyzing transfer market with AI..."):
                try:
                    prompt = f"""
                    You are a football analyst. Here are player stats from a Football Manager 2024 transfer market export:

                    {transfer_df.to_markdown(index=False)}

                    Answer the user question based on these stats:

                    Question: {transfer_question}
                    """
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a tactical football analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 🧠 AI Insights on Transfers")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"API call failed: {e}")

# --- Player detail and radar chart ---

def display_player_details(df, title="Player Details"):
    st.subheader(title)
    player_names = df["Name"].dropna().unique().tolist()
    selected_player = st.selectbox("Select a player to view details and radar chart", player_names)
    if selected_player:
        player_row = df[df["Name"] == selected_player].iloc[0]
        st.markdown(f"### {selected_player}")
        # Show player raw stats
        st.write(player_row)

        # Get normalized position for metrics selection
        pos = player_row.get("Normalized Position", "Unknown")
        metrics = position_metrics.get(pos, position_metrics["Unknown"])

        # Prepare stats dict for radar chart
        stats = {}
        for m in metrics:
            val = player_row.get(m)
            # Try to convert to float if possible
            try:
                val = float(val)
            except Exception:
                val = 0
            stats[m] = val

        if len(stats) >= 3:
            plot_radar_chart(stats, list(stats.keys()), selected_player)
        else:
            st.warning("⚠️ Not enough data to generate radar chart for this player.")

if not squad_df.empty:
    display_player_details(squad_df, "Squad Player Details")

if not transfer_df.empty:
    display_player_details(transfer_df, "Transfer
