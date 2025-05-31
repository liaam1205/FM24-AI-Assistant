import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np

# --- App Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Market Analyzer", layout="wide")
st.title("⚽ FM24 Squad & Transfer Market Analyzer")
st.markdown(
    "Upload your FM24 exported HTML squad and transfer market files to get AI insights and detailed player analysis with position-based radar charts."
)

# --- API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- Constants ---

# Comprehensive position metrics map including fallback logic
position_metrics_map = {
    "Goalkeeper": [
        "Save Ratio",
        "Clean Sheets",
        "Saves Held",
        "Saves Parried",
        "Saves Tipped",
        "Pass Completion Ratio",
    ],
    "Central Defender": [
        "Tackle Completion Ratio",
        "Interceptions",
        "Headers Won",
        "Pass Completion Ratio",
        "Key Passes",
        "Dribbles Made",
    ],
    "Left Back": [
        "Tackle Completion Ratio",
        "Interceptions",
        "Headers Won",
        "Pass Completion Ratio",
        "Key Passes",
        "Dribbles Made",
    ],
    "Right Back": [
        "Tackle Completion Ratio",
        "Interceptions",
        "Headers Won",
        "Pass Completion Ratio",
        "Key Passes",
        "Dribbles Made",
    ],
    "Wing Back": [
        "Tackle Completion Ratio",
        "Interceptions",
        "Headers Won",
        "Pass Completion Ratio",
        "Key Passes",
        "Dribbles Made",
    ],
    "Defensive Midfielder": [
        "Pass Completion Ratio",
        "Tackle Completion Ratio",
        "Interceptions",
        "Key Passes",
        "Dribbles Made",
        "Expected Assists",
    ],
    "Central Midfielder": [
        "Pass Completion Ratio",
        "Key Passes",
        "Assists",
        "Expected Assists",
        "Dribbles Made",
        "Tackle Completion Ratio",
    ],
    "Attacking Midfielder": [
        "Pass Completion Ratio",
        "Key Passes",
        "Assists",
        "Expected Assists",
        "Dribbles Made",
        "Goals",
        "Expected Goals per 90 Minutes",
        "Conversion %",
    ],
    "Left Midfielder": [
        "Pass Completion Ratio",
        "Key Passes",
        "Assists",
        "Expected Assists",
        "Dribbles Made",
        "Goals",
        "Expected Goals per 90 Minutes",
    ],
    "Right Midfielder": [
        "Pass Completion Ratio",
        "Key Passes",
        "Assists",
        "Expected Assists",
        "Dribbles Made",
        "Goals",
        "Expected Goals per 90 Minutes",
    ],
    "Left Winger": [
        "Pass Completion Ratio",
        "Key Passes",
        "Assists",
        "Expected Assists",
        "Dribbles Made",
        "Goals",
        "Expected Goals per 90 Minutes",
        "Conversion %",
    ],
    "Right Winger": [
        "Pass Completion Ratio",
        "Key Passes",
        "Assists",
        "Expected Assists",
        "Dribbles Made",
        "Goals",
        "Expected Goals per 90 Minutes",
        "Conversion %",
    ],
    "Striker": [
        "Goals",
        "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance",
        "Assists",
        "Expected Assists",
        "Conversion %",
        "Dribbles Made",
        "Key Passes",
    ],
    "Centre Forward": [
        "Goals",
        "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance",
        "Assists",
        "Expected Assists",
        "Conversion %",
        "Dribbles Made",
        "Key Passes",
    ],
}

# Fallback groups for positions not exactly matched
fallback_positions = {
    "Goalkeeper": ["Goalkeeper", "GK"],
    "Central Defender": ["Centre Back", "Central Defender", "D (C)", "Defender"],
    "Left Back": ["Left Back", "D (L)"],
    "Right Back": ["Right Back", "D(R)"],
    "Wing Back": ["Wing Back", "WB"],
    "Defensive Midfielder": ["Defensive Midfielder", "DM", "Holding Midfielder"],
    "Central Midfielder": ["Central Midfielder", "M (C)"],
    "Attacking Midfielder": ["Attacking Midfielder", "AM (C)"],
    "Left Midfielder": ["Left Midfielder", "M (L)"],
    "Right Midfielder": ["Right Midfielder", "M (R)"],
    "Left Winger": ["Left Winger", "AM (L)"],
    "Right Winger": ["Right Winger", "AM (R)"],
    "Striker": ["Striker", "ST", "Forward"],
    "Centre Forward": ["Centre Forward", "CF"],
}

# --- Helpers ---


def parse_html_to_df(file):
    try:
        soup = BeautifulSoup(file, "html.parser")
        table = soup.find("table")
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
            else:
                # skip rows with mismatch but log
                st.warning(f"Skipping row with column mismatch: {cols}")

        df = pd.DataFrame(rows, columns=unique_headers)
        return df
    except Exception as e:
        st.error(f"Failed to parse HTML file: {e}")
        return pd.DataFrame()


def clean_and_prepare_df(df):
    # Drop rows without player name
    if "Name" in df.columns:
        df = df[df["Name"].notna() & (df["Name"] != "")]
    # Convert numeric columns to float if possible
    for col in df.columns:
        # Skip 'Name' and 'Position'
        if col not in ["Name", "Position"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def get_position_metrics(position):
    position = position.strip()
    # Exact match
    if position in position_metrics_map:
        return position_metrics_map[position]
    # Fallback matches
    for key, aliases in fallback_positions.items():
        if any(alias.lower() in position.lower() for alias in aliases):
            return position_metrics_map[key]
    # Default fallback if nothing found
    return [
        "Goals",
        "Assists",
        "Expected Goals per 90 Minutes",
        "Expected Assists",
        "Pass Completion Ratio",
    ]


def plot_radar_chart(player_stats, metrics, player_name, position):
    if len(metrics) < 3:
        st.warning("Not enough metrics to plot radar chart (need at least 3).")
        return

    values = []
    labels = []
    for metric in metrics:
        val = player_stats.get(metric)
        if val is None or pd.isna(val):
            # Substitute zero if missing
            val = 0
        values.append(val)
        labels.append(metric)

    # Normalize values for radar plot (0-1 scale)
    max_vals = [max(1, np.nanmax(player_stats.get(metric, 1))) for metric in metrics]
    max_vals = [max(1, v) for v in values]  # Avoid zero division, fallback max = value
    # We'll normalize to max of current player values (self-relative radar)
    max_val = max(values) if max(values) > 0 else 1
    normalized = [v / max_val for v in values]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    # Complete the loop
    normalized += normalized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, normalized, color="tab:blue", alpha=0.25)
    ax.plot(angles, normalized, color="tab:blue", linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f"{player_name} - {position}", fontsize=14, pad=20)
    st.pyplot(fig)


def get_player_stats(df, player_name):
    # Return stats as dict for selected player
    player_row = df[df["Name"] == player_name]
    if player_row.empty:
        return None
    return player_row.iloc[0].to_dict()


# --- App UI and Logic ---

st.sidebar.header("Upload Files")

uploaded_squad_file = st.sidebar.file_uploader(
    "Upload your FM24 squad HTML export", type=["html"], key="squad"
)
uploaded_transfer_file = st.sidebar.file_uploader(
    "Upload FM24 transfer market HTML export", type=["html"], key="transfer"
)

if not uploaded_squad_file and not uploaded_transfer_file:
    st.info("Please upload at least one file: squad or transfer market HTML export.")
    st.stop()

# Parse files if uploaded
squad_df = pd.DataFrame()
transfer_df = pd.DataFrame()

if uploaded_squad_file:
    squad_df = parse_html_to_df(uploaded_squad_file)
    squad_df = clean_and_prepare_df(squad_df)

if uploaded_transfer_file:
    transfer_df = parse_html_to_df(uploaded_transfer_file)
    transfer_df = clean_and_prepare_df(transfer_df)

# Tabs for squad and transfer market views
tab = st.radio("Choose view:", ["Squad Analysis", "Transfer Market"])

if tab == "Squad Analysis" and not squad_df.empty:
    st.header("📋 Squad Player List")
    st.dataframe(squad_df[["Name", "Position"]], use_container_width=True)

    selected_player = st.selectbox("Select a player to analyze:", squad_df["Name"].tolist())

    if selected_player:
        player_stats = get_player_stats(squad_df, selected_player)
        position = player_stats.get("Position", "Unknown")
        st.subheader(f"Player Details: {selected_player} ({position})")

        # Show all stats (except Name and Position)
        stats_display = {k: v for k, v in player_stats.items() if k not in ["Name", "Position"]}
        st.write(stats_display)

        # Radar chart
        metrics = get_position_metrics(position)
        # Check metrics exist in data and have numeric values
        metrics_available = [m for m in metrics if m in squad_df.columns and pd.notna(player_stats.get(m))]
        if len(metrics_available) >= 3:
            plot_radar_chart(player_stats, metrics_available, selected_player, position)
        else:
            st.warning("Not enough data to generate radar chart for this player.")

elif tab == "Transfer Market" and not transfer_df.empty:
    st.header("🔎 Transfer Market Player List")
    st.dataframe(transfer_df[["Name", "Position"]], use_container_width=True)

    selected_player_tm = st.selectbox(
        "Select a transfer market player to analyze:", transfer_df["Name"].tolist(), key="tm_select"
    )

    if selected_player_tm:
        player_stats_tm = get_player_stats(transfer_df, selected_player_tm)
        position_tm = player_stats_tm.get("Position", "Unknown")
        st.subheader(f"Player Details: {selected_player_tm} ({position_tm})")

        # Show all stats (except Name and Position)
        stats_display_tm = {k: v for k, v in player_stats_tm.items() if k not in ["Name", "Position"]}
        st.write(stats_display_tm)

        # Radar chart
        metrics_tm = get_position_metrics(position_tm)
        metrics_available_tm = [m for m in metrics_tm if m in transfer_df.columns and pd.notna(player_stats_tm.get(m))]
        if len(metrics_available_tm) >= 3:
            plot_radar_chart(player_stats_tm, metrics_available_tm, selected_player_tm, position_tm)
        else:
            st.warning("Not enough data to generate radar chart for this player.")

# --- AI Analysis Section ---

st.header("🤖 Ask AI about your Squad or Transfer Market")

ai_context = ""
if tab == "Squad Analysis" and not squad_df.empty:
    ai_context = f"Squad players stats:\n{ squad_df.head(10).to_markdown(index=False) }"
elif tab == "Transfer Market" and not transfer_df.empty:
    ai_context = f"Transfer Market players stats:\n{ transfer_df.head(10).to_markdown(index=False) }"

user_question = st.text_area(
    "Ask a question about your current squad or transfer market (e.g. 'Who are the best defenders?', 'Any promising strikers available?')"
)

if st.button("Get AI Insights"):
    if not user_question.strip():
        st.warning("Please enter a question for AI.")
    else:
        with st.spinner("Analyzing with AI..."):
            try:
                prompt = f"""
You are a football analyst assistant helping with Football Manager 2024 data analysis.

Here is some player data:

{ai_context}

Answer the user's question based on this data:

User question: {user_question}
"""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a tactical football analyst."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                )
                answer = response.choices[0].message.content
                st.markdown("### 🧠 AI's Insights")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Failed to get AI response: {e}")
else:
    st.info("Enter a question and press 'Get AI Insights' to get recommendations.")


# --- End of app ---
