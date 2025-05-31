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
    "AM (L)": "Inside Forward",
    "AM (R)": "Inside Forward",
    "W (L)": "Winger",
    "W (R)": "Winger",
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

# --- Parse HTML file to DataFrame ---
def parse_html_to_df(html_file):
    try:
        soup = BeautifulSoup(html_file, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

        headers = [th.get_text(strip=True) for th in table.find_all("th")]

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            row = [td.get_text(strip=True) for td in cells]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Convert numeric columns where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "").str.replace(",", ""), errors='coerce')

        # Normalize positions
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

# --- Plot radar (pizza) chart for player ---
def plot_player_radar(player_data, metrics, title="Player Radar Chart"):
    labels = metrics
    values = []
    for m in metrics:
        val = player_data.get(m)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        values.append(float(val))

    # close the loop
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

# --- Display player details and radar ---
def display_player_details(df, player_name):
    player_row = df[df["Name"] == player_name]
    if player_row.empty:
        st.warning("Player not found in dataset.")
        return
    player_data = player_row.iloc[0].to_dict()

    st.subheader(f"Player Details: {player_name} ({player_data.get('Nationality', '')}, {player_data.get('Position', '')})")

    # Show player stats table excluding some big columns
    details_to_show = {k: v for k, v in player_data.items() if k not in ['Normalized Position']}
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
    squad_df = parse_html_to_df(squad_file)
    if squad_df is not None:
        st.success(f"Squad data loaded: {len(squad_df)} players")
        st.dataframe(squad_df)

if transfer_file is not None:
    transfer_df = parse_html_to_df(transfer_file)
    if transfer_df is not None:
        st.success(f"Transfer market data loaded: {len(transfer_df)} players")
        st.dataframe(transfer_df)

# --- Player search and display ---

st.markdown("---")
st.subheader("🔍 Search Player Details & Radar Chart")

player_name_input = st.text_input("Enter player name to display details (case sensitive):")

if player_name_input:
    df_to_search = squad_df if squad_df is not None else transfer_df
    if df_to_search is None:
        st.warning("Please upload at least one dataset to search players.")
    else:
        display_player_details(df_to_search, player_name_input)

# --- Transfer market search by criteria ---

st.markdown("---")
st.subheader("📊 Transfer Market Search by Criteria")

if transfer_df is not None:
    metrics_options = list(transfer_df.columns)
    metrics_options = [col for col in metrics_options if transfer_df[col].dtype in [float, int]]
    selected_metric = st.selectbox("Select metric to filter by", options=metrics_options)

    if selected_metric:
        min_val = float(transfer_df[selected_metric].min())
        max_val = float(transfer_df[selected_metric].max())
        val_range = st.slider(f"Filter {selected_metric} range", min_value=min_val, max_value=max_val,
                              value=(min_val, max_val))

        filtered_transfer = transfer_df[
            (transfer_df[selected_metric] >= val_range[0]) & (transfer_df[selected_metric] <= val_range[1])
        ]

        st.write(f"Players matching criteria: {len(filtered_transfer)}")
        st.dataframe(filtered_transfer)

else:
    st.info("Upload transfer market data to use search functionality.")

# --- AI Chat for questions about squad or transfer ---

st.markdown("---")
st.subheader("🤖 Ask AI about your Squad or Transfer Market")

user_question = st.text_area("Enter your question about your squad or transfer data:")

def generate_ai_answer(question, squad_df=None, transfer_df=None):
    prompt = f"You are a Football Manager 2024 assistant analyzing player data.\n"
    if squad_df is not None:
        prompt += f"Squad Data Sample:\n{ squad_df.head(5).to_dict(orient='records') }\n"
    if transfer_df is not None:
        prompt += f"Transfer Market Data Sample:\n{ transfer_df.head(5).to_dict(orient='records') }\n"
    prompt += f"Answer this question based on the above data:\n{question}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with AI: {e}"

if st.button("Ask AI") and user_question.strip():
    answer = generate_ai_answer(user_question, squad_df, transfer_df)
    st.markdown("**AI Answer:**")
    st.write(answer)
