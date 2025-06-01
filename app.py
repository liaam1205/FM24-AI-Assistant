import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np
import time

# --- App Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("⚽ Football Manager 2024 Squad & Transfer Analyzer")
st.markdown(
    """
Upload your FM24 exported **squad** and **transfer market** HTML files to analyze your squad and transfer targets.  
Ask AI questions, get detailed player stats with radar/bar charts, and search transfer market players easily!
"""
)

# --- Sidebar Instructions ---
with st.sidebar:
    st.header("Instructions")
    st.write(
        """
1. Upload your **Squad** and/or **Transfer Market** HTML export files.  
2. Switch between **Squad** and **Transfer Market** tabs below.  
3. Search/select a player to view detailed stats and AI scouting report.  
4. Expand the sections for charts and reports.
"""
    )
    st.markdown("---")

# --- OpenAI API Key ---
api_key = st.secrets["API_KEY"]
openai.api_key = api_key

# --- Position Normalization ---
position_aliases = {
    "GK": "Goalkeeper",
    "D(C)": "Centre Back",
    "D(L)": "Fullback",
    "D(R)": "Fullback",
    "WB(L)": "Wingback",
    "WB(R)": "Wingback",
    "DM": "Defensive Midfielder",
    "M(C)": "Central Midfielder",
    "MC": "Central Midfielder",
    "AM": "Attacking Midfielder",
    "AM(C)": "Attacking Midfielder",
    "M(L)": "Wide Midfielder",
    "M(R)": "Wide Midfielder",
    "AM(L)": "Inside Forward",
    "AM(R)": "Inside Forward",
    "IF": "Inside Forward",
    "ST(C)": "Striker",
    "FW": "Forward",
    "CF": "Complete Forward",
    "WF": "Wide Forward",
}

def normalize_position(pos_str: str) -> str:
    if not isinstance(pos_str, str):
        return "Unknown"
    positions = [p.strip().upper().replace(" ", "").replace(")", "").replace("(", "") for p in pos_str.split(",")]
    for pos in positions:
        if pos in position_aliases:
            return position_aliases[pos]
    return "Unknown"

# --- Position-based metrics for charts ---
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
    "Inside Forward": [
        "Assists", "Goals", "Dribbles Made", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Key Passes"
    ],
    "Complete Forward": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Key Passes"
    ],
    "Striker": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Key Passes"
    ],
    "Forward": [
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Key Passes"
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

# --- Header mapping for unique column names ---
header_mapping = {
    "Inf": "Information",
    "Name": "Name",
    "Club": "Club",
    "Position": "Position",
    "Age": "Age",
    "Potential": "Potential",
    "Ability": "Current Ability",
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
    "Svt": "Saves Tipped",
}

def parse_html_with_pandas(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        html = file.read().decode("utf-8")
        # Read all tables from the HTML
        dfs = pd.read_html(html, encoding='utf-8')
        if not dfs:
            st.error("No tables found in the HTML file.")
            return None
        # Usually FM export has only one main table but sometimes multiple,
        # choose the largest table by number of rows as main data
        df = max(dfs, key=lambda d: d.shape[0])
        
        # Normalize headers: rename columns if necessary (map headers)
        df.rename(columns=header_mapping, inplace=True)
        
        # Drop fully empty columns and rows
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        
        # Normalize position column
        if "Position" in df.columns:
            df["Normalized Position"] = df["Position"].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"Error parsing HTML with pandas: {e}")
        return None

def plot_player_barchart(player_row: pd.Series, metrics: list[str], player_name: str):
    labels = [m for m in metrics if pd.notnull(player_row.get(m)) and player_row.get(m) not in ["", "N/A"]]
    def clean_value(val):
        if isinstance(val, str):
            val = val.replace("%", "")
        try:
            return float(val)
        except:
            return 0.0

    values = [clean_value(player_row[m]) for m in labels]

    if not labels or not values:
        st.warning("Not enough data to create bar chart.")
        return

    fig, ax = plt.subplots(figsize=(5, 0.4 * len(labels) + 1))
    bars = ax.barh(labels, values, color='tab:blue', alpha=0.85)

    for bar in bars:
        width = bar.get_width()
        offset = max(width * 0.02, 0.3)
        text_color = "white" if width > plt.xlim()[1] * 0.1 else "black"
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va='center',
            fontsize=8,
            color=text_color,
            fontweight='bold'
        )

    ax.set_title(f"Key Stats for {player_name}", fontsize=14, pad=12)
    ax.set_xlabel("Value", fontsize=10)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.tick_params(axis='y', labelsize=10, colors='black')
    ax.tick_params(axis='x', labelsize=9, colors='black')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')

    plt.tight_layout()
    st.pyplot(fig)

def get_ai_scouting_report(player_name: str, player_data: pd.Series) -> str:
    prompt = f"""
You are a football scouting expert. Based on the following player data, write a concise, insightful scouting report for {player_name}:

{player_data.to_dict()}

Focus on strengths, weaknesses, and potential.
"""
    try:
        with st.spinner("Generating AI scouting report..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert football scout."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
            )
            report = response.choices[0].message.content.strip()
            time.sleep(0.5)
            return report
    except Exception as e:
        return f"Error generating AI scouting report: {e}"

# --- Upload Section ---
with st.sidebar:
    squad_file = st.file_uploader("Upload Squad HTML file", type=["html", "htm"])
    transfer_file = st.file_uploader("Upload Transfer Market HTML file", type=["html", "htm"])

df_squad = parse_html(squad_file) if squad_file else None
df_transfer = parse_html(transfer_file) if transfer_file else None

# --- Tabs for Squad and Transfer Market ---
tab1, tab2 = st.tabs(["🏟️ Squad", "🌍 Transfer Market"])

def player_selector(df: pd.DataFrame, label: str) -> str | None:
    player_names = df["Name"].dropna().drop_duplicates().tolist()
    search = st.text_input(f"Search {label} player by name")
    if search:
        filtered = [p for p in player_names if search.lower() in p.lower()]
    else:
        filtered = player_names
    if filtered:
        return st.selectbox(f"Select {label} player", filtered)
    else:
        st.info(f"No {label} players found matching '{search}'.")
        return None

with tab1:
    st.header("Squad Players")
    if df_squad is not None and not df_squad.empty:
        st.dataframe(df_squad.head(15))
        selected_player = player_selector(df_squad, "Squad")
        if selected_player:
            filtered_squad = df_squad[df_squad["Name"] == selected_player]
            if not filtered_squad.empty:
                player_row = filtered_squad.iloc[0]
                pos = player_row.get("Normalized Position", "Unknown")
                metrics = position_metrics.get(pos, position_metrics["Unknown"])

                # Display top metrics
                cols = st.columns(4)
                stat_list = ["Age", "Current Ability", "Potential Ability", "Goals", "Assists", "Pass Completion Ratio"]
                for i, stat in enumerate(stat_list[:4]):
                    with cols[i]:
                        val = player_row.get(stat)
                        if pd.notna(val):
                            st.metric(label=stat, value=f"{val}")
                        else:
                            st.metric(label=stat, value="N/A")

                # Player stats chart
                with st.expander(f"Player Stats Chart for {selected_player}"):
                    plot_player_barchart(player_row, metrics, selected_player)

                # AI scouting report
                with st.expander(f"AI Scouting Report for {selected_player}"):
                    report = get_ai_scouting_report(selected_player, player_row)
                    st.write(report)
            else:
                st.warning(f"No data found for squad player: {selected_player}")
    else:
        st.info("Please upload a Squad HTML file to view squad players.")

with tab2:
    st.header("Transfer Market Players")
    if df_transfer is not None and not df_transfer.empty:
        st.dataframe(df_transfer.head(15))
        selected_transfer_player = player_selector(df_transfer, "Transfer Market")
        if selected_transfer_player:
            filtered_transfer = df_transfer[df_transfer["Name"] == selected_transfer_player]
            if not filtered_transfer.empty:
                player_row = filtered_transfer.iloc[0]
                pos = player_row.get("Normalized Position", "Unknown")
                metrics = position_metrics.get(pos, position_metrics["Unknown"])

                cols = st.columns(4)
                stat_list = ["Age", "Current Ability", "Potential Ability", "Goals", "Assists", "Pass Completion Ratio"]
                for i, stat in enumerate(stat_list[:4]):
                    with cols[i]:
                        val = player_row.get(stat)
                        if pd.notna(val):
                            st.metric(label=stat, value=f"{val}")
                        else:
                            st.metric(label=stat, value="N/A")

                with st.expander(f"Player Stats Chart for {selected_transfer_player}"):
                    plot_player_barchart(player_row, metrics, selected_transfer_player)

                with st.expander(f"AI Scouting Report for {selected_transfer_player}"):
                    report = get_ai_scouting_report(selected_transfer_player, player_row)
                    st.write(report)
            else:
                st.warning(f"No data found for transfer player: {selected_transfer_player}")
    else:
        st.info("Please upload a Transfer Market HTML file to view players.")

# --- Inform user if no data loaded in either tab ---
if (df_squad is None or df_squad.empty) and (df_transfer is None or df_transfer.empty):
    st.info("Upload Squad and/or Transfer Market HTML files using the sidebar to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    Developed with ❤️ for Football Manager 2024 enthusiasts.  
    Powered by OpenAI GPT and Streamlit.
    """
)
