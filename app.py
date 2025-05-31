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
    "AM (L)": "Inside Forward",  # Your requested alias 1
    "AM (R)": "Inside Forward",  # Your requested alias 2
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
        "Assists", "Goals", "Headers", "Tackle Completion Ratio",
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
def parse_html(file):
    try:
        content = file.read()
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

        # Prefer <thead>, fallback to first <tr> if no thead
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        else:
            first_row = table.find("tr")
            headers = [th.get_text(strip=True) for th in first_row.find_all(["th", "td"])]

        rows = []
        tbody = table.find("tbody")
        if tbody:
            trs = tbody.find_all("tr")
        else:
            trs = table.find_all("tr")[1:]  # skip header row

        for tr in trs:
            cells = tr.find_all("td")
            if not cells:
                continue
            row = [td.get_text(strip=True) for td in cells]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Clean and convert columns that are numeric
        for col in df.columns:
            # Remove % and commas for numeric conversion
            df[col] = df[col].str.replace("%", "").str.replace(",", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Normalize position columns
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

    st.subheader(f"{player_name} — {player_data.get('Nationality', '')} — {player_data.get('Position', '')}")

    # Basic Info and Key Stats in 2 columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Basic Info**")
        st.write(f"**Age:** {player_data.get('Age', 'N/A')}")
        st.write(f"**Club:** {player_data.get('Club', 'N/A')}")
        st.write(f"**Current Ability:** {player_data.get('CA', 'N/A')}")
        st.write(f"**Potential Ability:** {player_data.get('PA', 'N/A')}")
        st.write(f"**Transfer Value:** {player_data.get('Transfer Value', 'N/A')}")
        st.write(f"**Wage:** {player_data.get('Wage', 'N/A')}")

    with col2:
        st.markdown("**Key Stats**")
        offensive = ["Gls", "Ast", "xG/90", "xG-OP", "xA", "K Pas", "Drb"]
        defensive = ["Tck R", "Itc", "Hdrs"]
        passing = ["Pas %"]
        goalkeeper = ["Sv %", "Clean Sheets", "Svh", "Svp", "Sv"]

        def show_stats(stats):
            for stat in stats:
                val = player_data.get(stat)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    st.write(f"**{stat}:** {val}")

        show_stats
