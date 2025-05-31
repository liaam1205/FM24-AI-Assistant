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
api_key = st.secrets.get("API_KEY", "")
if not api_key:
    st.error("OpenAI API key not found in secrets. Please add it before using the AI features.")
client = openai.OpenAI(api_key=api_key) if api_key else None

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
    "AM (L)": "Inside Forward",  # Dual aliases for Inside Forward
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

# --- Helper: Clean metric keys for parsing ---
metric_key_map = {
    "Assists": "Ast",
    "Goals": "Gls",
    "Expected Goals per 90 Minutes": "xG/90",
    "Expected Goals Overperformance": "xG-OP",
    "Expected Assists": "xA",
    "Key Passes": "K Pas",
    "Dribbles Made": "Drb",
    "Pass Completion Ratio": "Pas %",
    "Interceptions": "Itc",
    "Headers": "Hdrs",
    "Tackle Completion Ratio": "Tck R",
    "Save Ratio": "Sv %",
    "Clean Sheets": "Clean Sheets",
    "Saves Held": "Svh",
    "Saves Parried": "Svp",
    "Saves Tipped": "Sv",
    "Conversion %": "Conversion %",
}

# --- Parse FM24 HTML export into DataFrame ---
def parse_html(html_file):
    try:
        soup = BeautifulSoup(html_file, "html.parser")
        table = soup.find("table")
        if not table:
            st.error("No table found in the HTML file.")
            return None

        # Try to get headers from thead or first tr
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        else:
            # fallback: get headers from first tr if no thead
            first_row = table.find("tr")
            headers = [th.get_text(strip=True) for th in first_row.find_all(["th", "td"])]

        rows = []
        tbody = table.find("tbody")
        if tbody:
            row_tags = tbody.find_all("tr")
        else:
            # fallback: all tr except first if no tbody
            row_tags = table.find_all("tr")[1:]

        for tr in row_tags:
            cells = tr.find_all("td")
            if not cells or len(cells) != len(headers):
                continue
            row = [td.get_text(strip=True) for td in cells]
            rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Clean percentage signs and convert to numeric where possible
        for col in df.columns:
            df[col] = df[col].str.replace("%", "", regex=False)
            df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Normalize position column if exists
        pos_col = None
        for c in ["Position", "Pos"]:
            if c in df.columns:
                pos_col = c
                break
        if pos_col:
            df["Normalized Position"] = df[pos_col].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df
    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return None

# --- Plot radar (pizza) chart for a player ---
def plot_player_radar(player_data, position, title="Player Radar Chart"):
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    # Map readable metrics to actual columns in data
    cols = [metric_key_map.get(m, m) for m in metrics]
    values = []
    labels = []

    for m, col in zip(metrics, cols):
        val = player_data.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        else:
            try:
                val = float(val)
            except Exception:
                val = 0
        values.append(val)
        labels.append(m)

    # Need at least 3 metrics to plot
    if len(values) < 3 or all(v == 0 for v in values):
