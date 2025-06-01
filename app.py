import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import openai

# --- SETUP ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("⚽ Football Manager 2024 Squad & Transfer Analyzer")

# --- API KEY INPUT ---
st.sidebar.subheader("🔐 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to generate AI scouting reports.")
else:
    openai.api_key = api_key

# --- FILE UPLOAD ---
st.sidebar.header("📁 Upload HTML Files")
squad_file = st.sidebar.file_uploader("Upload Squad HTML", type=["html", "htm"])
transfer_file = st.sidebar.file_uploader("Upload Transfer Market HTML", type=["html", "htm"])

# --- POSITION MAPPING ---
position_aliases = {
    "GK": "Goalkeeper", "D (C)": "Centre Back", "D (L)": "Fullback", "D (R)": "Fullback",
    "WB (L)": "Wingback", "WB (R)": "Wingback", "DM": "Defensive Midfielder", "M (C)": "Central Midfielder",
    "MC": "Central Midfielder", "AM": "Attacking Midfielder", "AM (C)": "Attacking Midfielder",
    "M (L)": "Wide Midfielder", "M (R)": "Wide Midfielder", "AM (L)": "Inside Forward",
    "AM (R)": "Inside Forward", "IF": "Inside Forward", "ST (C)": "Striker",
    "FW": "Forward", "CF": "Complete Forward", "WF": "Wide Forward",
}

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    positions = [p.strip().upper().replace(" ", "") for p in pos_str.split(",")]
    for pos in positions:
        for alias_key in position_aliases:
            if alias_key.replace(" ", "").upper() == pos:
                return position_aliases[alias_key]
    return "Unknown"

# --- KEY METRICS BY POSITION ---
position_metrics = {
    "Goalkeeper": ["Pass Completion Ratio", "Save Ratio", "Clean Sheets", "Saves Held", "Saves Parried", "Saves Tipped"],
    "Centre Back": ["Assists", "Goals", "Headers Won", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio"],
    "Fullback": ["Assists", "Goals", "Dribbles Made", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio"],
    "Wingback": ["Assists", "Goals", "Dribbles Made", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio"],
    "Defensive Midfielder": ["Assists", "Goals", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio", "Key Passes"],
    "Central Midfielder": ["Assists", "Goals", "Key Passes", "Dribbles Made", "Pass Completion Ratio", "Interceptions"],
    "Attacking Midfielder": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Expected Assists", "Key Passes"],
    "Wide Midfielder": ["Assists", "Goals", "Dribbles Made", "Key Passes", "Pass Completion Ratio", "Expected Goals per 90 Minutes"],
    "Inside Forward": ["Assists", "Goals", "Dribbles Made", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes"],
    "Complete Forward": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes"],
    "Striker": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes"],
    "Forward": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes"],
    "Wide Forward": ["Assists", "Goals", "Dribbles Made", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes"],
    "Unknown": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Expected Assists", "Key Passes"]
}

# --- HEADER TRANSLATION ---
header_mapping = {
    "Inf": "Information", "Name": "Name", "Club": "Club", "Position": "Position", "Age": "Age",
    "Potential": "Potential", "Ability": "Current Ability", "CA": "Current Ability", "PA": "Potential Ability",
    "Transfer Value": "Transfer Value", "Wage": "Wage", "Ast": "Assists", "Gls": "Goals",
    "xG/90": "Expected Goals per 90 Minutes", "xG-OP": "Expected Goals Overperformance", "xA": "Expected Assists",
    "K Pas": "Key Passes", "Drb": "Dribbles Made", "Pas %": "Pass Completion Ratio",
    "Itc": "Interceptions", "Hdrs": "Headers Won", "Tck R": "Tackle Completion Ratio",
    "Sv %": "Save Ratio", "Clean Sheets": "Clean Sheets", "Svh": "Saves Held",
    "Svp": "Saves Parried", "Svt": "Saves Tipped", "Rec": "Recommendation"
}

def deduplicate_headers(headers):
    seen = {}
    deduped = []
    for h in headers:
        if h not in seen:
            seen[h] = 1
            deduped.append(h)
        else:
            seen[h] += 1
            deduped.append(f"{h} ({seen[h]})")
    return deduped

def parse_currency(value):
    if not isinstance(value, str):
        return None
    value = value.replace("£", "").replace(",", "").strip().upper()
    multiplier = 1
    if value.endswith("K"):
        multiplier = 1_000
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000
        value = value[:-1]
    try:
        return float(value) * multiplier
    except:
        return None

def parse_html(file):
    try:
        # Read all tables using lxml parser
        tables = pd.read_html(file, flavor="lxml")
    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return pd.DataFrame()

    # Find table with the most relevant columns (intersection with METRIC_MAPPING keys)
    df = max(tables, key=lambda t: len(set(t.columns) & set(header_mapping.values())))

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename columns according to header mapping
    df = df.rename(columns=header_mapping)

    # Normalize position names if present
    if "Position" in df.columns:
        df["Normalized Position"] = df["Position"].apply(normalize_position)

    # Clean and convert numeric columns
    for col in df.columns:
        if col in ["Transfer Value", "Wage"]:
            df[col] = df[col].apply(parse_currency)
        elif col not in ["Name", "Club", "Position", "Normalized Position"]:
            # Clean strings safely
            df[col] = df[col].astype(str).apply(lambda x: x.replace(",", "").replace("%", "").strip())
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# --- BAR CHART FUNCTION ---
def plot_player_barchart(player_row, metrics, player_name):
    labels = [m for m in metrics if player_row.get(m) not in [None, "", "N/A", np.nan]]
    values = [float(player_row[m]) if pd.notnull(player_row[m]) else 0 for m in labels]

    if not labels or not values:
        st.warning("Not enough data to plot.")
        return

    fig, ax = plt.subplots(figsize=(5, 0.4 * len(labels) + 1))
    bars = ax.barh(labels, values, color='steelblue')
    for bar in bars:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center', fontsize=8)
    ax.set_title(player_name)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    st.pyplot(fig)

# --- AI SCOUTING REPORT ---
def get_ai_scouting_report(player_name, player_row):
    if not api_key:
        return "API key not provided."
    prompt = f"""You are a professional football scout. Write a short, clear scouting report on the player {player_name} based on the following stats:

{player_row.to_dict()}

Highlight strengths, weaknesses, and role suitability."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert football scout."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from OpenAI: {e}"

# --- LOAD DATA ---
df_squad = parse_html(squad_file) if squad_file else None
df_transfer = parse_html(transfer_file) if transfer_file else None

# --- TABS ---
tab1, tab2 = st.tabs(["🏟️ Squad", "🌍 Transfer Market"])

with tab1:
    if df_squad is not None:
        st.subheader("Squad Players")
        st.dataframe(df_squad)
        player = st.selectbox("Select a player", df_squad["Name"].unique())
        selected = df_squad[df_squad["Name"] == player].iloc[0]
        pos = selected["Normalized Position"]
        plot_player_barchart(selected, position_metrics.get(pos, []), player)
        with st.expander("🧠 AI Scouting Report"):
            st.markdown(get_ai_scouting_report(player, selected))
    else:
        st.info("Upload a squad HTML file.")

with tab2:
    if df_transfer is not None:
        st.subheader("Transfer Market Players")
        st.dataframe(df_transfer)
        player = st.selectbox("Select transfer target", df_transfer["Name"].unique())
        selected = df_transfer[df_transfer["Name"] == player].iloc[0]
        pos = selected["Normalized Position"]
        plot_player_barchart(selected, position_metrics.get(pos, []), player)
        with st.expander("🧠 AI Scouting Report"):
            st.markdown(get_ai_scouting_report(player, selected))
    else:
        st.info("Upload a transfer market HTML file.")

st.markdown("Made for ⚽ FM24 managers — powered by GPT 🧠 + Streamlit 🛠️")
