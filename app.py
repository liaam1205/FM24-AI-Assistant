import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("⚽ FM24 Squad & Transfer Analyzer")

# --- SIDEBAR INPUT ---
st.sidebar.subheader("🔐 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to generate AI scouting reports.")

# --- FILE UPLOADS ---
st.sidebar.header("📁 Upload HTML Files")
squad_file = st.sidebar.file_uploader("Upload Squad HTML", type=["html", "htm"])
transfer_file = st.sidebar.file_uploader("Upload Transfer Market HTML", type=["html", "htm"])

# --- POSITION NORMALIZATION ---
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
        for key in position_aliases:
            if key.replace(" ", "").upper() == pos:
                return position_aliases[key]
    return "Unknown"

# --- METRICS BY POSITION ---
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

# --- RAW HEADER TRANSLATION ---
raw_to_final_header = {
    "Inf": "Information", "Name": "Name", "Club": "Club", "Position": "Position", "Age": "Age",
    "Potential": "Potential", "Ability": "Current", "CA": "Current Ability", "PA": "Potential Ability",
    "Transfer Value": "Transfer Value", "Wage": "Wage", "Rec": "Recommendation",
    "Ast": "Assists", "Gls": "Goals", "xG/90": "Expected Goals per 90 Minutes", 
    "xG-OP": "Expected Goals Overperformance", "xA": "Expected Assists", "K Pas": "Key Passes",
    "Drb": "Dribbles Made", "Pas %": "Pass Completion Ratio", "Itc": "Interceptions",
    "Hdrs": "Headers Won", "Tck R": "Tackle Completion Ratio", "Sv %": "Save Ratio",
    "Clean Sheets": "Clean Sheets", "Svh": "Saves Held", "Svp": "Saves Parried", "Svt": "Saves Tipped"
}

def deduplicate_column_names(headers):
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

def parse_html(file) -> pd.DataFrame:
    dfs = pd.read_html(file)
    df = dfs[0]

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    numeric_cols = [
        "Age", "Potential", "Current Ability", "Potential Ability",
        "Assists", "Goals", "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance", "Expected Assists", "Key Passes",
        "Dribbles Made", "Pass Completion Ratio", "Interceptions", "Headers Won",
        "Tackle Completion Ratio", "Save Ratio", "Clean Sheets", "Saves Held",
        "Saves Parried", "Saves Tipped"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").str.replace("%", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def clean_str(s):
        if pd.isna(s):
            return s
        # Replace all whitespace characters (including non-breaking) with a single space
        return re.sub(r'\s+', ' ', str(s)).strip()

    for col in ["Transfer Value", "Wage"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_str)

    if "Position" in df.columns:
        def normalize_position(pos):
            pos = str(pos).upper()
            if "GK" in pos:
                return "Goalkeeper"
            elif "CB" in pos or "DC" in pos:
                return "Centre Back"
            elif "FB" in pos or "WB" in pos or "DL" in pos or "DR" in pos:
                return "Full Back"
            elif "DM" in pos:
                return "Defensive Midfielder"
            elif "CM" in pos:
                return "Central Midfielder"
            elif "AM" in pos:
                return "Attacking Midfielder"
            elif "ST" in pos or "CF" in pos:
                return "Striker"
            elif "W" in pos:
                return "Winger"
            else:
                return "Other"

        df["Normalized Position"] = df["Position"].apply(normalize_position)

    return df

# --- VISUALIZATION ---
def plot_player_barchart(player_row, metrics, player_name):
    labels = [m for m in metrics if pd.notnull(player_row.get(m))]
    values = [float(player_row[m]) for m in labels]

    if not labels:
        st.warning("No data to visualize.")
        return

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(labels)))
    bars = ax.barh(labels, values, color="teal")
    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va="center")
    ax.set_title(player_name)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    st.pyplot(fig)

# --- AI REPORT ---
def get_ai_scouting_report(player_name, player_row):
    if not api_key:
        return "API key not provided."

    prompt = f"""You are a professional football scout. Write a short, clear scouting report on the player {player_name} based on the following stats:

{player_row.to_dict()}

Highlight strengths, weaknesses, and role suitability."""

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert football scout."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"

# --- LOAD DATA ---
df_squad = parse_html(squad_file) if squad_file else None
df_transfer = parse_html(transfer_file) if transfer_file else None

# --- UI LAYOUT ---
tab1, tab2 = st.tabs(["🏟️ Squad", "🌍 Transfer Market"])

with tab1:
    if df_squad is not None:
        st.subheader("📋 Squad Overview")
        st.dataframe(df_squad)
        selected_name = st.selectbox("Select player", df_squad["Name"].unique())
        selected = df_squad[df_squad["Name"] == selected_name].iloc[0]
        pos = selected["Normalized Position"]
        st.markdown(f"**Position:** {pos}")
        plot_player_barchart(selected, position_metrics.get(pos, []), selected_name)
        with st.expander("🧠 AI Scouting Report"):
            st.markdown(get_ai_scouting_report(selected_name, selected))
    else:
        st.info("Upload a squad HTML file to begin.")

with tab2:
    if df_transfer is not None:
        st.subheader("🌍 Transfer Market Overview")
        st.dataframe(df_transfer)
        selected_name = st.selectbox("Select transfer target", df_transfer["Name"].unique())
        selected = df_transfer[df_transfer["Name"] == selected_name].iloc[0]
        pos = selected["Normalized Position"]
        st.markdown(f"**Position:** {pos}")
        plot_player_barchart(selected, position_metrics.get(pos, []), selected_name)
        with st.expander("🧠 AI Scouting Report"):
            st.markdown(get_ai_scouting_report(selected_name, selected))
    else:
        st.info("Upload a transfer market HTML file to begin.")

st.caption("Built for FM24 ⚽ Powered by GPT + Streamlit 🚀")
