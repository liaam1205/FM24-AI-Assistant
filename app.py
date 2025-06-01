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

# --- Header mapping for consistent column names ---
header_mapping = {
    "Inf": "Information",
    "Name": "Name",
    "Club": "Club",
    "Position": "Position",
    "Age": "Age",
    "Potential": "Potential",
    "Ability": "Potential",
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

# --- Robust HTML parser function ---
def parse_html(file) -> pd.DataFrame | None:
    try:
        html = file.read().decode("utf-8")
        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

        # Headers extraction from thead or first row
        thead = table.find("thead")
        if thead:
            header_cells = thead.find_all("th")
        else:
            first_tr = table.find("tr")
            header_cells = first_tr.find_all("th") if first_tr else []

        if not header_cells:
            st.error("No table headers found.")
            return None

        headers_raw = [th.get_text(strip=True) for th in header_cells]
        # Map headers using mapping; None for unknown headers
        headers = [header_mapping.get(h, None) for h in headers_raw]

        # Filter only columns with valid headers
        valid_cols_idx = [i for i, h in enumerate(headers) if h is not None]
        valid_headers = [h for h in headers if h is not None]

        rows = []
        trs = table.find_all("tr")
        for tr in trs:
            cells = tr.find_all("td")
            if len(cells) == 0:
                continue  # skip header or empty rows

            row = []
            for i in valid_cols_idx:
                if i < len(cells):
                    row.append(cells[i].get_text(strip=True))
                else:
                    row.append("")
            if len(row) == len(valid_headers):
                rows.append(row)

        if not rows:
            st.warning("No data rows found in the table.")
            return None

        df = pd.DataFrame(rows, columns=valid_headers)

        # Clean numeric columns - FIXED dtype access with hasattr check
        for col in df.columns:
            if col is None or col not in df:
                continue
            series = df[col]
            if hasattr(series, "dtype") and series.dtype == object:
                series = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
                try:
                    df[col] = pd.to_numeric(series, errors="ignore")
                except Exception:
                    pass

        # Normalize positions
        if "Position" in df.columns:
            df["Normalized Position"] = df["Position"].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return None

def plot_player_barchart(player_row, metrics, player_name):
    labels = [metric for metric in metrics if player_row.get(metric) not in ["N/A", None, ""]]

    def clean_value(val):
        if isinstance(val, str) and "%" in val:
            val = val.replace("%", "")
        try:
            return float(val)
        except:
            return 0.0

    values = [clean_value(player_row[metric]) for metric in labels]

    if not labels or not values:
        st.warning("Not enough data to create bar chart.")
        return

    fig, ax = plt.subplots(figsize=(4, 0.3 * len(labels) + 1))
    bars = ax.barh(labels, values, color='tab:blue', alpha=0.8)

    for bar in bars:
        offset = max(bar.get_width() * 0.02, 0.3)
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}",
            va='center',
            fontsize=8,
            color='white'
        )

    ax.set_title(player_name, fontsize=10, color='white', pad=10)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.tick_params(axis='y', labelsize=8, colors='white')
    ax.tick_params(axis='x', labelsize=7, colors='white')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')

    plt.tight_layout()
    st.pyplot(fig)

def get_ai_scouting_report(player_name, player_data):
    prompt = f"""
You are a football scouting expert. Based on the following player data, write a concise, insightful scouting report for {player_name}:

{player_data.to_dict()}

Focus on strengths, weaknesses, and potential.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert football scout."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
        )
        report = response.choices[0].message.content.strip()
        return report
    except Exception as e:
        return f"Error generating AI scouting report: {e}"

# --- Main App ---

# Upload squad and transfer market files
st.sidebar.header("Upload Files")
squad_file = st.sidebar.file_uploader("Upload Squad HTML file", type=["html", "htm"])
transfer_file = st.sidebar.file_uploader("Upload Transfer Market HTML file", type=["html", "htm"])

squad_df = None
transfer_df = None

if squad_file:
    squad_df = parse_html(squad_file)
    if squad_df is not None:
        st.sidebar.success(f"Squad loaded: {len(squad_df)} players")

if transfer_file:
    transfer_df = parse_html(transfer_file)
    if transfer_df is not None:
        st.sidebar.success(f"Transfer market loaded: {len(transfer_df)} players")

# Select dataset to work on
dataset_choice = st.sidebar.selectbox("Select dataset to analyze", options=["Squad", "Transfer Market"])

df = squad_df if dataset_choice == "Squad" else transfer_df

if df is not None:
    st.subheader(f"{dataset_choice} Players Overview")
    st.dataframe(df)

    # Player search
    player_name = st.text_input(f"Search {dataset_choice} player by name")
    if player_name:
        player_rows = df[df["Name"].str.contains(player_name, case=False, na=False)]
        if player_rows.empty:
            st.warning("No players found with that name.")
        else:
            for idx, player_row in player_rows.iterrows():
                st.markdown(f"### {player_row['Name']} ({player_row.get('Club', 'N/A')})")
                st.write(player_row)

                # Radar/bar chart based on position
                pos = player_row.get("Normalized Position", "Unknown")
                metrics = position_metrics.get(pos, position_metrics["Unknown"])

                plot_player_barchart(player_row, metrics, player_row["Name"])

                # AI scouting report button
                if st.button(f"Generate AI scouting report for {player_row['Name']}", key=f"ai_report_{idx}"):
                    with st.spinner("Generating AI scouting report..."):
                        report = get_ai_scouting_report(player_row["Name"], player_row)
                        st.info(report)
else:
    st.info("Please upload squad or transfer market HTML files to begin analysis.")
