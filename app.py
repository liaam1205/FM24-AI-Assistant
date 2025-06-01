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

# --- Metrics for radar charts per position ---
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

# --- HTML parsing function ---
def parse_html(file) -> pd.DataFrame | None:
    try:
        html = file.read().decode("utf-8")
        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

        # Extract headers from thead or first tr
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
        headers = [header_mapping.get(h, None) for h in headers_raw]

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

        # Clean numeric columns
        for col in df.columns:
            if col is None or col not in df:
                continue
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # Normalize positions
        if "Position" in df.columns:
            df["Normalized Position"] = df["Position"].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return None

# --- Plot player bar chart for key metrics ---
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

# --- Generate AI scouting report ---
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

# --- Main UI ---

st.header("Upload Squad and Transfer Files")

col1, col2 = st.columns(2)

with col1:
    squad_file = st.file_uploader("Upload FM24 Squad HTML Export", type=["html", "htm"])

with col2:
    transfer_file = st.file_uploader("Upload FM24 Transfer Market HTML Export", type=["html", "htm"])

squad_df = None
transfer_df = None

if squad_file is not None:
    squad_df = parse_html(squad_file)
    if squad_df is not None:
        st.success(f"Loaded Squad data with {len(squad_df)} players")
        # Show squad preview
        st.dataframe(squad_df.head(10))

if transfer_file is not None:
    transfer_df = parse_html(transfer_file)
    if transfer_df is not None:
        st.success(f"Loaded Transfer Market data with {len(transfer_df)} players")
        # Show transfer preview
        st.dataframe(transfer_df.head(10))

# --- Squad Analysis ---
if squad_df is not None:
    st.header("Squad Player Details and AI Scouting")

    player_names = squad_df["Name"].dropna().unique()
    selected_player = st.selectbox("Select a player from squad", player_names)

    if selected_player:
        player_row = squad_df[squad_df["Name"] == selected_player].iloc[0]
        st.subheader(f"Stats for {selected_player}")
        metrics = position_metrics.get(player_row["Normalized Position"], position_metrics["Unknown"])
        plot_player_barchart(player_row, metrics, selected_player)

        if st.button("Generate AI Scouting Report"):
            with st.spinner("Generating scouting report..."):
                report = get_ai_scouting_report(selected_player, player_row)
            st.markdown(f"**AI Scouting Report:**\n\n{report}")

# --- Transfer Market Search ---
if transfer_df is not None:
    st.header("Transfer Market Player Search & Filter")

    search_name = st.text_input("Search player by name")

    positions = ["All"] + sorted(transfer_df["Normalized Position"].dropna().unique().tolist())
    filter_position = st.selectbox("Filter by position", positions)

    # Age slider limits
    min_age_val = int(transfer_df["Age"].min())
    max_age_val = int(transfer_df["Age"].max())

    min_age, max_age = st.slider(
        "Filter by Age",
        min_value=min_age_val,
        max_value=max_age_val,
        value=(min_age_val, max_age_val),
        step=1,
    )

    # Filter dataframe
    filtered_df = transfer_df.copy()

    if search_name:
        filtered_df = filtered_df[filtered_df["Name"].str.contains(search_name, case=False, na=False)]

    if filter_position != "All":
        filtered_df = filtered_df[filtered_df["Normalized Position"] == filter_position]

    filtered_df = filtered_df[(filtered_df["Age"] >= min_age) & (filtered_df["Age"] <= max_age)]

    st.write(f"Found {len(filtered_df)} players matching filter criteria")
    st.dataframe(filtered_df.reset_index(drop=True))
