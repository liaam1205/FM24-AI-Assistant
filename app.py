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

# --- Header translation from HTML export headers to normalized headers ---
header_mapping = {
    "Name": "Name",
    "Club": "Club",
    "Position": "Position",
    "Age": "Age",
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
    "Hdrs": "Headers",
    "Tck R": "Tackle Completion Ratio",
    "Sv %": "Save Ratio",
    "Clean Sheets": "Clean Sheets",
    "Svh": "Saves Held",
    "Svp": "Saves Parried",
    "Sv": "Saves Tipped",
    "Conversion %": "Conversion %",  # If available
}

# --- Parse HTML file to DataFrame ---
def parse_html(file) -> pd.DataFrame | None:
    try:
        html = file.read().decode("utf-8")
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

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
                continue
            row = [cells[i].get_text(strip=True) for i in valid_cols_idx if i < len(cells)]
            if len(row) == len(valid_headers):
                rows.append(row)

        if not rows:
            st.warning("No data rows found in the table.")
            return None

        df = pd.DataFrame(rows, columns=valid_headers)

        # Clean numeric columns
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").str.replace("%", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Position" in df.columns:
            df["Normalized Position"] = df["Position"].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return None

# --- Plot radar (pizza) chart for player ---
def plot_player_radar(player_data: dict, metrics: list[str], title="Player Radar Chart"):
    labels = metrics
    values = []
    for m in metrics:
        val = player_data.get(m)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        values.append(float(val))

    values += values[:1]  # close the circle

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, color='grey', size=10)
    ax.plot(angles, values, color='red', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='red', alpha=0.25)

    # Remove ytick labels and set grid
    ax.set_yticklabels([])
    ax.set_ylim(0, max(10, max(values)))  # dynamic max scale

    plt.title(title, size=14, color='black', y=1.1)
    st.pyplot(fig)

# --- Show player details in a clean table ---
def display_player_details(player: pd.Series):
    details = {
        "Name": player.get("Name", ""),
        "Club": player.get("Club", ""),
        "Age": player.get("Age", ""),
        "Position": player.get("Position", ""),
        "Current Ability": player.get("Current Ability", ""),
        "Potential Ability": player.get("Potential Ability", ""),
        "Transfer Value": player.get("Transfer Value", ""),
        "Wage": player.get("Wage", ""),
        "Goals": player.get("Goals", ""),
        "Assists": player.get("Assists", ""),
    }
    # Display in a two-column table
    details_df = pd.DataFrame.from_dict(details, orient='index', columns=['Value'])
    st.table(details_df)

# --- AI Scouting report placeholder ---
def generate_ai_report(player_name: str):
    # NOTE: For now, just a placeholder static message
    # Integrate with OpenAI ChatCompletion API as needed
    prompt = f"Write a detailed scouting report for the football player {player_name} based on FM24 data."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300
        )
        report = response.choices[0].message.content
    except Exception as e:
        report = f"AI report generation error: {e}"
    return report

# --- Main ---
def main():
    st.sidebar.header("Upload Your Files")
    squad_file = st.sidebar.file_uploader("Upload Squad HTML Export", type=["html", "htm"])
    transfer_file = st.sidebar.file_uploader("Upload Transfer Market HTML Export", type=["html", "htm"])

    squad_df = None
    transfer_df = None

    if squad_file:
        squad_df = parse_html(squad_file)
    if transfer_file:
        transfer_df = parse_html(transfer_file)

    if squad_df is not None:
        st.header("📋 Squad Overview")
        st.dataframe(squad_df)

        player_name_list = squad_df["Name"].dropna().unique()
        selected_player = st.selectbox("Select a player to analyze:", player_name_list)

        if selected_player:
            player = squad_df[squad_df["Name"] == selected_player].iloc[0]

            st.subheader("Player Details")
            display_player_details(player)

            st.subheader("Player Radar Chart")

            pos = player.get("Normalized Position", "Unknown")
            metrics = position_metrics.get(pos, position_metrics["Unknown"])

            # Ensure metrics exist in DataFrame columns
            available_metrics = [m for m in metrics if m in squad_df.columns]

            if len(available_metrics) < 3:
                st.warning("Not enough data to plot radar chart for this player.")
            else:
                player_data = player.to_dict()
                plot_player_radar(player_data, available_metrics, title=f"{selected_player} - {pos}")

            st.subheader("AI Scouting Report")
            with st.spinner("Generating AI scouting report..."):
                report = generate_ai_report(selected_player)
            st.write(report)

    if transfer_df is not None:
        st.header("🔎 Transfer Market Players")
        st.dataframe(transfer_df)

        transfer_search = st.text_input("Search Transfer Market by Player Name")
        if transfer_search:
            filtered_transfers = transfer_df[transfer_df["Name"].str.contains(transfer_search, case=False, na=False)]
            st.dataframe(filtered_transfers)

if __name__ == "__main__":
    main()
