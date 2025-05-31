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

# --- Header mapping based on your mapping for consistent column names ---
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

        # Clean numeric columns: remove commas, %, convert to numeric
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

# --- Plot radar (pizza) chart ---
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

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, color='grey', size=10)

    ax.plot(angles, values, color='orange', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='orange', alpha=0.25)

    ax.set_rlabel_position(0)
    plt.yticks([20,40,60,80,100], ["20","40","60","80","100"], color="grey", size=8)
    plt.ylim(0, 100)

    plt.title(title, size=15, color='darkorange', y=1.1)
    st.pyplot(fig)

# --- AI Scouting Report ---
def get_ai_scouting_report(player_name, player_data):
    prompt = f"""
You are a football scouting expert. Based on the following player data, write a concise, insightful scouting report for {player_name}:

{player_data.to_dict()}

Focus on strengths, weaknesses, and potential.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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

# --- UI: File Upload ---
st.sidebar.header("Upload Files")
squad_file = st.sidebar.file_uploader("Upload Squad HTML Export", type=["html", "htm"])
transfer_file = st.sidebar.file_uploader("Upload Transfer Market HTML Export", type=["html", "htm"])

squad_df = parse_html(squad_file) if squad_file else None
transfer_df = parse_html(transfer_file) if transfer_file else None

# --- Main Interface ---
if squad_df is not None:
    st.subheader("Squad Overview")
    st.dataframe(squad_df[["Name", "Club", "Position", "Age", "Current Ability", "Potential Ability"]].sort_values(by="Current Ability", ascending=False))

    player_name = st.selectbox("Select Player for Detailed View", squad_df["Name"].tolist())

    if player_name:
        player_row = squad_df[squad_df["Name"] == player_name].iloc[0]
        st.markdown("### Player Details")
        # Present player details in a table
        player_info = {
            "Name": player_row["Name"],
            "Club": player_row["Club"],
            "Position": player_row["Position"],
            "Age": int(player_row["Age"]) if not pd.isna(player_row["Age"]) else "N/A",
            "Current Ability": int(player_row["Current Ability"]) if not pd.isna(player_row["Current Ability"]) else "N/A",
            "Potential Ability": int(player_row["Potential Ability"]) if not pd.isna(player_row["Potential Ability"]) else "N/A",
            "Transfer Value": player_row["Transfer Value"],
            "Wage": player_row["Wage"],
            "Goals": player_row.get("Goals", "N/A"),
            "Assists": player_row.get("Assists", "N/A"),
            "Expected Goals per 90 Minutes": player_row.get("Expected Goals per 90 Minutes", "N/A"),
            "Expected Goals Overperformance": player_row.get("Expected Goals Overperformance", "N/A"),
            "Expected Assists": player_row.get("Expected Assists", "N/A"),
            "Key Passes": player_row.get("Key Passes", "N/A"),
            "Dribbles Made": player_row.get("Dribbles Made", "N/A"),
            "Pass Completion Ratio": player_row.get("Pass Completion Ratio", "N/A"),
            "Interceptions": player_row.get("Interceptions", "N/A"),
            "Headers Won": player_row.get("Headers Won", "N/A"),
            "Tackle Completion Ratio": player_row.get("Tackle Completion Ratio", "N/A"),
            "Save Ratio": player_row.get("Save Ratio", "N/A"),
            "Clean Sheets": player_row.get("Clean Sheets", "N/A"),
            "Saves Held": player_row.get("Saves Held", "N/A"),
            "Saves Parried": player_row.get("Saves Parried", "N/A"),
            "Saves Tipped": player_row.get("Saves Tipped", "N/A"),
        }
        # Display as table
        info_df = pd.DataFrame.from_dict(player_info, orient="index", columns=["Value"])
        st.table(info_df)

        # Radar Chart
        pos = player_row["Normalized Position"]
        metrics = position_metrics.get(pos, position_metrics["Unknown"])
        plot_player_radar(player_row, metrics, title=f"{player_name} - {pos} Radar")

        # AI Scouting Report
        if st.button("Generate AI Scouting Report"):
            with st.spinner("Generating report..."):
                report = get_ai_scouting_report(player_name, player_row)
            st.markdown("### AI Scouting Report")
            st.write(report)

if transfer_df is not None:
    st.subheader("Transfer Market Overview")
    st.dataframe(transfer_df[["Name", "Club", "Position", "Age", "Current Ability", "Potential Ability"]].sort_values(by="Current Ability", ascending=False))

    transfer_search = st.text_input("Search Transfer Market Players by Name or Club")
    if transfer_search:
        filtered = transfer_df[
            transfer_df["Name"].str.contains(transfer_search, case=False, na=False) |
            transfer_df["Club"].str.contains(transfer_search, case=False, na=False)
        ]
        st.dataframe(filtered)

# --- Footer ---
st.markdown("---")
st.markdown("Created with ❤️ by Your FM24 AI Assistant")
