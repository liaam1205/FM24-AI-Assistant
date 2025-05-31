import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np

# --- App Setup ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("⚽ Football Manager 2024 Squad & Transfer Analyzer")
st.markdown(
    """
Upload your FM24 exported **Squad** and **Transfer Market** HTML files to analyze players,  
view radar charts tailored by position, and generate AI-powered scouting reports!  
"""
)

# --- OpenAI API Key ---
try:
    openai.api_key = st.secrets["API_KEY"]
except Exception:
    openai.api_key = None

# --- Position aliases and normalization ---
position_aliases = {
    "GK": "Goalkeeper",
    "D(C)": "Centre Back",
    "DC": "Centre Back",
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

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    positions = [p.strip().upper().replace(" ", "").replace(" ", "") for p in pos_str.split(",")]
    for pos in positions:
        if pos in position_aliases:
            return position_aliases[pos]
    return "Unknown"

# --- Metrics per position for radar chart ---
position_metrics = {
    "Goalkeeper": ["Pas %", "Sv %", "Clean Sheets", "Svh", "Svp", "Sv"],
    "Centre Back": ["Ast", "Gls", "Hdrs", "Tck R", "Itc", "Pas %"],
    "Fullback": ["Ast", "Gls", "Drb", "Tck R", "Itc", "Pas %"],
    "Wingback": ["Ast", "Gls", "Drb", "Tck R", "Itc", "Pas %"],
    "Defensive Midfielder": ["Ast", "Gls", "Tck R", "Itc", "Pas %", "K Pas"],
    "Central Midfielder": ["Ast", "Gls", "K Pas", "Drb", "Pas %", "Itc"],
    "Attacking Midfielder": ["Ast", "Gls", "xG/90", "xG-OP", "xA", "K Pas"],
    "Wide Midfielder": ["Ast", "Gls", "Drb", "K Pas", "Pas %", "xG/90"],
    "Inside Forward": ["Ast", "Gls", "Drb", "xG/90", "xG-OP", "K Pas"],
    "Complete Forward": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas"],
    "Striker": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas"],
    "Forward": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas"],
    "Wide Forward": ["Ast", "Gls", "Drb", "xG/90", "xG-OP", "K Pas"],
    "Unknown": ["Ast", "Gls", "xG/90", "xG-OP", "xA", "K Pas"],
}

# --- Parse FM24 HTML export into DataFrame ---
def parse_html(file) -> pd.DataFrame | None:
    try:
        soup = BeautifulSoup(file, 'html.parser')
        table = soup.find("table")
        if not table:
            st.error("No table found in the uploaded HTML file.")
            return None

        # Get headers
        header_row = table.find("thead")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
        else:
            first_tr = table.find("tr")
            headers = [th.get_text(strip=True) for th in first_tr.find_all("th")]
            if not headers:
                headers = [td.get_text(strip=True) for td in first_tr.find_all("td")]

        # Parse rows
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            row = [td.get_text(strip=True) for td in cells]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)

        # Clean numeric columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(",", "").str.replace("%", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors='ignore')

        # Normalize positions
        pos_col = None
        for colname in ["Position", "Pos"]:
            if colname in df.columns:
                pos_col = colname
                break
        if pos_col:
            df["Normalized Position"] = df[pos_col].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df
    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return None

# --- Plot radar chart ---
def plot_player_radar(player_data: dict, metrics: list[str], title="Player Radar Chart"):
    labels = metrics
    values = []
    for m in metrics:
        val = player_data.get(m)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        values.append(float(val))

    values += values[:1]  # close the radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#1f77b4', alpha=0.25)
    ax.plot(angles, values, color='#1f77b4', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color='#333333')

    ax.spines['polar'].set_visible(False)
    ax.grid(color='#cccccc', linestyle='--', linewidth=0.5)
    ax.set_title(title, y=1.1, fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

# --- Display player details in table ---
def display_player_details(df: pd.DataFrame, player_name: str):
    player_rows = df[df["Name"] == player_name]
    if player_rows.empty:
        st.warning("Player not found.")
        return
    player_data = player_rows.iloc[0].to_dict()

    st.subheader(f"{player_name} — {player_data.get('Normalized Position', 'Unknown')}")

    # Exclude some columns for display clarity
    exclude_cols = ["Normalized Position", "Rec", "Potential", "Ability", "Name", "Position", "Pos"]
    stats = {col: player_data.get(col, "") for col in df.columns if col not in exclude_cols}

    stats_df = pd.DataFrame(stats.items(), columns=["Stat", "Value"])
    stats_df["Value"] = stats_df["Value"].astype(str)

    # Display as a table for neatness
    st.table(stats_df)

    # Radar chart
    position = player_data.get("Normalized Position", "Unknown")
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    valid_metrics = [m for m in metrics if m in player_data and pd.notna(player_data[m])]

    if len(valid_metrics) < 3:
        st.info("Not enough data to display radar chart.")
        return

    plot_player_radar(player_data, valid_metrics, title=f"{player_name} — {position}")

# --- AI scouting report ---
def generate_ai_scouting_report(player_data: dict) -> str:
    if not openai.api_key:
        return "OpenAI API key not configured."

    prompt = (
        f"Write a detailed football scouting report for the player based on these stats:\n\n"
        f"Name: {player_data.get('Name', 'Unknown')}\n"
        f"Position: {player_data.get('Normalized Position', 'Unknown')}\n"
        f"Stats:\n"
    )
    for stat, val in player_data.items():
        if stat in ["Name", "Normalized Position"] or val == "" or pd.isna(val):
            continue
        prompt += f"- {stat}: {val}\n"

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            n=1,
            stop=None,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating AI report: {e}"

# --- UI ---
with st.sidebar:
    st.header("Upload your FM24 files")
    squad_file = st.file_uploader("Upload Squad HTML Export", type=["html"])
    transfer_file = st.file_uploader("Upload Transfer Market HTML Export", type=["html"])

if squad_file:
    squad_df = parse_html(squad_file)
else:
    squad_df = None

if transfer_file:
    transfer_df = parse_html(transfer_file)
else:
    transfer_df = None

if squad_df is not None:
    st.header("Squad Players")
    player_names = squad_df["Name"].tolist()
    selected_player = st.selectbox("Select Player", player_names)

    if selected_player:
        display_player_details(squad_df, selected_player)

        with st.expander("AI Scouting Report"):
            if st.button("Generate AI Report"):
                player_data = squad_df[squad_df["Name"] == selected_player].iloc[0].to_dict()
                with st.spinner("Generating scouting report..."):
                    report = generate_ai_scouting_report(player_data)
                    st.write(report)

if transfer_df is not None:
    st.header("Transfer Market Players")
    st.dataframe(transfer_df)
