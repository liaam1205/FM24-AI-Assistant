import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("⚽ Football Manager 2024 Squad & Transfer Analyzer")

st.markdown(
    """
Upload your FM24 exported **Squad** and **Transfer Market** HTML files to analyze player stats,  
view detailed player radar charts tailored to positions, and generate AI scouting reports.
"""
)

# --- OpenAI API Key ---
api_key = st.secrets.get("API_KEY")
if not api_key:
    st.error("⚠️ Please add your OpenAI API key to Streamlit secrets as 'API_KEY'.")
    st.stop()
client = openai.OpenAI(api_key=api_key)

# --- Position aliases & normalization ---
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
    "AM (C)": "Attacking Midfielder",
    "AM (L)": "Inside Forward",
    "AM (R)": "Inside Forward",
    "M (L)": "Wide Midfielder",
    "M (R)": "Wide Midfielder",
    "ST (C)": "Striker",
    "CF": "Complete Forward",
    "WF": "Wide Forward",
    "FW": "Forward",
    # add more if needed
}

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    for key, val in position_aliases.items():
        if key in pos_str:
            return val
    return "Unknown"

# --- Position-specific radar metrics ---
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
    "Complete Forward": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas", "Pas %"],
    "Striker": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas", "Pas %"],
    "Forward": ["Ast", "Gls", "xG/90", "xG-OP", "K Pas", "Pas %"],
    "Wide Forward": ["Ast", "Gls", "Drb", "xG/90", "xG-OP", "K Pas"],
    "Unknown": ["Ast", "Gls", "xG/90", "xG-OP", "xA", "K Pas"],
}

# --- HTML Parsing function ---
def parse_html(file):
    content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    table = soup.find("table")
    if not table:
        st.error("No table found in uploaded HTML.")
        return None

    # Get headers
    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    else:
        first_row = table.find("tr")
        headers = [th.get_text(strip=True) for th in first_row.find_all(["th", "td"])]

    # Get rows
    tbody = table.find("tbody")
    if tbody:
        rows = tbody.find_all("tr")
    else:
        rows = table.find_all("tr")[1:]  # skip header row

    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) != len(headers):
            continue
        data.append([td.get_text(strip=True) for td in cols])

    df = pd.DataFrame(data, columns=headers)

    # Clean columns (remove commas and % and convert to numbers where possible)
    for col in df.columns:
        df[col] = df[col].str.replace(",", "").str.replace("%", "")
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Add normalized position
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

# --- Radar Chart function ---
def plot_radar(player, metrics, title):
    labels = metrics
    values = []
    for m in metrics:
        val = player.get(m)
        if pd.isna(val):
            val = 0
        values.append(float(val))
    values += values[:1]  # close the loop

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#1f77b4', alpha=0.25)
    ax.plot(angles, values, color='#1f77b4', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold', color='#333')

    ax.spines['polar'].set_visible(False)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_title(title, fontsize=16, fontweight='bold', y=1.1)
    plt.tight_layout()
    st.pyplot(fig)

# --- Display Player Details ---
def display_player_details(player):
    st.markdown("### Player Details")
    # Prepare a dataframe with key-value pairs
    display_data = {k: v for k, v in player.items() if pd.notna(v) and k != "Normalized Position"}
    df_disp = pd.DataFrame(display_data.items(), columns=["Attribute", "Value"])
    
    # Formatting percentages nicely
    def fmt(row):
        if row["Attribute"] in ["Pas %", "Sv %"]:
            try:
                return f"{float(row['Value']):.1f}%"
            except:
                return row["Value"]
        return row["Value"]

    df_disp["Value"] = df_disp.apply(fmt, axis=1)
    st.table(df_disp.set_index("Attribute"))

# --- Generate AI scouting report ---
def generate_scouting_report(player_name, player_stats):
    prompt = f"""
You are a football scouting AI. Analyze the following player stats and write a concise scouting report.

Player: {player_name}

Stats:
"""
    for k, v in player_stats.items():
        prompt += f"{k}: {v}\n"
    prompt += "\nWrite a professional scouting report summarizing strengths, weaknesses, and playing style."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        report = response.choices[0].message.content.strip()
        return report
    except Exception as e:
        return f"Error generating report: {e}"

# --- Main App ---

with st.sidebar:
    st.header("Upload Files")
    squad_file = st.file_uploader("Upload Squad HTML export", type=["html"])
    transfer_file = st.file_uploader("Upload Transfer Market HTML export", type=["html"])
    st.markdown("---")
    st.info("Select a player from the squad list below after uploading.")

# Parse files and prepare dataframes
squad_df = parse_html(squad_file) if squad_file else None
transfer_df = parse_html(transfer_file) if transfer_file else None

if squad_df is not None:
    st.success(f"Loaded Squad: {len(squad_df)} players")
    player_names = squad_df["Name"].tolist() if "Name" in squad_df.columns else []

    selected_player_name = st.selectbox("Select a Player", options=player_names)

    if selected_player_name:
        player_row = squad_df[squad_df["Name"] == selected_player_name].iloc[0].to_dict()
        display_player_details(player_row)

        pos = player_row.get("Normalized Position", "Unknown")
        metrics = position_metrics.get(pos, position_metrics["Unknown"])

        # Check we have data for the pizza chart
        if all(m in player_row for m in metrics):
            plot_radar(player_row, metrics, title=f"{selected_player_name} - {pos}")
        else:
            st.warning("Not enough data to generate radar chart for this player.")

        st.markdown("### AI Scouting Report")
        report = generate_scouting_report(selected_player_name, player_row)
        st.write(report)

else:
    st.info("Please upload the Squad HTML file.")

if transfer_df is not None:
    st.success(f"Loaded Transfer Market: {len(transfer_df)} players")
    st.markdown("### Transfer Market Overview")

    # Optional filter by club or position
    filter_club = st.text_input("Filter Transfer Market by Club (optional):").strip()
    filter_pos = st.selectbox("Filter Transfer Market by Position (optional):",
                              options=[""] + sorted(transfer_df["Position"].unique()) if "Position" in transfer_df.columns else [""])

    df_filtered = transfer_df
    if filter_club:
        df_filtered = df_filtered[df_filtered["Club"].str.contains(filter_club, case=False, na=False)]
    if filter_pos:
        if filter_pos != "":
            df_filtered = df_filtered[df_filtered["Position"] == filter_pos]

    st.dataframe(df_filtered.reset_index(drop=True))

else:
    st.info("Please upload the Transfer Market HTML file.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your FM24 AI Assistant")
