import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import re

# --- Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("⚽ Football Manager 2024 Squad & Transfer Analyzer")

st.markdown("""
Upload your FM24 **squad** and **transfer market** HTML export files.  
View player details, radar charts, and get AI-generated scouting reports!  
Search the transfer market by player name.
""")

# --- OpenAI API Key ---
if "API_KEY" not in st.secrets:
    st.error("Set your OpenAI API key in Streamlit secrets as 'API_KEY'.")
    st.stop()
openai.api_key = st.secrets["API_KEY"]

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
    pos_list = [p.strip() for p in pos_str.split(",")]
    for pos in pos_list:
        pos_clean = pos.upper().replace(" ", "")
        for alias in position_aliases:
            if alias.replace(" ", "") == pos_clean:
                return position_aliases[alias]
    return "Unknown"

# --- Radar chart metrics per position ---
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
    "Unknown": ["Ast", "Gls", "xG/90", "xG-OP", "xA", "K Pas"]
}

# --- Parse FM24 HTML export ---
def parse_html(html_file):
    try:
        content = html_file.read().decode("utf-8")
        soup = BeautifulSoup(content, "html.parser")
        table = soup.find("table")
        if not table:
            st.error("No table found in the HTML file.")
            return None

        # Headers
        thead = table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        else:
            first_row = table.find("tr")
            headers = [th.get_text(strip=True) for th in first_row.find_all(["th","td"])]

        # Rows
        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
        else:
            rows = table.find_all("tr")[1:]

        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) == len(headers):
                data.append([col.get_text(strip=True) for col in cols])

        df = pd.DataFrame(data, columns=headers)

        # Clean numeric columns: remove % and commas
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace("%", "", regex=False).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Normalize position column
        pos_col = None
        for candidate in ["Position", "Pos"]:
            if candidate in df.columns:
                pos_col = candidate
                break
        if pos_col:
            df["Normalized Position"] = df[pos_col].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"Error parsing HTML file: {e}")
        return None

# --- Radar chart ---
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

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, fontsize=10)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=8)
    max_val = max(values) if max(values) > 0 else 1
    plt.ylim(0, max_val * 1.1)

    ax.plot(angles, values, color="tab:orange", linewidth=3, linestyle='solid')
    ax.fill(angles, values, color="tab:orange", alpha=0.25)
    ax.set_title(title, fontsize=14, y=1.1)

    st.pyplot(fig)

# --- Display player details ---
def display_player_details(player_data):
    exclude_keys = ["Normalized Position"]
    display_keys = [k for k in player_data.keys() if k not in exclude_keys and player_data[k] not in [None, np.nan]]

    formatted_data = {}
    for k in display_keys:
        new_key = re.sub(r"([a-z])([A-Z])", r"\1 \2", k)
        new_key = new_key.replace("_", " ").title()
        val = player_data[k]
        if isinstance(val, float):
            val = round(val, 2)
        formatted_data[new_key] = val

    df_display = pd.DataFrame(formatted_data.items(), columns=["Attribute", "Value"])
    st.table(df_display)

# --- AI scouting report generation ---
def generate_ai_scout_report(player_name, player_data):
    prompt = f"Provide a detailed Football Manager 2024 scouting report for player {player_name}.\n"
    prompt += "Player stats:\n"
    for k, v in player_data.items():
        if k != "Normalized Position" and v not in [None, "", np.nan]:
            prompt += f"- {k}: {v}\n"
    prompt += "\nInclude strengths, weaknesses, and playing style."

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating AI scouting report: {e}"

# --- Main app ---

squad_file = st.file_uploader("Upload FM24 Squad HTML Export", type=["html"])
transfer_file = st.file_uploader("Upload FM24 Transfer Market HTML Export", type=["html"])

if squad_file:
    squad_df = parse_html(squad_file)
    if squad_df is not None:
        st.success("Squad data loaded successfully.")
        st.dataframe(squad_df.head(10))

        player_name = st.selectbox("Select a player to view details", options=squad_df["Name"].unique())

        if player_name:
            player_row = squad_df[squad_df["Name"] == player_name].iloc[0].to_dict()
            st.subheader(f"Player Details: {player_name}")
            display_player_details(player_row)

            pos = player_row.get("Normalized Position", "Unknown")
            metrics = position_metrics.get(pos, position_metrics["Unknown"])

            st.subheader("Player Radar Chart")
            plot_player_radar(player_row, metrics, title=f"{player_name} - {pos}")

            st.subheader("AI Scouting Report")
            if st.button("Generate Scouting Report"):
                with st.spinner("Generating AI scouting report..."):
                    report = generate_ai_scout_report(player_name, player_row)
                st.write(report)

if transfer_file:
    transfer_df = parse_html(transfer_file)
    if transfer_df is not None:
        st.success("Transfer market data loaded successfully.")
        st.dataframe(transfer_df.head(10))

        search_name = st.text_input("Search Transfer Market by Player Name")
        if search_name:
            filtered = transfer_df[transfer_df["Name"].str.contains(search_name, case=False, na=False)]
            st.dataframe(filtered)
