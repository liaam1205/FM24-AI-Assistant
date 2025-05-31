import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import openai

# Set your OpenAI API key here or use st.secrets or env var
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

st.set_page_config(page_title="FM24 AI Assistant", layout="wide")

# -- Parsing function --
def parse_html(file) -> pd.DataFrame:
    try:
        soup = BeautifulSoup(file.read(), "html.parser")
        table = soup.find("table")
        if table is None:
            st.error("No table found in the HTML file.")
            return None

        # Find headers robustly
        thead = table.find("thead")
        if thead:
            header_row = thead.find_all("th")
        else:
            # fallback: use first tr as header
            header_row = table.find("tr").find_all("th")

        headers = [th.get_text(strip=True) for th in header_row]

        # Extract rows
        rows = []
        tbody = table.find("tbody")
        if tbody:
            tr_rows = tbody.find_all("tr")
        else:
            # fallback: all trs except first (header)
            tr_rows = table.find_all("tr")[1:]

        for tr in tr_rows:
            cells = tr.find_all(["td", "th"])
            row = [cell.get_text(strip=True) for cell in cells]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)
        return df

    except Exception as e:
        st.error(f"Error parsing HTML file: {e}")
        return None

# -- Data cleaning and normalization --
def clean_squad_df(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns for consistency
    df.rename(columns={
        "Name": "Name",
        "Position": "Position",
        "Age": "Age",
        "CA": "CA",
        "PA": "PA",
        "Transfer Value": "Transfer Value",
        "Wage": "Wage",
        "Ast": "Assists",
        "Gls": "Goals",
        "xG/90": "xG/90",
        "xG-OP": "xG-OP",
        "xA": "xA",
        "K Pas": "Key Passes",
        "Drb": "Dribbles",
        "Pas %": "Pass Completion %",
        "Itc": "Interceptions",
        "Hdrs": "Headers",
        "Tck R": "Tackle %",
        "Sv %": "Save %",
        "Clean Sheets": "Clean Sheets",
        "Svh": "Saves Held",
        "Svp": "Saves Parried",
        "Sv": "Saves Tipped"
    }, inplace=True)

    # Remove columns not needed
    drop_cols = [col for col in df.columns if col in ["Potential", "Ability", "Rec"]]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Convert numeric columns
    numeric_cols = ["Age", "CA", "PA", "Transfer Value", "Wage", "Assists", "Goals",
                    "xG/90", "xG-OP", "xA", "Key Passes", "Dribbles", "Pass Completion %",
                    "Interceptions", "Headers", "Tackle %", "Save %", "Clean Sheets",
                    "Saves Held", "Saves Parried", "Saves Tipped"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace('[£,]', '', regex=True), errors='coerce')

    # Normalize position to categories for radar chart selection
    def normalize_position(pos):
        pos = pos.lower()
        if any(x in pos for x in ["gk", "goalkeeper"]):
            return "GK"
        elif any(x in pos for x in ["cb", "centre back", "center back", "defender"]):
            return "DEF"
        elif any(x in pos for x in ["wb", "wing back", "full back"]):
            return "DEF"
        elif any(x in pos for x in ["dm", "defensive mid"]):
            return "MID"
        elif any(x in pos for x in ["cm", "centre mid", "center mid", "midfield"]):
            return "MID"
        elif any(x in pos for x in ["am", "attacking mid", "winger"]):
            return "MID"
        elif any(x in pos for x in ["fw", "forward", "striker", "attacker"]):
            return "FWD"
        else:
            return "Unknown"

    df["Normalized Position"] = df["Position"].apply(normalize_position)
    df.dropna(subset=["Name"], inplace=True)  # drop rows with no name
    return df

# -- Player details display --
def display_player_details(df: pd.DataFrame, player_name: str):
    player_rows = df[df["Name"] == player_name]
    if player_rows.empty:
        st.warning("Player not found.")
        return

    player_data = player_rows.iloc[0].to_dict()

    # Display player info in a table format for neatness
    info_fields = {
        "Name": player_data.get("Name", ""),
        "Position": player_data.get("Position", ""),
        "Age": player_data.get("Age", ""),
        "Current Ability (CA)": player_data.get("CA", ""),
        "Potential Ability (PA)": player_data.get("PA", ""),
        "Transfer Value (£)": player_data.get("Transfer Value", ""),
        "Wage (£)": player_data.get("Wage", "")
    }
    info_df = pd.DataFrame.from_dict(info_fields, orient='index', columns=["Value"])
    st.markdown("### Player Details")
    st.table(info_df)

    # Stats table
    stats_keys = ["Assists", "Goals", "xG/90", "xG-OP", "xA", "Key Passes", "Dribbles",
                  "Pass Completion %", "Interceptions", "Headers", "Tackle %",
                  "Clean Sheets", "Saves Held", "Saves Parried", "Saves Tipped", "Save %"]
    stats = {k: player_data.get(k, None) for k in stats_keys if k in df.columns}
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=["Value"])
    st.markdown("### Player Stats")
    st.table(stats_df)

    return player_data

# -- Radar chart based on position --
position_metrics = {
    "GK": ["Save %", "Clean Sheets", "Saves Held", "Saves Parried", "Saves Tipped"],
    "DEF": ["Tackle %", "Interceptions", "Headers", "Pass Completion %", "Dribbles"],
    "MID": ["Key Passes", "Assists", "Dribbles", "Pass Completion %", "Tackle %"],
    "FWD": ["Goals", "xG/90", "xG-OP", "Assists", "Dribbles"],
    "Unknown": ["Goals", "Assists", "Key Passes", "Dribbles", "Pass Completion %"]
}

def plot_radar_chart(player_data):
    position = player_data.get("Normalized Position", "Unknown")
    metrics = position_metrics.get(position, position_metrics["Unknown"])

    # Filter available metrics only
    available_metrics = []
    values = []
    for metric in metrics:
        val = player_data.get(metric)
        if val is not None and not pd.isna(val):
            available_metrics.append(metric)
            values.append(float(val))
    if len(available_metrics) < 3:
        st.info("Not enough data to display radar chart.")
        return

    # Radar plot setup
    num_vars = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]  # Close the loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="tab:blue", linewidth=2)
    ax.fill(angles, values, color="tab:blue", alpha=0.25)
    ax.set_yticklabels([])  # Hide radial labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, fontsize=12)

    ax.set_title(f"{player_data.get('Name', 'Player')} - {position} Radar Chart", fontsize=14, pad=20)
    st.pyplot(fig)

# -- AI scouting report generation --
def generate_ai_scouting_report(player_data: dict) -> str:
    # Create prompt for OpenAI
    prompt = f"""
You are a football scout. Based on the following player data, provide a detailed scouting report:

Name: {player_data.get('Name', 'Unknown')}
Position: {player_data.get('Position', 'Unknown')}
Age: {player_data.get('Age', 'Unknown')}
Current Ability: {player_data.get('CA', 'Unknown')}
Potential Ability: {player_data.get('PA', 'Unknown')}
Stats:
"""
    for key in position_metrics.get(player_data.get("Normalized Position", "Unknown"), []):
        prompt += f"- {key}: {player_data.get(key, 'N/A')}\n"

    prompt += "\nProvide strengths, weaknesses, and playing style."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7,
        )
        report = response.choices[0].message.content.strip()
        return report
    except Exception as e:
        return f"Error generating AI scouting report: {e}"

# -- Streamlit app layout --
def main():
    st.title("⚽ FM24 AI Assistant")

    st.sidebar.header("Upload Files")

    squad_file = st.sidebar.file_uploader("Upload Squad HTML Export", type=["html", "htm"])
    transfer_file = st.sidebar.file_uploader("Upload Transfer HTML Export (optional)", type=["html", "htm"])

    squad_df = None
    transfer_df = None

    if squad_file:
        squad_df = parse_html(squad_file)
        if squad_df is not None:
            squad_df = clean_squad_df(squad_df)
            st.sidebar.success(f"Squad data loaded: {len(squad_df)} players")

    if transfer_file:
        transfer_df = parse_html(transfer_file)
        if transfer_df is not None:
            st.sidebar.success(f"Transfer data loaded: {len(transfer_df)} entries")

    if squad_df is not None:
        player_names = squad_df["Name"].dropna().unique()
        selected_player = st.selectbox("Select Player", player_names)

        if selected_player:
            player_data = display_player_details(squad_df, selected_player)
            if player_data:
                plot_radar_chart(player_data)

                with st.expander("AI Scouting Report"):
                    if st.button("Generate AI Report"):
                        with st.spinner("Generating AI scouting report..."):
                            report = generate_ai_scouting_report(player_data)
                            st.write(report)

    else:
        st.info("Please upload a squad HTML export file to begin.")

if __name__ == "__main__":
    main()
