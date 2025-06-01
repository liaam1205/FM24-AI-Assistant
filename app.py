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

    # Clean numeric columns
        for col in df.columns:
            if col is None or col not in df:
                continue  # skip unknown or invalid columns

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

def plot_player_pizza(player_data, metrics, player_name,
                      player_color='mediumseagreen'):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = metrics
    values = []

    for m in metrics:
        val = player_data.get(m, 0)
        if isinstance(val, str):
            val = val.replace(",", "").replace("%", "")
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            val_float = 0.0
        values.append(val_float)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    fig, ax = plt.subplots(figsize=(1.5, 1.5), subplot_kw=dict(polar=True), facecolor='none')

    max_val = max(values) if max(values) > 0 else 1
    normalized = [v / max_val * 100 for v in values]

    width = 2 * np.pi / N * 0.7
    bar_bottom = 0

    bars = ax.bar(
        angles,
        normalized,
        width=width,
        bottom=bar_bottom,
        color=player_color,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.8,
        align='center'
    )

    grid_vals = [20, 40, 60, 80, 100]
    ax.set_yticks(grid_vals)
    ax.set_yticklabels([str(g) for g in grid_vals], fontsize=5, color='gray')
    ax.yaxis.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 110)

    ax.set_xticks([])

    label_radius = 110
    value_radius = 85  # increased gap from label to value (was 95 before)

    for angle, label, val in zip(angles, labels, values):
        rotation = np.rad2deg(angle)
        align = 'left'
        rotation_text = rotation
        if rotation > 90 and rotation < 270:
            rotation_text += 180
            align = 'right'

        ax.text(
            angle,
            label_radius,
            label,
            ha=align,
            va='center',
            rotation=rotation_text,
            rotation_mode='anchor',
            fontsize=6,
            fontweight='bold',
            color='black'
        )
        ax.text(
            angle,
            value_radius,
            f"{val:.1f}",
            ha=align,
            va='center',
            rotation=rotation_text,
            rotation_mode='anchor',
            fontsize=5,
            color='dimgray'
        )

    ax.spines['polar'].set_visible(False)

    plt.title(player_name, y=1.08, fontsize=8, fontweight='bold', color='black')

    st.pyplot(fig, transparent=True)
                          
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

# --- Main Interface for Squad ---
if squad_df is not None:
    st.subheader("Squad Overview")
    st.dataframe(
        squad_df[["Name", "Club", "Position", "Age", "Current Ability", "Potential Ability"]]
        .sort_values(by="Current Ability", ascending=False)
    )

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
            "Transfer Value": player_row.get("Transfer Value", "N/A"),
            "Wage": player_row.get("Wage", "N/A"),
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
        info_df = pd.DataFrame.from_dict(player_info, orient="index", columns=["Value"])
        st.table(info_df)

        # Pizza Chart
        pos = player_row["Normalized Position"]
        metrics = position_metrics.get(pos, position_metrics["Unknown"])
        plot_player_pizza(player_row, metrics, title=f"{selected_player} - {pos} Pizza Chart")

        # AI Scouting Report
        if st.button("Generate AI Scouting Report"):
            with st.spinner("Generating report..."):
                report = get_ai_scouting_report(player_name, player_row)
            st.markdown("### AI Scouting Report")
            st.write(report)

# --- Main Interface for Transfer Market ---
if transfer_df is not None and not transfer_df.empty:
    st.subheader("Transfer Market Overview")

    # Show full sorted transfer market
    filtered = transfer_df[
        ["Name", "Club", "Position", "Age", "Current Ability", "Potential Ability"]
    ].sort_values(by="Current Ability", ascending=False)

    st.dataframe(filtered)

    if not filtered.empty:
        player_names = filtered["Name"].unique().tolist()
        selected_player = st.selectbox("Select a player to view details", player_names)

        if selected_player:
            player_row = transfer_df[transfer_df["Name"] == selected_player].iloc[0]

            st.markdown(f"### Player Details: {player_row['Name']}")
            st.write(f"**Club:** {player_row['Club']}")
            st.write(f"**Position:** {player_row['Position']}")
            st.write(f"**Age:** {player_row['Age']}")
            st.write(f"**Current Ability:** {player_row['Current Ability']}")
            st.write(f"**Potential Ability:** {player_row['Potential Ability']}")

            # Pizza Chart for Transfer Market Player
            pos = player_row.get("Normalized Position", "Unknown")
            metrics = position_metrics.get(pos, position_metrics["Unknown"])
            st.markdown("#### Performance Overview (Pizza Chart)")
            plot_player_pizza(player_data, metrics, player_name, player_color='mediumseagreen'):

            # AI Scout Report
            if st.button("Generate AI Scout Report for Transfer Player"):
                with st.spinner("Generating report..."):
                    report = get_ai_scouting_report(selected_player, player_row)
                st.markdown("#### AI Scout Report")
                st.markdown(report)

else:
    st.warning("No transfer data available.")
