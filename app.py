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
            try:
                if col is None or col not in df:
                    continue  # skip unknown or invalid columns
                series = df[col]
                if series.dtype == object:
                    series = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
                    df[col] = pd.to_numeric(series, errors="coerce")
            except Exception as conv_error:
                st.warning(f"Could not convert column '{col}' to numeric: {conv_error}")

        # Normalize positions
        if "Position" in df.columns:
            df["Normalized Position"] = df["Position"].apply(normalize_position)
        else:
            df["Normalized Position"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        return None

# --- Plot bar chart for player stats ---
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

    ax.set_title(player_name, fontsize=10, color='black', pad=10)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.tick_params(axis='y', labelsize=8, colors='black')
    ax.tick_params(axis='x', labelsize=7, colors='black')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')

    st.pyplot(fig)

# --- AI scouting report ---
def get_ai_report(player_name, squad_df):
    prompt = f"""
You are a Football Manager AI scout. Based on the following squad data, provide a scouting report on the player named '{player_name}'.
Squad data columns: {list(squad_df.columns)}.

Player data:
{ squad_df[squad_df['Name'] == player_name].to_dict(orient='records')[0] if player_name in squad_df['Name'].values else 'No data available.' }

Please give a summary including strengths, weaknesses, and potential.
"""
    response = client.chat.completions.create(
        model="gpt 3.45 turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# --- Sidebar: Upload Files ---
st.sidebar.header("Upload your FM24 data")
squad_file = st.sidebar.file_uploader("Upload Squad HTML file", type=["html", "htm"])
transfer_file = st.sidebar.file_uploader("Upload Transfer Market HTML file", type=["html", "htm"])

squad_df = None
transfer_df = None

if squad_file:
    squad_df = parse_html(squad_file)
    if squad_df is not None:
        st.sidebar.success(f"Squad file loaded: {len(squad_df)} players")

if transfer_file:
    transfer_df = parse_html(transfer_file)
    if transfer_df is not None:
        st.sidebar.success(f"Transfer market file loaded: {len(transfer_df)} players")

# --- Main layout ---
if squad_df is None and transfer_df is None:
    st.info("Upload at least one file to get started.")
    st.stop()

# --- Squad analysis ---
if squad_df is not None:
    st.header("Your Squad Overview")

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
       # Drop NaNs before calculating min and max
valid_ages = transfer_df["Age"].dropna()

if not valid_ages.empty:
    age_min = int(valid_ages.min())
    age_max = int(valid_ages.max())
    age_filter_tr = st.slider("Age Filter", min_value=age_min, max_value=age_max, value=(age_min, age_max))
else:
    st.warning("No valid age data available for filtering.")
    age_filter_tr = (18, 35)  # or whatever sensible default
    with col2:
        ability_filter = st.slider("Filter by Potential Ability", min_value=int(squad_df['Potential'].min()), max_value=int(squad_df['Potential'].max()), value=(int(squad_df['Potential'].min()), int(squad_df['Potential'].max())))
    with col3:
        positions = ["All"] + sorted(squad_df["Normalized Position"].dropna().unique().tolist())
        selected_position = st.selectbox("Filter by Position", positions)

    filtered_squad = squad_df[
        (squad_df['Age'] >= age_filter[0]) &
        (squad_df['Age'] <= age_filter[1]) &
        (squad_df['Potential'] >= ability_filter[0]) &
        (squad_df['Potential'] <= ability_filter[1])
    ]

    if selected_position != "All":
        filtered_squad = filtered_squad[filtered_squad["Normalized Position"] == selected_position]

    st.write(f"Players matching filters: {len(filtered_squad)}")

    player_names = filtered_squad["Name"].tolist()
    selected_player = st.selectbox("Select a player to view details:", player_names)

    if selected_player:
        player_row = filtered_squad[filtered_squad["Name"] == selected_player].iloc[0]

        st.subheader(f"Player: {selected_player}")

        # Basic info
        cols = st.columns(3)
        cols[0].write(f"**Age:** {player_row['Age']}")
        cols[1].write(f"**Position:** {player_row['Position']}")
        cols[2].write(f"**Potential:** {player_row['Potential']}")

        # Radar / bar chart metrics
        pos = player_row["Normalized Position"] if "Normalized Position" in player_row else "Unknown"
        metrics = position_metrics.get(pos, position_metrics["Unknown"])
        st.markdown("### Key Stats")
        plot_player_barchart(player_row, metrics, selected_player)

        # AI scouting report button
        if st.button("Get AI Scouting Report", key="squad_ai_report"):
            with st.spinner("Generating AI report..."):
                report = get_ai_report(selected_player, filtered_squad)
                st.markdown(f"**AI Report:**\n\n{report}")

# --- Transfer market search ---
if transfer_df is not None:
    st.header("Transfer Market")

    # Filters for transfer market
    tr_col1, tr_col2, tr_col3 = st.columns(3)
    with tr_col1:
        age_filter_tr = st.slider("Age Filter", min_value=int(transfer_df['Age'].min()), max_value=int(transfer_df['Age'].max()), value=(int(transfer_df['Age'].min()), int(transfer_df['Age'].max())))
    with tr_col2:
        value_filter = st.slider("Max Transfer Value (in thousands)", min_value=0, max_value=int(transfer_df['Transfer Value'].max()), value=int(transfer_df['Transfer Value'].max()))
    with tr_col3:
        positions_tr = ["All"] + sorted(transfer_df["Normalized Position"].dropna().unique().tolist())
        selected_position_tr = st.selectbox("Position Filter", positions_tr, key="transfer_pos_filter")

    filtered_transfer = transfer_df[
        (transfer_df['Age'] >= age_filter_tr[0]) &
        (transfer_df['Age'] <= age_filter_tr[1]) &
        (transfer_df['Transfer Value'] <= value_filter)
    ]

    if selected_position_tr != "All":
        filtered_transfer = filtered_transfer[filtered_transfer["Normalized Position"] == selected_position_tr]

    st.write(f"Players available: {len(filtered_transfer)}")

    # Show top 10 transfer players sorted by Potential Ability desc
    top_transfer = filtered_transfer.sort_values(by="Potential", ascending=False).head(10)

    for idx, row in top_transfer.iterrows():
        st.markdown(f"### {row['Name']} ({row['Club']})")
        cols = st.columns(3)
        cols[0].write(f"Age: {row['Age']}")
        cols[1].write(f"Position: {row['Position']}")
        cols[2].write(f"Potential: {row['Potential']}")
        st.write(f"Transfer Value: {row['Transfer Value']:,} | Wage: {row['Wage']:,}")

        # Show bar chart for transfer player
        pos_tr = row["Normalized Position"] if "Normalized Position" in row else "Unknown"
        metrics_tr = position_metrics.get(pos_tr, position_metrics["Unknown"])
        plot_player_barchart(row, metrics_tr, row['Name'])

        # Optional: AI report button (commented to avoid too many calls)
        if st.button(f"Get AI Report for {row['Name']}", key=f"transfer_ai_report_{idx}_{row['Name']}"):
            with st.spinner("Generating AI report..."):
                report = get_ai_report(row['Name'], filtered_transfer)
                st.markdown(f"**AI Report:**\n\n{report}")
