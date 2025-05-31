import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- App Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Market Analyzer", layout="wide")
st.title("📊 FM24 Squad & Transfer Market Analyzer")

# --- API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- Upload Files ---
st.sidebar.header("📁 Upload Files")
squad_file = st.sidebar.file_uploader("Upload Squad Export (.html)", type=["html"], key="squad")
market_file = st.sidebar.file_uploader("Upload Transfer Market Export (.html)", type=["html"], key="market")

# --- HTML Parser ---
def parse_html_to_df(file):
    soup = BeautifulSoup(file, "html.parser")
    table = soup.find("table")

    if not table:
        st.warning("⚠️ No table found in the uploaded HTML.")
        return pd.DataFrame()

    headers = [th.get_text(strip=True).replace("\xa0", " ") for th in table.find_all("th")]

    seen = {}
    unique_headers = []
    for col in headers:
        if col in seen:
            seen[col] += 1
            unique_headers.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_headers.append(col)

    rows = []
    for row in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True).replace("\xa0", " ") for td in row.find_all("td")]
        if len(cols) == len(unique_headers):
            rows.append(cols)

    df = pd.DataFrame(rows, columns=unique_headers)

    # Detect name column
    name_candidates = [col for col in df.columns if col.lower() in ["name", "player", "full name", "nombre"]]
    if name_candidates:
        df.rename(columns={name_candidates[0]: "Name"}, inplace=True)

    if "Name" in df.columns:
        df = df[df["Name"].str.strip() != ""]
    else:
        st.warning("⚠️ Could not find a 'Name' column in the file.")
        return pd.DataFrame()

    return df

# --- Radar Chart Function ---
def plot_radar(player_name, player_row, stat_cols):
    try:
        values = []
        for col in stat_cols:
            val = player_row.get(col, "0")
            val = val.strip("%") if isinstance(val, str) else val
            try:
                val = float(val)
            except:
                val = 0.0
            values.append(val)
        values = np.array(values)
    except Exception as e:
        st.error(f"Radar data error: {e}")
        return None

    max_val = max(values.max(), 1)
    scaled = values / max_val * 100

    angles = np.linspace(0, 2 * np.pi, len(stat_cols), endpoint=False).tolist()
    scaled = np.concatenate((scaled, [scaled[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scaled, color="blue", linewidth=2)
    ax.fill(angles, scaled, color="skyblue", alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(stat_cols, fontsize=9)
    ax.set_title(f"{player_name} — Radar Chart", size=14)
    ax.grid(True)
    return fig

# --- Load Files ---
squad_df = parse_html_to_df(squad_file) if squad_file else None
market_df = parse_html_to_df(market_file) if market_file else None

# --- Display Tables ---
if squad_df is not None and not squad_df.empty:
    st.subheader("🏠 Your Squad")
    st.dataframe(squad_df, use_container_width=True)

if market_df is not None and not market_df.empty:
    st.subheader("🛒 Transfer Market")
    st.dataframe(market_df, use_container_width=True)

# --- AI Section ---
st.subheader("🤖 AI Analysis")
user_query = st.text_area("Ask a question (e.g., 'Best defenders on the market', 'Top creators in my squad')")

if st.button("Analyze with ChatGPT") and user_query:
    with st.spinner("Analyzing..."):
        try:
            prompt = "You are analyzing a Football Manager 2024 squad and transfer market.\n\n"

            if squad_df is not None:
                prompt += "Squad Stats:\n" + squad_df.head(50).to_markdown(index=False) + "\n\n"
            if market_df is not None:
                prompt += "Transfer Market Stats:\n" + market_df.head(50).to_markdown(index=False) + "\n\n"

            prompt += f"User Query: {user_query}"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a tactical football analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            st.markdown("### 💡 ChatGPT's Answer")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Error calling ChatGPT: {e}")

# --- Radar Chart Section ---
st.subheader("📈 Player Stat Radar")
player_source = st.radio("Choose from:", ["Squad", "Transfer Market"])
player_df = squad_df if player_source == "Squad" else market_df

if player_df is not None and "Name" in player_df.columns:
    selected_player = st.selectbox("Select Player", player_df["Name"].unique())
    player_row = player_df[player_df["Name"] == selected_player].iloc[0]

    st.markdown("### 📋 Player Stats")
    st.dataframe(player_row.to_frame().T)

    # Metrics from user's FM24 export
    radar_metrics = [
        "Assists",
        "Goals",
        "Expected Goals per 90 Minutes",
        "Expected Goals Overperformance",
        "Expected Assists",
        "Key Passes",
        "Dribbles Made",
        "Pass Completion Ratio",
        "Interceptions",
        "Headers Won",
        "Tackle Completion Ratio"
    ]

    available_metrics = [m for m in radar_metrics if m in player_df.columns]

    if len(available_metrics) >= 3:
        fig = plot_radar(selected_player, player_row, available_metrics)
        st.pyplot(fig)
    else:
        st.info("Not enough metrics available for radar chart.")
