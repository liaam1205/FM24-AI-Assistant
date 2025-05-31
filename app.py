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

# --- Upload Squad and Market Files ---
st.sidebar.header("📁 Upload Files")
squad_file = st.sidebar.file_uploader("Upload Squad Export (.html)", type=["html"], key="squad")
market_file = st.sidebar.file_uploader("Upload Transfer Market Export (.html)", type=["html"], key="market")

# --- Improved HTML Parser ---
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

    # Try to identify the player name column
    name_candidates = [col for col in df.columns if col.lower() in ["name", "player", "full name", "nombre"]]
    if name_candidates:
        df.rename(columns={name_candidates[0]: "Name"}, inplace=True)

    # Drop rows without valid names
    if "Name" in df.columns:
        df = df[df["Name"].str.strip() != ""]
    else:
        st.warning("⚠️ Could not find a 'Name' column in the file.")
        return pd.DataFrame()

    return df

# --- Pizza Chart ---
def plot_pizza_chart(player_name, player_row, stat_cols):
    try:
        stats = player_row[stat_cols].astype(float).values
    except:
        stats = [float(str(x).replace("%", "").strip()) if str(x).replace(".", "", 1).replace("%", "").isdigit() else 0 for x in player_row[stat_cols]]
        stats = np.array(stats)

    max_val = np.max(stats)
    stats = stats / max_val * 100 if max_val > 0 else stats

    angles = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, color="green", linewidth=2)
    ax.fill(angles, stats, color="green", alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(stat_cols, fontsize=10)
    ax.set_title(f"{player_name} — Performance Radar", size=14)
    ax.grid(True)
    return fig

# --- Load DataFrames ---
squad_df = parse_html_to_df(squad_file) if squad_file else None
market_df = parse_html_to_df(market_file) if market_file else None

# --- Display Tables ---
if squad_df is not None:
    st.subheader("🏠 Your Squad")
    st.dataframe(squad_df, use_container_width=True)

if market_df is not None:
    st.subheader("🛒 Transfer Market")
    st.dataframe(market_df, use_container_width=True)

# --- AI Insights ---
st.subheader("🤖 AI Analysis")
user_query = st.text_area("Ask a question (e.g., 'Best midfielders on the market', 'Compare my CBs with available ones')")

if st.button("Analyze with ChatGPT") and user_query:
    with st.spinner("Thinking..."):
        try:
            prompt = "You are an assistant analyzing a Football Manager 2024 squad and transfer market.\n\n"

            if squad_df is not None:
                prompt += "Current Squad (top 50 rows):\n"
                prompt += squad_df.head(50).to_markdown(index=False)
                prompt += "\n\n"

            if market_df is not None:
                prompt += "Transfer Market (top 50 rows):\n"
                prompt += market_df.head(50).to_markdown(index=False)
                prompt += "\n\n"

            prompt += f"User question: {user_query}"

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
            st.error(f"❌ ChatGPT API call failed: {e}")

# --- Player Stat Radar ---
st.subheader("📈 Player Stat Radar")

player_source = st.radio("Select from:", ["Squad", "Transfer Market"])
player_df = squad_df if player_source == "Squad" else market_df

if player_df is not None and "Name" in player_df.columns:
    selected_player = st.selectbox("Choose a player", player_df["Name"].unique())
    player_data = player_df[player_df["Name"] == selected_player]

    if not player_data.empty:
        st.markdown("### 📊 Detailed Stats")
        st.dataframe(player_data.T, use_container_width=True)

        radar_stats = [
            "xG", "xA", "Goals", "Assists", "KeyPasses",
            "DribblesCompleted", "ShotsOnTarget%", "PassAccuracy",
            "Tackles", "Interceptions"
        ]
        radar_cols = [col for col in radar_stats if col in player_data.columns]

        if len(radar_cols) >= 3:
            fig = plot_pizza_chart(selected_player, player_data.iloc[0], radar_cols)
            st.pyplot(fig)
        else:
            st.info("⚠️ Not enough metrics for radar chart.")
else:
    st.info("Please upload a valid file with a 'Name' column.")
