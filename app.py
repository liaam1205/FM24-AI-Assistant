import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for compatibility
import matplotlib.pyplot as plt
import numpy as np

# --- App Config ---
st.set_page_config(page_title="FM24 Squad Analyzer", layout="wide")
st.title("📊 Football Manager 2024 Squad Analyzer")
st.markdown("Upload your exported FM24 squad stats (.html), and get AI-powered insights and visualizations.")

# --- API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your FM24 HTML export", type=["html"])

# --- Helper: Parse HTML to DataFrame ---
def parse_html_to_df(file):
    soup = BeautifulSoup(file, "html.parser")
    table = soup.find("table")
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

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
        cols = [td.get_text(strip=True).replace("-", "") for td in row.find_all("td")]
        if len(cols) == len(unique_headers):
            rows.append(cols)
        else:
            st.warning(f"Skipping row due to column mismatch: {cols}")

    df = pd.DataFrame(rows, columns=unique_headers)
    return df

# --- Helper: Radar Chart ---
def plot_pizza_chart(player_name, player_row, stat_cols):
    try:
        stats = player_row[stat_cols].astype(float).values
    except:
        stats = [float(str(x).replace("%", "").strip()) if str(x).replace(".", "", 1).replace("%", "").isdigit() else 0 for x in player_row[stat_cols]]
        stats = np.array(stats)

    # Normalize to 0–100 for better comparison
    max_val = np.max(stats)
    stats = stats / max_val * 100 if max_val > 0 else stats

    angles = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))  # close loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, color="blue", linewidth=2)
    ax.fill(angles, stats, color="blue", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(stat_cols, fontsize=10)
    ax.set_title(f"{player_name} — Performance Radar", size=14)
    ax.grid(True)

    return fig

# --- Main Logic ---
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    df = parse_html_to_df(uploaded_file)

    st.subheader("📋 Raw Player Stats")
    st.dataframe(df, use_container_width=True)

    # --- AI Query ---
    st.subheader("🤖 Ask the AI About Your Squad")
    user_query = st.text_area("Ask a question (e.g., 'Who should I sell?', 'Top 3 midfielders?', 'Any weak defenders?')")

    if st.button("Analyze with ChatGPT") and user_query:
        with st.spinner("Thinking..."):
            try:
                trimmed_df = df.head(50)  # Limit to avoid token overflow
                prompt = f"""
You are an assistant analyzing a Football Manager 2024 squad.
Here are the player stats (first 50 rows):

{trimmed_df.to_markdown(index=False)}

Answer the user's question based on these stats:
"""
                full_prompt = prompt + "\n\nUser question: " + user_query

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a tactical football analyst."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800
                )

                answer = response.choices[0].message.content
                st.markdown("### 🧠 ChatGPT's Insights")
                st.markdown(answer)

            except Exception as e:
                st.error(f"⚠️ ChatGPT API call failed: {e}")

    # --- Player Insights ---
    st.subheader("📈 Player Insights")

    if "Name" in df.columns:
        selected_player = st.selectbox("Select a player to view detailed stats", df["Name"].unique())
        player_data = df[df["Name"] == selected_player]

        if not player_data.empty:
            st.markdown("### 🔍 Detailed Stats")
            st.dataframe(player_data.T, use_container_width=True)

            radar_stat_cols = [
                "xG", "xA", "Goals", "Assists", "KeyPasses",
                "DribblesCompleted", "ShotsOnTarget%", "PassAccuracy",
                "Tackles", "Interceptions"
            ]
            radar_stat_cols = [col for col in radar_stat_cols if col in player_data.columns]

            if len(radar_stat_cols) >= 3:
                fig = plot_pizza_chart(selected_player, player_data.iloc[0], radar_stat_cols)
                st.pyplot(fig)
            else:
                st.info("⚠️ Not enough stats available for a radar chart.")
    else:
        st.warning("⚠️ No 'Name' column found. Make sure your FM24 export includes player names.")

else:
    st.info("📁 Please upload your FM24 HTML export to begin.")
