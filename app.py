import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# --- App Config ---
st.set_page_config(page_title="FM24 Squad Analyzer", layout="wide")
st.title("📊 Football Manager 2024 Squad Analyzer")
st.markdown("Upload your exported FM24 squad stats (.html), and get AI-powered insights and charts.")

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

    df = pd.DataFrame(rows, columns=unique_headers)
    return df

# --- Helper: Radar Chart ---
def plot_radar_chart(metrics, values, player_name):
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], metrics)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'skyblue', alpha=0.4)

    plt.title(f"Performance Metrics: {player_name}", size=14)
    st.pyplot(fig)

# --- Position-based Metrics ---
position_metrics_map = {
    "Goalkeeper": [
        "Save Ratio",
        "Clean Sheets",
        "Saves Held",
        "Saves Parried",
        "Saves Tipped",
        "Pass Completion Ratio"
    ],
    # Add mappings for other positions as needed
}

# --- App Logic ---
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    df = parse_html_to_df(uploaded_file)

    st.subheader("📋 Raw Player Stats")
    st.dataframe(df, use_container_width=True)

    st.subheader("🔍 Select a Player for Radar Chart")
    player_names = df["Name"].unique().tolist()
    selected_player = st.selectbox("Choose a player:", player_names)

    if selected_player:
        player_row = df[df["Name"] == selected_player].iloc[0]
        position = player_row.get("Position", "")

        if position in position_metrics_map:
            metrics = position_metrics_map[position]
            values = []

            for metric in metrics:
                try:
                    val = float(player_row[metric])
                    values.append(val)
                except:
                    pass

            if len(values) >= 3:
                st.subheader("📈 Player Radar Chart")
                plot_radar_chart(metrics[:len(values)], values, selected_player)
            else:
                st.warning("⚠️ Not enough metrics available to plot radar chart.")
        else:
            st.warning(f"⚠️ No metrics configured for position: {position}")

    st.subheader("🤖 Ask the AI About Your Squad")
    user_query = st.text_area("Ask a question (e.g., 'Who should I sell?', 'Top 3 midfielders?', 'Any weak defenders?')")

    if st.button("Analyze with ChatGPT") and user_query:
        with st.spinner("Thinking..."):
            try:
                sample_df = df.copy()
                sample_df = sample_df.head(25)  # Avoid API token limit

                prompt = f"""
                You are an assistant analyzing a Football Manager 2024 squad.
                Here are the player stats:

                {sample_df.to_markdown(index=False)}

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
else:
    st.info("📁 Please upload your FM24 HTML export to begin.")
