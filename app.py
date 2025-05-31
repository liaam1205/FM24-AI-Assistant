import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import matplotlib.pyplot as plt
import numpy as np
import re

# --- App Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("⚽ Football Manager 2024 Squad & Transfer Analyzer")
st.markdown(
    """
Upload your FM24 exported squad and transfer market HTML files to analyze your squad and the available players.
Ask AI questions about your squad or transfer targets, and view detailed player stats with radar charts!
"""
)

# --- API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- Position Normalization with FM24 roles & positions ---
position_aliases = {
    "GK": "Goalkeeper",
    "D (C)": "Centre Back",
    "D (L)": "Fullback",
    "D (R)": "Fullback",
    "DM": "Defensive Midfielder",
    "M (C)": "Central Midfielder",
    "M (L)": "Wide Midfielder",
    "M (R)": "Wide Midfielder",
    "AM (C)": "Attacking Midfielder",
    "AM (L)": "Winger",
    "AM (R)": "Winger",
    "ST (C)": "Striker"
}

def normalize_position(pos_str):
    if not isinstance(pos_str, str):
        return "Unknown"
    positions = [p.strip() for p in pos_str.split(",")]
    for pos in positions:
        if pos in position_aliases:
            return position_aliases[pos]
    return "Unknown"

# --- Position-based metrics for radar charts ---
position_metrics = {
    "Goalkeeper": ["Pass Completion Ratio", "Save Ratio", "Clean Sheets", "Saves Held", "Saves Parried", "Saves Tipped"],
    "Centre Back": ["Assists", "Goals", "Headers Won", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio"],
    "Fullback": ["Assists", "Goals", "Dribbles Made", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio"],
    "Defensive Midfielder": ["Assists", "Goals", "Tackle Completion Ratio", "Interceptions", "Pass Completion Ratio", "Key Passes"],
    "Central Midfielder": ["Assists", "Goals", "Key Passes", "Dribbles Made", "Pass Completion Ratio", "Interceptions"],
    "Attacking Midfielder": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Expected Assists", "Key Passes"],
    "Wide Midfielder": ["Assists", "Goals", "Dribbles Made", "Key Passes", "Pass Completion Ratio", "Expected Goals per 90 Minutes"],
    "Winger": ["Assists", "Goals", "Dribbles Made", "Key Passes", "Pass Completion Ratio", "Expected Goals per 90 Minutes"],
    "Inside Forward": ["Assists", "Goals", "Dribbles Made", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Key Passes"],
    "Striker": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Conversion %", "Key Passes"],
    "Unknown": ["Assists", "Goals", "Expected Goals per 90 Minutes", "Expected Goals Overperformance", "Expected Assists", "Key Passes"]
}

def plot_radar_chart(player_row, position):
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    values = []
    labels = []

    for metric in metrics:
        try:
            val = float(player_row.get(metric, 0))
            values.append(val)
            labels.append(metric)
        except (ValueError, TypeError):
            pass

    if len(values) < 3:
        st.warning("⚠️ Not enough metrics for radar chart.")
        return

    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='green', linewidth=2)
    ax.fill(angles, values, color='green', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Radar Chart", size=12, color='black')
    st.pyplot(fig)
    except Exception as e:
        st.warning(f"Radar chart error: {e}")

# --- File Upload ---
squad_file = st.file_uploader("Upload Squad HTML Export", type="html")
transfer_file = st.file_uploader("Upload Transfer Market HTML Export", type="html")

squad_df, transfer_df = None, None
if squad_file:
    squad_df = parse_html_to_df(squad_file)
if transfer_file:
    transfer_df = parse_html_to_df(transfer_file)

if squad_df is not None:
    st.subheader("📋 Squad Data")
    st.dataframe(squad_df, use_container_width=True)

    st.subheader("🔍 Player Insight")
    player_names = squad_df["Name"].dropna().unique().tolist()
    selected_player = st.selectbox("Select a player to view radar chart", player_names)
    player_row = squad_df[squad_df["Name"] == selected_player].iloc[0]
    position = normalize_position(player_row.get("Position", ""))
    st.markdown(f"**Position**: {position}")
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    radar_chart(player_row, metrics)

    st.subheader("🤖 AI Squad Analysis")
    question = st.text_area("Ask AI about your squad")
    if st.button("Analyze Squad") and question:
        try:
            prompt = f"""
You are an assistant analyzing a Football Manager 2024 squad. Here are the player stats:
{(squad_df.head(30)).to_markdown(index=False)}

Answer the user's question based on the stats:
User question: {question}
"""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a tactical football analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown("### 💬 ChatGPT's Insights")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"ChatGPT API error: {e}")

if transfer_df is not None:
    st.subheader("📂 Transfer Market Data")
    st.dataframe(transfer_df, use_container_width=True)

    st.subheader("🔍 Player Analysis (Transfer Targets)")
    transfer_names = transfer_df["Name"].dropna().unique().tolist()
    selected_transfer = st.selectbox("Select a transfer target to view radar chart", transfer_names)
    transfer_row = transfer_df[transfer_df["Name"] == selected_transfer].iloc[0]
    position = normalize_position(transfer_row.get("Position", ""))
    st.markdown(f"**Position**: {position}")
    metrics = position_metrics.get(position, position_metrics["Unknown"])
    radar_chart(transfer_row, metrics)

    st.subheader("🤖 AI Transfer Market Analysis")
    transfer_question = st.text_area("Ask AI about transfer targets")
    if st.button("Analyze Transfers") and transfer_question:
        try:
            prompt = f"""
You are an assistant evaluating transfer market players in FM24. Here are their stats:
{(transfer_df.head(30)).to_markdown(index=False)}

Answer the user's question based on the stats:
User question: {transfer_question}
"""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a tactical football analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown("### 💬 ChatGPT's Transfer Insights")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"ChatGPT API error: {e}")
