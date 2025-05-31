import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

st.set_page_config(page_title="FM24 Data Load Test", layout="wide")
st.title("FM24 Squad Data Load Test")

def parse_html_to_df(html_file):
    try:
        soup = BeautifulSoup(html_file, "html.parser")
        table = soup.find("table")
        if not table:
            st.error("No table found in HTML.")
            return None

        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            row = [td.get_text(strip=True) for td in cells]
            if len(row) == len(headers):
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)
        return df
    except Exception as e:
        st.error(f"Parse error: {e}")
        return None

def display_player_details(df, player_name):
    try:
        player_row = df[df["Name"] == player_name]
        if player_row.empty:
            st.warning("Player not found.")
            return
        player_data = player_row.iloc[0].to_dict()
        st.write("### Player Details")
        st.json(player_data)
    except Exception as e:
        st.error(f"Display error: {e}")

# --- Upload squad file ---
squad_file = st.file_uploader("Upload FM24 squad HTML export", type=["html"])

if squad_file is not None:
    file_content = squad_file.read()
    squad_df = parse_html_to_df(file_content)

    if squad_df is not None:
        st.success("Squad data loaded successfully!")
        st.dataframe(squad_df.head())

        players = squad_df["Name"].dropna().unique().tolist()
        st.write(f"Loaded {len(players)} players.")

        selected_player = st.selectbox("Select a player", players)
        if selected_player:
            display_player_details(squad_df, selected_player)
    else:
        st.warning("Failed to load squad data.")
