import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import io

st.set_page_config(page_title="FM24 Squad Analyzer", layout="wide")
st.title("📊 Football Manager 2024 Squad Analyzer")
st.markdown("Upload your exported FM24 squad stats (.html), and get AI-powered insights and charts.")

# --- Sidebar API Key Input ---
api_key = st.sidebar.text_input("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", type="password")
if api_key:
    openai.api_key = api_key

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your FM24 HTML export", type=["html"])

def parse_html_to_df(file):
    soup = BeautifulSoup(file, "html.parser")
    table = soup.find("table")
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    # Fix duplicate column names
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

# --- Display and Analyze ---
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    df = parse_html_to_df(uploaded_file)

    st.subheader("📋 Raw Player Stats")
    st.dataframe(df, use_container_width=True)

    st.subheader("🤖 Ask the AI About Your Squad")
    user_query = st.text_area("Ask a question (e.g., 'Who should I sell?', 'Top 3 midfielders?', 'Any weak defenders?')")

    if st.button("Analyze with ChatGPT") and api_key and user_query:
        with st.spinner("Thinking..."):
            prompt = f"""
You are an assistant analyzing a Football Manager 2024 squad.
Here are the player stats:

{df.to_markdown(index=False)}

Answer the user's question based on these stats:
"""
              response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a tactical football analyst."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    answer = response["choices"][0]["message"]["content"]
    st.markdown("### 🧠 ChatGPT's Insights")
    st.markdown(answer)
except Exception as e:
    st.error(f"⚠️ ChatGPT API call failed: {e}")

            answer = response["choices"][0]["message"]["content"]
            st.markdown("### 🧠 ChatGPT's Insights")
            st.markdown(answer)

elif uploaded_file is None:
    st.info("Please upload your FM24 export to begin.")
