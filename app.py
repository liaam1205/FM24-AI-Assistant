import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai

# --- App Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("📊 Football Manager 2024 Squad & Transfer Market Analyzer")
st.markdown("Upload your FM24 squad and transfer market exports (.html or .csv), and get AI-powered insights and recruitment advice.")

# --- API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- Helper function to parse FM24 HTML table exports ---
def parse_html_to_df(file):
    soup = BeautifulSoup(file, "html.parser")
    table = soup.find("table")
    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    # Make column headers unique
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

# --- Squad File Upload ---
st.header("1️⃣ Upload Your Squad Export")
uploaded_squad_file = st.file_uploader("Upload your FM24 squad HTML export", type=["html"], key="squad")

df_squad = None
if uploaded_squad_file is not None:
    st.success("✅ Squad file uploaded successfully!")
    df_squad = parse_html_to_df(uploaded_squad_file)
    st.subheader("📋 Raw Squad Player Stats")
    st.dataframe(df_squad, use_container_width=True)

# --- Transfer Market File Upload ---
st.header("2️⃣ Upload Transfer Market Data")
uploaded_transfer_file = st.file_uploader("Upload Transfer Market data (HTML or CSV)", type=["html", "csv"], key="transfer")

df_transfer = None
if uploaded_transfer_file is not None:
    if uploaded_transfer_file.type == "text/csv":
        df_transfer = pd.read_csv(uploaded_transfer_file)
    else:
        df_transfer = parse_html_to_df(uploaded_transfer_file)
    st.success("✅ Transfer market file uploaded successfully!")
    st.subheader("📋 Transfer Market Players")
    st.dataframe(df_transfer, use_container_width=True)

# --- AI Analysis Section ---
st.header("3️⃣ AI-Powered Analysis")

# User input for squad questions
if df_squad is not None:
    st.subheader("Ask AI About Your Squad")
    user_query = st.text_area("Ask a question about your squad (e.g., 'Who should I sell?', 'Top 3 midfielders?', 'Any weak defenders?')")

    if st.button("Analyze Squad with ChatGPT") and user_query:
        with st.spinner("Analyzing squad..."):
            try:
                prompt = f"""
You are a tactical football analyst analyzing a Football Manager 2024 squad.

Here are the player stats:

{df_squad.to_markdown(index=False)}

Answer the user's question based on these stats:

User question: {user_query}
"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a tactical football analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800
                )

                answer = response.choices[0].message.content
                st.markdown("### 🧠 ChatGPT's Squad Insights")
                st.markdown(answer)

            except Exception as e:
                st.error(f"⚠️ ChatGPT API call failed: {e}")

# User input for transfer market questions / recommendations
if df_squad is not None and df_transfer is not None:
    st.subheader("Ask AI About Transfer Market or Get Recommendations")
    transfer_query = st.text_area("Ask about potential recruitments or leave blank for AI suggestions")

    if st.button("Analyze Transfer Market with ChatGPT"):
        with st.spinner("Analyzing transfer market..."):
            try:
                prompt = f"""
You are a Football Manager 2024 recruitment analyst.

Current squad:
{df_squad.to_markdown(index=False)}

Transfer market players:
{df_transfer.to_markdown(index=False)}

{transfer_query if transfer_query.strip() else "Recommend 3 players from the transfer market who would best strengthen the current squad."}
"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a tactical football recruitment analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800
                )

                answer = response.choices[0].message.content
                st.markdown("### 🧠 ChatGPT's Transfer Market Recommendations")
                st.markdown(answer)

            except Exception as e:
                st.error(f"⚠️ ChatGPT API call failed: {e}")

if df_squad is None:
    st.info("📁 Please upload your FM24 squad HTML export to begin.")

if df_squad is not None and df_transfer is None:
    st.info("📁 Upload transfer market data (HTML or CSV) to analyze recruitment options.")
