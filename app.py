import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import openai
import re

# --- App Config ---
st.set_page_config(page_title="FM24 Squad & Transfer Analyzer", layout="wide")
st.title("📊 Football Manager 2024 Squad & Transfer Market Analyzer")
st.markdown("Upload your FM24 squad and transfer market exports (.html or .csv), browse players with filters, and get AI-powered insights.")

# --- API Key ---
api_key = st.secrets["API_KEY"]
client = openai.OpenAI(api_key=api_key)

# --- Helper: Parse FM24 HTML table exports ---
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

# --- Helper: Convert monetary strings like '€10M', '£500K' to numeric ---
def money_to_float(money_str):
    if pd.isna(money_str):
        return None
    money_str = money_str.strip()
    if money_str == "":
        return None
    # Remove currency symbols
    money_str = re.sub(r"[^\d\.KMk]", "", money_str)
    multiplier = 1
    if money_str.endswith("M") or money_str.endswith("m"):
        multiplier = 1_000_000
        money_str = money_str[:-1]
    elif money_str.endswith("K") or money_str.endswith("k"):
        multiplier = 1_000
        money_str = money_str[:-1]
    try:
        return float(money_str) * multiplier
    except:
        return None

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

    # --- Clean numeric columns in transfer market for filtering ---
    # Convert Value and Wage columns if they exist
    for col in ["Value", "Wage"]:
        if col in df_transfer.columns:
            df_transfer[col + "_num"] = df_transfer[col].apply(money_to_float)

    # Convert Age, Overall, Potential to numeric if they exist
    for col in ["Age", "Overall", "Potential"]:
        if col in df_transfer.columns:
            df_transfer[col] = pd.to_numeric(df_transfer[col], errors='coerce')

    # -------------------------------
    # Player Search & Filtering Section
    # -------------------------------
    st.header("🔍 Player Search & Filtering")

    # Define filters for columns present in df_transfer
    filter_columns = {
        "Position": None,
        "Age": (15, 45),
        "Value_num": (0, int(df_transfer["Value_num"].max() if "Value_num" in df_transfer.columns else 100000000)),
        "Wage_num": (0, int(df_transfer["Wage_num"].max() if "Wage_num" in df_transfer.columns else 100000)),
        "Overall": (0, 100),
        "Potential": (0, 100),
    }

    available_filters = {k: v for k, v in filter_columns.items() if k in df_transfer.columns}

    filtered_df = df_transfer.copy()

    for col, range_vals in available_filters.items():
        if range_vals is None:
            options = sorted(filtered_df[col].dropna().unique())
            selected = st.multiselect(f"Filter by {col}", options, default=options)
            filtered_df = filtered_df[filtered_df[col].isin(selected)]
        else:
            min_val = int(filtered_df[col].min()) if pd.notna(filtered_df[col].min()) else range_vals[0]
            max_val = int(filtered_df[col].max()) if pd.notna(filtered_df[col].max()) else range_vals[1]

            slider_vals = st.slider(
                f"Filter by {col.replace('_num','').capitalize()}",
                min_value=range_vals[0],
                max_value=range_vals[1],
                value=(min_val, max_val)
            )
            filtered_df = filtered_df[(filtered_df[col] >= slider_vals[0]) & (filtered_df[col] <= slider_vals[1])]

    st.markdown(f"### Showing {len(filtered_df)} players after filtering")

    # Drop the helper numeric columns for display (like Value_num)
    display_cols = [col for col in filtered_df.columns if not col.endswith("_num")]
    st.dataframe(filtered_df[display_cols], use_container_width=True)

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
