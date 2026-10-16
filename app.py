
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import sqlite3
from urllib.parse import urlparse

st.set_page_config(
    page_title="PhishGuard Dashboard",
    page_icon="🛡️",
    layout="wide"
)

DB_NAME = "phishguard.db"

# Create permanent database
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                scan_time TEXT DEFAULT
                    (datetime('now', 'localtime'))
            )
        """)

def get_history():
    with sqlite3.connect(DB_NAME) as conn:
        return pd.read_sql_query("""
            SELECT
                id AS 'Scan ID',
                url AS 'URL',
                prediction AS 'Prediction',
                confidence AS 'Confidence (%)',
                scan_time AS 'Scan Time'
            FROM scans
            ORDER BY id DESC
        """, conn)

def save_scan(url, prediction, confidence):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            INSERT INTO scans (url, prediction, confidence)
            VALUES (?, ?, ?)
        """, (url, prediction, confidence))

def clear_history():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("DELETE FROM scans")
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name = 'scans'"
        )

@st.cache_resource
def load_model():
    return joblib.load("phishguard_model.pkl")

init_db()
model = load_model()

if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #172554);
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ PhishGuard")
st.subheader("Phishing URL Detection & Website Safety Checker")
st.write("Scan URLs, review results, and export your scan history.")

st.divider()

# Dashboard
history = get_history()
total_scans = len(history)

phishing_count = (
    int((history["Prediction"] == "Phishing").sum())
    if total_scans else 0
)
legitimate_count = (
    int((history["Prediction"] == "Likely Legitimate").sum())
    if total_scans else 0
)

st.subheader("📊 Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Total Scans", total_scans)
c2.metric("Possible Phishing", phishing_count)
c3.metric("Likely Legitimate", legitimate_count)

st.divider()

# URL Scanner
st.subheader("🔍 Scan a Website URL")

url = st.text_input(
    "Enter website URL",
    placeholder="https://example.com"
)

if st.button("Scan URL"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        candidate = url.strip()
        parsed = urlparse(
            candidate if "://" in candidate else "//" + candidate
        )
        hostname = parsed.hostname

        valid_domain = (
            bool(hostname)
            and "." in hostname
            and not any(ch.isspace() for ch in candidate)
        )

        if not valid_domain:
            st.error("Enter a valid-looking domain, such as example.com.")
        else:
            normalized_url = (
                candidate if "://" in candidate
                else "https://" + candidate
            )

            try:
                prediction = int(model.predict([normalized_url])[0])
                probabilities = model.predict_proba([normalized_url])[0]
                confidence = round(
                    float(max(probabilities)) * 100, 2
                )

                result = (
                    "Phishing" if prediction == 0
                    else "Likely Legitimate"
                )

                # Save permanently to SQLite
                save_scan(normalized_url, result, confidence)

                st.session_state.last_result = {
                    "URL": normalized_url,
                    "Prediction": result,
                    "Confidence (%)": confidence
                }

                st.rerun()

            except Exception as error:
                st.error("Unable to scan this URL.")
                st.caption(str(error))

# Latest Result
last_result = st.session_state.last_result

if last_result:
    st.subheader("🔎 Latest Scan Result")
    st.write("**URL:**", last_result["URL"])

    if last_result["Prediction"] == "Phishing":
        st.error("⚠️ Possible phishing URL")
        st.write(
            "Avoid entering personal information until the website "
            "is independently verified."
        )
    else:
        st.success("✅ Classified as likely legitimate")
        st.write(
            "This result does not guarantee that the website is safe."
        )

    st.metric(
        "Model Confidence",
        f'{last_result["Confidence (%)"]:.1f}%'
    )

st.divider()

# Scan History
st.subheader("🕘 Scan History")
history = get_history()

if not history.empty:
    st.dataframe(
        history.drop(columns=["Scan ID"]),
        use_container_width=True,
        hide_index=True
    )

    csv_data = history.drop(
        columns=["Scan ID"]
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Scan History (CSV)",
        data=csv_data,
        file_name="phishguard_scan_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Scan History"):
        clear_history()
        st.session_state.last_result = None
        st.rerun()
else:
    st.info("No scans yet. Scan a URL to create your history.")

st.divider()

# Scan Analytics
st.subheader("📈 Scan Analytics")

if not history.empty:
    chart_data = (
        history["Prediction"]
        .value_counts()
        .rename_axis("Result")
        .reset_index(name="Number of Scans")
    )

    fig = px.pie(
        chart_data,
        names="Result",
        values="Number of Scans",
        title="Phishing vs Likely Legitimate URLs",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        chart_data,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Scan some URLs to view analytics.")

st.divider()

# Information panels
left, right = st.columns(2)

with left:
    st.subheader("🔎 What it checks")
    st.write("- URL character patterns")
    st.write("- Patterns learned from labeled URL examples")
    st.write("- Machine-learning classification")

with right:
    st.subheader("🔐 Safety Tips")
    st.write("- Check the domain carefully.")
    st.write("- Never enter passwords on suspicious websites.")
    st.write("- Verify important links through official sources.")

st.caption(
    "PhishGuard | Student Cybersecurity Project | "
    "Predictions can be incorrect; confidence is not a safety guarantee."
)
