import io
import os

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Twitter Sentiment Analysis - Frontend
# Talks to the FastAPI backend (see ../backend/main.py) instead
# of loading the model itself.
# ============================================================

API_URL = os.environ.get("SENTIMENT_API_URL", "http://localhost:8000")

LABEL_ORDER = ["Irrelevant", "Negative", "Neutral", "Positive"]

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="centered",
)

st.title("🐦 Twitter Sentiment Analysis")
st.caption(f"Backend: {API_URL}")
st.write(
    "Fine-tuned BERT model that classifies a tweet as Positive, "
    "Negative, Neutral, or Irrelevant."
)

st.divider()

# ------------------------------------------------------------
# Backend health check
# ------------------------------------------------------------
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if health.get("status") == "ok":
        st.success(f"Connected to backend (running on {health.get('device')}).")
    else:
        st.warning("Backend is reachable but still loading the model. Try again shortly.")
except requests.exceptions.RequestException:
    st.error(
        f"Can't reach the backend at {API_URL}. "
        "Make sure it's running: `uvicorn main:app --reload --port 8000` "
        "from the backend/ folder."
    )
    st.stop()

single_tab, batch_tab = st.tabs(["Single tweet", "Batch (CSV upload)"])

# ------------------------------------------------------------
# Single tweet tab
# ------------------------------------------------------------
with single_tab:
    tweet = st.text_area(
        "Enter your tweet",
        height=150,
        placeholder="Example: I absolutely love this game!",
    )

    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if not tweet.strip():
            st.warning("Please enter a tweet.")
        else:
            with st.spinner("Analyzing tweet..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict", json={"text": tweet}, timeout=30
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Request to backend failed: {e}")
                    result = None

            if result:
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]
                cleaned_tweet = result["cleaned_text"]

                st.divider()
                st.subheader("Prediction")

                if sentiment == "Positive":
                    st.success(f"😊 {sentiment}")
                elif sentiment == "Negative":
                    st.error(f"😞 {sentiment}")
                elif sentiment == "Neutral":
                    st.info(f"😐 {sentiment}")
                else:
                    st.warning(f"❓ {sentiment}")

                st.metric("Confidence", f"{confidence:.2%}")

                st.subheader("Class Probabilities")
                for label in LABEL_ORDER:
                    probability = probabilities[label]
                    st.write(f"**{label}:** {probability:.2%}")
                    st.progress(float(probability))

                with st.expander("View processed tweet"):
                    st.write(cleaned_tweet if cleaned_tweet else "*(empty after cleaning)*")

# ------------------------------------------------------------
# Batch CSV tab
# ------------------------------------------------------------
with batch_tab:
    st.write(
        "Upload a CSV with a column of tweet text and get a sentiment "
        "prediction for every row."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read that CSV: {e}")
            df = None

        if df is not None:
            if df.empty:
                st.warning("That CSV has no rows.")
            else:
                text_col = st.selectbox(
                    "Which column contains the tweet text?",
                    options=list(df.columns),
                    index=len(df.columns) - 1,
                )

                if st.button("🔍 Analyze CSV", use_container_width=True):
                    texts = df[text_col].fillna("").astype(str).tolist()

                    with st.spinner(f"Analyzing {len(texts)} rows..."):
                        results = []
                        try:
                            # The API guards batches at 256 rows per call, so chunk larger files.
                            chunk_size = 256
                            for start in range(0, len(texts), chunk_size):
                                chunk = texts[start : start + chunk_size]
                                resp = requests.post(
                                    f"{API_URL}/predict/batch",
                                    json={"texts": chunk},
                                    timeout=120,
                                )
                                resp.raise_for_status()
                                results.extend(resp.json())
                        except requests.exceptions.RequestException as e:
                            st.error(f"Request to backend failed: {e}")
                            results = None

                    if results:
                        result_df = df.copy()
                        result_df["predicted_sentiment"] = [r["sentiment"] for r in results]
                        result_df["confidence"] = [r["confidence"] for r in results]

                        st.divider()
                        st.subheader("Results")
                        st.dataframe(result_df, use_container_width=True)

                        st.subheader("Sentiment breakdown")
                        st.bar_chart(result_df["predicted_sentiment"].value_counts())

                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "⬇️ Download results as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="sentiment_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
