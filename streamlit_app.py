# streamlit_app.py
import streamlit as st
import pandas as pd
from utils_bert import predict_sentiment

st.set_page_config(page_title="Myntra Sentiment Analyzer", layout="centered")
st.title("🛍 Myntra Customer Review Sentiment Analyzer (BERT)")

st.write("Enter a single review below or upload a CSV of reviews to analyze sentiments.")

# --- Single Review ---
st.subheader("Single Review Prediction")
review_text = st.text_area("Type a review:", height=100)
if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment = predict_sentiment(review_text)
        st.success(f"Predicted Sentiment: **{sentiment.upper()}**")

# --- Batch Reviews ---
st.subheader("Batch Review Prediction (CSV)")
uploaded_file = st.file_uploader("Upload CSV with a 'review' column", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'review' not in df.columns:
        st.error("CSV must have a 'review' column.")
    else:
        df['predicted_sentiment'] = df['review'].apply(predict_sentiment)
        st.write("Predicted Sentiments for Uploaded Data:")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, "predicted_sentiments.csv", "text/csv")

st.info("This app uses a fine-tuned BERT model (or pretrained fallback) to classify Myntra customer reviews.")
