# Customer Review Sentiment Analyzer (BERT)

A Streamlit web app that classifies Myntra customer reviews as Positive, Neutral, or Negative using a fine-tuned BERT model.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cust-review-analyzer.streamlit.app/)

![App Preview](IMAGES/THUBMNAIL.png)


## Features
- Fine-tuned DistilBERT for e-commerce reviews
- Single review and bulk CSV upload support
- Automatically loads your model from `bert_sentiment_model/`
- Batch sentiment predictions downloadable as CSV

## Setup
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
