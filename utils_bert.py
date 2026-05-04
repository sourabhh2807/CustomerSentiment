# utils_bert.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os

# Path to your fine-tuned model directory
MODEL_DIR = "bert_sentiment_model"

# Check if fine-tuned model exists, otherwise load pretrained
if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.pkl"))
    id_to_label = {v: k for k, v in label_map.items()}
else:
    # fallback to pretrained DistilBERT (2-class)
    PRETRAINED = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED)
    # 2-class fallback mapping
    id_to_label = {0: "negative", 1: "positive"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_sentiment(text: str) -> str:
    """Predict sentiment of a single text string."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding=True, max_length=192)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()

    return id_to_label.get(pred_id, "unknown")
