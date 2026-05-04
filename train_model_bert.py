import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import joblib

# ----------------------
# Config
# ----------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
OUTPUT_DIR = "bert_sentiment_model"

# ----------------------
# Dataset
# ----------------------
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                padding='max_length', max_length=self.max_len)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ----------------------
# Load data
# ----------------------
print("Loading CSV...")
df = pd.read_csv("myntra_reviews.csv")  # <--- replace with your file name
texts = df["review"].astype(str).tolist()
labels = df["sentiment"].astype(str).tolist()

# Encode labels to ints
le = LabelEncoder()
y = le.fit_transform(labels)

# If any class has fewer than 2 samples, skip stratify
if pd.Series(y).value_counts().min() < 2:
    stratify_opt = None
else:
    stratify_opt = y

# Automatically handle tiny datasets
if len(texts) < 20:  # tiny dataset
    stratify_opt = None
    test_size_opt = 0.4  # bigger test fraction
else:
    stratify_opt = y
    test_size_opt = 0.2

# Automatically handle tiny datasets
if len(texts) < 20:  # tiny dataset
    stratify_opt = None
    test_size_opt = 0.4  # bigger test fraction
else:
    stratify_opt = y
    test_size_opt = 0.2

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, y, test_size=test_size_opt, random_state=42, stratify=stratify_opt
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ----------------------
# Model
# ----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(le.classes_)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# ----------------------
# Training loop
# ----------------------
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.3f}")

# ----------------------
# Save model + tokenizer
# ----------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
joblib.dump({cls: idx for idx, cls in enumerate(le.classes_)},
            os.path.join(OUTPUT_DIR, "label_map.pkl"))
print(f"Model saved to {OUTPUT_DIR}")
