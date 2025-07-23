import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from  tokenizer import Token, save_vocab , load_vocab , build_vocab 
from dataset import Mood2FlowersDataset 
from model.model import TransformerClassifier
import pandas as pd
from tokenizer import pad_and_truncate 

# Parameters
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "mood2flowers.csv")
vocab_path = os.path.join(current_dir, "..", "data", "vocab.json")
max_len = 64
batch_size = 32
num_epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv(data_path)
df["emotion"], df["occasion"], df["description"] = zip(*df["text"].apply(lambda x: x.split(" [SEP] ")))

# Convert labels to srt 
labels = df['label'].astype(str).tolist() 
label2id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
id2label = {idx: label for label, idx in label2id.items()}
num_classes = len(label2id)
labels = [label2id[label] for label in labels]

# Load or generate vocabulary
if os.path.exists(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
else:
    vocab_list = load_vocab("data/vocab.json")
    vocab = {word: idx + 2 for idx, word in enumerate(vocab_list)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    save_vocab(vocab, vocab_path)

tokenizer = Token(vocab)
# Split data
texts = df["text"].tolist()
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Combine text and label into one DataFrame as expected by Mood2FlowersDataset
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})

# Create datasets
train_dataset = Mood2FlowersDataset(train_df, tokenizer, max_len)
val_dataset = Mood2FlowersDataset(val_df, tokenizer, max_len)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Initialize model
model = TransformerClassifier(
    vocab_size=len(vocab),
    num_classes=num_classes,
    d_model=128,
    num_heads=4,
    d_ff=512,
    num_layers=2,
    max_len=max_len,
    dropout=0.1,
    pooling="mean"
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = total_loss / len(val_loader)
    val_acc = correct / total

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
