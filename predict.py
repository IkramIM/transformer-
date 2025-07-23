import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.model.model import TransformerClassifier 
from src.dataset import Mood2FlowersDataset
from src.tokenizer import Token, load_vocab

# ========== Config ==========
model_path = "model.pth"
vocab_path = "vocab.json"
data_path = "data/mood2flowers.csv"   # Test data file
max_len = 64
batch_size = 32

# ========== Load data ==========
df = pd.read_csv(data_path)
df = df.fillna("")  # Fill missing text with empty string
texts = df['text'].tolist()

# ========== Load tokenizer ==========
vocab = load_vocab(vocab_path)
tokenizer = Token(vocab)

# ========== Create dataset and dataloader ==========
dataset = Mood2FlowersDataset(texts, tokenizer, max_length=max_len)
loader = DataLoader(dataset, batch_size=batch_size)

# ========== Load model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(vocab_size=len(vocab), num_classes=10)  
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ========== Make predictions ==========
all_preds = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        outputs = model(input_ids)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())

# ========== Mapping IDs to labels ==========
label2id = {
    'A bouquet of sunflowers': 0,
    'A bouquet of red roses': 1,
    'A bouquet of lilies': 2,
    'A bouquet of white daisies': 3,
    'A bouquet of lavender': 4,
    'A bouquet of mixed colors': 5,
    'A bouquet of orchids': 6,
    'A bouquet of yellow tulips': 7,
    'A bouquet of blue hydrangeas': 8,
    'A bouquet of carnations': 9
}
id2label = {v: k for k, v in label2id.items()}
pred_labels = [id2label[pred] for pred in all_preds]

# ========== Save full prediction file (optional) ==========
df['Predicted_Composition'] = pred_labels
df.to_csv("predictions.csv", index=False)

# ========== Save submission.csv ==========
submission_df = pd.DataFrame({
    "index": list(range(len(pred_labels))),
    "label": pred_labels
})
submission_df.to_csv("submission.csv", index=False)


