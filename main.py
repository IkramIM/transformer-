import pandas as pd
from src.tokenizer import Token ,vocab_text
from src.dataset_cleaning import clean_data
from src.dataset import Mood2FlowersDataset
from src.dataloader import get_dataloader

df = clean_data()


#columns to use for vocabulary
text_columns = ['emotion', 'occasion', 'description'] 

# vocabulary creation from specified text columns
texts = []
for col in text_columns:
    texts += df[col].dropna().astype(str).tolist()
vocab = vocab_text(texts) 
print(f"Vocabulary size: {len(vocab)}")
print(f"Example from vocab: {vocab[:10]}")

# tokenizer creation 
tokenizer = Token(vocab)

# Create the dataset and dataloader
dataset = Mood2FlowersDataset(df, tokenizer, text_column='emotion', label_column='composition_(flowers)')

dataloader = get_dataloader(dataset, batch_size=4)

for batch in dataloader:
    print("Batch input_ids:", batch['input_ids'])
    print("Batch labels:", batch['label'])
    break 