import os
import torch
import pandas as pd
from torch.utils.data import Dataset , DataLoader, random_split
from tokenizer import Token, build_vocab, save_vocab, load_vocab, pad_and_truncate 


def prepare_tokenizer(data, vocab_path="vocab.json", min_freq=2, max_length=32):
    if not os.path.exists(vocab_path):
        print("🔧 vocab.json not found. Building new vocab...")
        texts = data[['emotion', 'occasion', 'description']].fillna("").agg(" ".join, axis=1).tolist()
        vocab = build_vocab(texts, min_freq=min_freq)
        save_vocab(vocab, vocab_path)
    else:
        print("📄 vocab.json found. Loading existing vocab...")

    vocab = load_vocab(vocab_path)
    tokenizer = Token(vocab)
    return tokenizer

class Mood2FlowersDataset(Dataset):
    def __init__(self, dataframe, vocab, label2id, max_length=64):
        self.texts = dataframe["text"].tolist()
        self.labels = [label2id[label] for label in dataframe["label"]]
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
     text = self.texts[idx]
     label = self.labels[idx]

     # Tokenize input text
     input_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in text.split()]
     attention_mask = [1] * len(input_ids)

     # Apply padding/truncation
     input_ids = pad_and_truncate(input_ids, self.max_length, pad_token=self.vocab["<pad>"])
     attention_mask = pad_and_truncate(attention_mask, self.max_length, pad_token=0)

     return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "label": torch.tensor(label, dtype=torch.long)
    }


    def get_num_classes(self):
        if 'label' not in self.data.columns:
            return 0
        return len(set(self.data['label']))



def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def create_dataloaders(df, tokenizer, batch_size=32, val_split=0.2, max_len=128):
    
    dataset = Mood2FlowersDataset(df, tokenizer, max_len=max_len)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, dataset.get_num_classes()
