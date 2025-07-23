import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tokenizer import build_vocab

DATA_PATH = 'data/bouquet.csv'
CLEANED_DATA_PATH = 'data/mood2flowers.csv'
VOCAB_PATH = 'data/vocab.json'
MAX_VOCAB_SIZE = 10000

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()



def prepare_dataset():
    df = pd.read_csv(DATA_PATH, sep=';')

    df = df[['Emotion', 'Occasion', 'Description', 'Composition (Flowers)']]
    df.columns = ['emotion', 'occasion', 'description', 'label']

    df['emotion'] = df['emotion'].apply(clean_text)
    df['occasion'] = df['occasion'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)
    df['label'] = df['label'].apply(clean_text)

    df['text'] = df['emotion'] + ' [SEP] ' + df['occasion'] + ' [SEP] ' + df['description']

    df = df[['text', 'label']]

    df.to_csv(CLEANED_DATA_PATH, index=False)

    if not os.path.exists(VOCAB_PATH):
        build_vocab(df['text'], VOCAB_PATH, MAX_VOCAB_SIZE)

if __name__ == '__main__':
    prepare_dataset()
