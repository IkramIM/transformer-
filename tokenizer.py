import json
from collections import Counter

class Token:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"

    def encode(self, text):
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in text.split()]

def build_vocab(texts, vocab_path="vocab.json", max_vocab_size=10000):
    all_tokens = []
    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)

    counter = Counter(all_tokens)
    most_common = counter.most_common(max_vocab_size - 4)

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<sos>": 2,
        "<eos>": 3,
    }
    for i, (word, _) in enumerate(most_common, start=4):
        vocab[word] = i

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    return vocab

def save_vocab(vocab, path="vocab.json"):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_vocab(path="vocab.json"):
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def pad_and_truncate(tokens, max_length, pad_token="<pad>"):
    if len(tokens) > max_length:
        return tokens[:max_length]
    else:
        return tokens + [pad_token] * (max_length - len(tokens))
