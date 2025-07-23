import torch
import torch.nn as nn

from model.positional_encoding import PositionalEncoding
from model.attention import MultiHeadAttention, PositionWiseFeedForward
from model.encoder import EncoderLayer
from model.classifier import Classifier

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=128, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask=None):
        # input_ids: [B, L]
        x = self.embedding(input_ids)  # [B, L, D]
        x = self.pos_encoding(x)       # [B, L, D]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)        # [B, L, D]

        return x                      # final encoder output


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, num_heads=4, d_ff=512,
                 num_layers=2, max_len=128, dropout=0.1, pooling="mean"):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout
        )

        self.classifier = Classifier(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(input_ids, attention_mask)  # [B, L, D]
        output = self.classifier(x, attention_mask)  # [B, num_classes]
        return output
