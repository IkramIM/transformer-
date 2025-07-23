
import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1, pooling='mean'):
        super().__init__()
        self.pooling = pooling  # "mean" or "cls"
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        if self.pooling == 'mean':
            if mask is not None:
                mask = mask.unsqueeze(-1)  # [B, L, 1]
                x = x * mask  # zero out padded values
                summed = x.sum(dim=1)
                counts = mask.sum(dim=1)
                pooled = summed / counts.clamp(min=1e-9)
            else:
                pooled = x.mean(dim=1)  # [batch_size, d_model]
        elif self.pooling == 'cls':
            pooled = x[:, 0, :]  # [CLS] token

        output = self.dropout(pooled)
        return self.fc(output)
