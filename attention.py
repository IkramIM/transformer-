import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None):
        
        scores = torch.matmul(query, key.transpose(-2, -1))

        
        d_k = query.size(-1)
        scores = scores / math.sqrt(d_k)

      
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        
        attention_weights = F.softmax(scores, dim=-1)

        
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "Model dimension must be divisible by number of heads" 

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers لتحويل Q, K, V
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Attention
        self.attention = ScaledDotProductAttention()

        
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.linear_q(query)  # [B, L, D]
        K = self.linear_k(key)
        V = self.linear_v(value)

       
        def split_heads(x):
            # [B, L, D] → [B, H, L, d_k]
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        
        def combine_heads(x):
           
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = combine_heads(attn_output)

        
        output = self.linear_out(output)

        return output, attn_weights

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
