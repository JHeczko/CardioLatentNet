import torch
from torch import nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, seq_len, 1)
        pooled = (weights * x).sum(dim=1)              # (B, hidden_dim)
        return pooled