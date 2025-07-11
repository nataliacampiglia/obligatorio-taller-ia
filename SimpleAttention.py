import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # x: (batch, features)
        attn_weights = torch.softmax(self.attn(x), dim=1)  # (batch, 1)
        attended = x * attn_weights
        return attended