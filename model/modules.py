import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv = (var + self.eps).rsqrt()
        return self.weight * (x - mean) * inv + self.bias


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.d_k = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, s, _ = x_q.shape
        i = x_kv.shape[1]

        q = self.to_q(x_q).view(b, s, self.heads, self.d_k).transpose(1, 2)   # (B, H, S, Dk)
        k = self.to_k(x_kv).view(b, i, self.heads, self.d_k).transpose(1, 2)   # (B, H, I, Dk)
        v = self.to_v(x_kv).view(b, i, self.heads, self.d_k).transpose(1, 2)   # (B, H, I, Dk)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, S, I)

        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(1).unsqueeze(1)   # (B,1,1,I)
            elif m.dim() == 3:
                m = m.unsqueeze(1)               # (B,1,S,I)
            elif m.dim() == 4:
                pass
            else:
                raise ValueError(f"Unsupported mask dim: {m.dim()}")
            m = m.to(torch.bool).to(scores.device)
            scores = scores.masked_fill(m, float("-1e9"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                 # (B, H, S, Dk)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)  # (B, S, D)
        return self.to_out(out)


class FFN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.drop(F.silu(self.fc1(x))))
