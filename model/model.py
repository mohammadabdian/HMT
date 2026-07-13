import math
from typing import Optional
import torch
import torch.nn as nn

from model.mamba_decoder import MambaBlock, MambaConfig
from model.modules import LayerNorm, CrossAttention, FFN
from model.mamba_encoder import MambaEncoder


# =====================================================
# Config
# =====================================================

class Config(MambaConfig):
    def __init__(
        self,
        dim: int = 384,
        layers: int = 3,
        heads: int = 6,
        hidden: int = 1536,
        dropout: float = 0.1,
        vocab: int = 49409,
        img_dim: int = 768,
        pos_enc: bool = True,
        max_len: int = 48,
        # --- vision encoder params ---
        vision_layers: int = 3,
        dt_rank: int = 16,
        d_state: int = 128,
        **kwargs
    ):
        super().__init__(d_model=dim, n_layers=1, **kwargs)

        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.hidden = hidden
        self.dropout = dropout
        self.vocab = vocab
        self.img_dim = img_dim
        self.pos_enc = pos_enc
        self.max_len = max_len

        # vision params
        self.vision_layers = vision_layers
        self.dt_rank = dt_rank
        self.d_state = d_state


# =====================================================
# Residual
# =====================================================

class Residual(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, fn):
        nx = self.norm(x)
        out = fn(nx) if not isinstance(fn, nn.Module) else fn(nx)
        return x + self.drop(out)


# =====================================================
# Positional Encoding
# =====================================================

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, dim, 2).float()
            * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(pos * div)

        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# =====================================================
# Decoder Block 
# =====================================================
class DecoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.seq_block = MambaBlock(cfg)
        self.cross = CrossAttention(cfg.d_model, cfg.heads, cfg.dropout)
        self.ffn = FFN(cfg.d_model, cfg.hidden, cfg.dropout)

        self.r1 = Residual(cfg.dim, cfg.dropout)
        self.r2 = Residual(cfg.dim, cfg.dropout)
        self.r3 = Residual(cfg.dim, cfg.dropout)


    def forward(self, x: torch.Tensor, img: torch.Tensor, img_mask=None):

        # Sequence modeling (Mamba)
        x = self.r1(x, self.seq_block)

        # ===== Causal Mask for Text Tokens =====
        seq_len = x.size(1)


        # Cross Attention with mask
        x = self.r2(
            x,
            lambda y: self.cross(y, img, img_mask),
        )

        # FFN
        x = self.r3(x, self.ffn)

        return x


# =====================================================
# Captioner Model
# =====================================================

class Captioner(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab, cfg.dim)

        self.img_map = nn.Linear(cfg.img_dim, cfg.dim)

        self.vision_encoder = nn.ModuleList(
            [
                MambaEncoder(
                    cfg.dim,
                    cfg.dt_rank,
                    cfg.d_state
                )
                for _ in range(cfg.vision_layers)
            ]
        )

        self.pos = (
            PositionalEncoding(cfg.dim, cfg.max_len)
            if cfg.pos_enc else None
        )

        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg.layers)]
        )

        self.norm = LayerNorm(cfg.dim)
        self.out = nn.Linear(cfg.dim, cfg.vocab)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

    # =================================================
    # Forward
    # =================================================
   
    def forward(
        self,
        tokens: torch.Tensor,
        img_feats: torch.Tensor,
        img_mask: Optional[torch.Tensor] = None,
    ):

        # Token embedding
        x = self.embed(tokens) * math.sqrt(self.embed.embedding_dim)

        if self.pos is not None:
            x = self.pos(x)

        # Image projection
        img = self.img_map(img_feats)

        # Vision encoding (stacked)
        for layer in self.vision_encoder:
            img = layer(img)

        # Decoder blocks
        for b in self.blocks:
            x = b(x, img, img_mask)

        return self.out(self.norm(x))
        
