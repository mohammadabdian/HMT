import torch
from torch import nn, Tensor
from einops import rearrange
from zeta.nn import SSM

class VisionMambaEncoder(nn.Module):
    """
    2D-Aware Vision Mamba Post-Encoder for CLIP features.
    Novelty: Processes spatial patches in both horizontal (row-wise) 
    and vertical (column-wise) directions independently to preserve 2D spatial locality,
    while leaving the global CLS token strictly untouched.
    """
    def __init__(
        self,
        dim: int,
        dt_rank: int,
        d_state: int,
        spatial_size: int = 7  # 7x7 = 49 for CLIP ViT-B/32
    ):
        super().__init__()
        self.dim = dim
        self.spatial_size = spatial_size
        
        self.norm = nn.LayerNorm(dim)
        
        # Two independent 1D Convs for horizontal and vertical parsing
        self.conv1d_h = nn.Conv1d(dim, dim, kernel_size=3, padding = 1)
        self.conv1d_v = nn.Conv1d(dim, dim, kernel_size=3, padding = 1)
        
        # SSMs for horizontal and vertical scans
        self.ssm_h = SSM(dim, dt_rank, dim, d_state)
        self.ssm_v = SSM(dim, dt_rank, dim, d_state)
        
        self.proj_x = nn.Linear(dim, dim)   
        self.proj_z = nn.Linear(dim, dim)  
        
        self.silu = nn.SiLU()
        self.out_norm = nn.LayerNorm(dim)
        
        # Learnable gating for the residual connection
        # Started slightly negative so initially it relies on original CLIP features
        self.gamma = nn.Parameter(torch.ones(1)*0.8) 

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [B, 50, D] (1 CLS + 49 Patches)
        
        # 1. ISOLATE CLS TOKEN (Crucial to prevent destroying global context)
        cls_token = x[:, 0:1, :]    # [B, 1, D]
        patches = x[:, 1:, :]       # [B, 49, D]
        
        skip_patches = patches
        patches = self.norm(patches)
        
        # Gating signal
        z = self.silu(self.proj_z(patches))
        patches = self.proj_x(patches)
        
        # 2. SPATIAL 2D SCAN
        B, N, D = patches.shape
        H = W = self.spatial_size # 7
        
        # Reshape to 2D grid: [B, H, W, D]
        patches_2d = patches.view(B, H, W, D)
        
        # --- Horizontal Scan (Row-wise) ---
        # Treat each row as an independent sequence. 
        # Shape becomes [B*H, W, D]
        h_seq = patches_2d.view(B * H, W, D)
        h_seq = rearrange(h_seq, "bh w d -> bh d w")
        h_seq = self.conv1d_h(h_seq)
        h_seq = rearrange(h_seq, "bh d w -> bh w d")
        h_out = self.ssm_h(h_seq)
        h_out = h_out.view(B, H, W, D)
        
        # --- Vertical Scan (Column-wise) ---
        # Transpose spatial dims to treat columns as sequences
        # Shape becomes [B*W, H, D]
        v_seq = patches_2d.transpose(1, 2).contiguous().view(B * W, H, D)
        v_seq = rearrange(v_seq, "bw h d -> bw d h")
        v_seq = self.conv1d_v(v_seq)
        v_seq = rearrange(v_seq, "bw d h -> bw h d")
        v_out = self.ssm_v(v_seq)
        # Transpose back to original [B, H, W, D]
        v_out = v_out.view(B, W, H, D).transpose(1, 2).contiguous()
        
        # 3. COMBINE 2D SCANS
        # Merge the horizontal and vertical context
        mamba_out = (h_out + v_out) 
        
        # Flatten back to [B, 49, D]
        mamba_out = mamba_out.view(B, N, D)
        
        # Apply gating and output norm
        mamba_out = mamba_out * z
        mamba_out = self.out_norm(mamba_out)
        
        # 4. RESIDUAL CONNECTION
        # Using sigmoid on gamma controls how much Mamba features are mixed with raw CLIP
        out_patches = skip_patches + (torch.sigmoid(self.gamma) * mamba_out)
        
        # 5. RE-ATTACH CLS TOKEN
        final_out = torch.cat([cls_token, out_patches], dim=1) # [B, 50, D]
        
        return final_out
