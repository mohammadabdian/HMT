import torch
from torch import nn, Tensor
from einops import rearrange
from zeta.nn import SSM


class MambaEncoder(nn.Module):
    """
    Bidirectional Vision Mamba block.
    Input:  (B, S, D)
    Output: (B, S, D)
    """

    def __init__(
        self,
        dim: int,
        dt_rank: int,
        d_state: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.forward_conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.backward_conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

        # SSM operates on embedding dimension D
        self.ssm = SSM(dim, dt_rank, dim, d_state)

        self.proj_x = nn.Linear(dim, dim)   
        self.proj_z = nn.Linear(dim, dim)  

        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        skip = x

        z = self.silu(self.proj_z(x))

        x = self.norm(x)
        x = self.proj_x(x)         

        x1 = self._process_direction(x, self.forward_conv1d)
        x2 = self._process_direction(x, self.backward_conv1d)

        x1 = x1 * z
        x2 = x2 * z

        return x1 + x2 + skip

    def _process_direction(
        self,
        x: Tensor,
        conv1d: nn.Conv1d,
    ) -> Tensor:

        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        x = rearrange(x, "b d s -> b s d")

        x = self.ssm(x)

        return x