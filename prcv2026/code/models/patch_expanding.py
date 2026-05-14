"""
Patch Expanding: inverse of Patch Merge.
Linear(C_in → 2*C_in) → PixelShuffle(H×W → 2H×2W) → Linear(2*C_in → C_out)

Used in M2SCAN decoder for symmetric upsampling.
"""

import torch
import torch.nn as nn


class PatchExpanding(nn.Module):
    """Symmetric counterpart to Patch Merging.

    Args:
        dim: Input channels
        dim_out: Output channels (default: half of input, matching
                 the 2× channel reduction of a Patch Merge inverse)
    """

    def __init__(self, dim: int, dim_out: int | None = None):
        super().__init__()
        if dim_out is None:
            dim_out = dim // 2
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim)
        self.project = nn.Linear(2 * dim, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C_out, 2H, 2W)
        """
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H, W, C) → (B, H, W, 2C)
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        x = self.norm(x)
        # (B, H, W, 2C) → (B, H, W, 2, 2, C//2) → (B, 2H, 2W, C//2)
        x = x.reshape(B, H, W, 2, 2, -1).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, 2 * H, 2 * W, -1)
        x = self.project(x)
        # → (B, C_out, 2H, 2W)
        return x.permute(0, 3, 1, 2)
