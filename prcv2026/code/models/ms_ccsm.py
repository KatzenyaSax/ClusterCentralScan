"""
Multi-Scale Cluster-Centric Scanning Module (MS-CCSM).

Runs CCSM (cluster-centric scanning) and SCFM (detail compensation) in
parallel, fuses their outputs. Also exposes centroid parameters for use
by Cluster-Gate.

References:
  MS-CCSM is the multi-scale extension of C2SSM's CCSM, augmented with SCFM.
"""

import torch
import torch.nn as nn

from .ccsman import CCSM
from .scfm import SCFM


class MSCCSM(nn.Module):
    """MS-CCSM: CCSM(K) + SCFM in parallel with fused output.

    Args:
        dim: Input/output channels
        K: Number of cluster centroids at this decoder stage
        d_state: Mamba S6 state dimension
    """

    def __init__(self, dim: int, K: int, d_state: int = 16):
        super().__init__()
        self.ccsm = CCSM(dim=dim, K=K, d_state=d_state)
        self.scfm = SCFM(dim=dim)
        # Fuse CCSM output + SCFM output
        self.fuse = nn.Conv2d(dim * 2, dim, 1)

    def forward(
        self, x: torch.Tensor, return_params: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
            return_params: return centroids & weights for Cluster-Gate
        Returns:
            F_out: (B, C, H, W)
            (Ĉ, W): optional centroid params
        """
        if return_params:
            f_global, centroids, weights = self.ccsm(x, return_params=True)
        else:
            f_global = self.ccsm(x)
        f_detail = self.scfm(x)
        f_out = self.fuse(torch.cat([f_global, f_detail], dim=1))

        if return_params:
            return f_out, centroids, weights
        return f_out
