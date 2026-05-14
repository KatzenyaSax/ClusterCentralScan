"""
Spatial-Channel Feature Modulator (SCFM).

Parallel spatial + channel attention that preserves high-frequency
details potentially lost during cluster-centric scanning.
"""

import torch
import torch.nn as nn


class SCFM(nn.Module):
    """Spatial-channel feature modulator.

    Runs spatial attention and channel attention in parallel,
    then fuses both modulated branches.

    Args:
        dim: Input / output channels
    """

    def __init__(self, dim: int):
        super().__init__()
        # ── Spatial attention branch ──
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.spatial_act = nn.Sigmoid()

        # ── Channel attention branch ──
        self.ch_reduce = nn.Conv2d(dim, dim // 4, 1)
        self.ch_restore = nn.Conv2d(dim // 4, dim, 1)

        # ── Output projections ──
        self.proj_spatial = nn.Conv2d(dim, dim, 1)
        self.proj_channel = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        # ── Spatial attention ──
        s_max, _ = x.max(dim=1, keepdim=True)  # (B, 1, H, W)
        s_avg = x.mean(dim=1, keepdim=True)     # (B, 1, H, W)
        s = torch.cat([s_max, s_avg], dim=1)     # (B, 2, H, W)
        Ws = self.spatial_act(self.spatial_conv(s))

        # ── Channel attention ──
        cd = torch.relu(self.ch_reduce(x))
        cd = self.ch_restore(cd)
        c_max = cd.amax(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        c_avg = cd.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        Wc = torch.sigmoid(c_max + c_avg)

        # ── Apply and fuse ──
        out_s = self.proj_spatial(Ws * x)
        out_c = self.proj_channel(Wc * x)
        return out_s + out_c
