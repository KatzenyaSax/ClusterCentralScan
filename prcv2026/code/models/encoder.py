"""
Mamba VSS Encoder for M2SCAN.

4-stage encoder with VSS Blocks (SS2D + FFN), Patch Merge downsampling,
and a final bottleneck stage.

Reference: VMamba (Liu et al.) — SS2D cross-scan mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# SS2D: 4-directional selective scan
# ═══════════════════════════════════════════════════════════════

class SS2D(nn.Module):
    """2D Selective Scan: unfold feature map along 4 directions,
    apply S6 to each, then merge.

    Pure-PyTorch fallback — replaces mamba-ssm CUDA kernel.
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        # Shared S6 parameters (4 directions share weights)
        self.s6 = _S6Block(dim, d_state)
        self.out_proj = nn.Linear(dim * 4, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 4 directional scans
        scans = [
            self._scan_hw(x),                                     # top-left → bottom-right
            self._scan_hw(torch.flip(x, dims=[2, 3])),            # bottom-right → top-left
            self._scan_hw(torch.flip(x, dims=[3])),               # flipped horizontally
            self._scan_hw(torch.flip(x, dims=[2])),               # flipped vertically
        ]
        # Apply S6 and de-scan
        outs = []
        for i, (seq, orig_shape, flip_dims) in enumerate(scans):
            seq_out = self.s6(seq)
            # Reshape back
            out_2d = seq_out.transpose(1, 2).reshape(orig_shape)
            # Reverse flips
            if flip_dims:
                out_2d = torch.flip(out_2d, dims=flip_dims)
            outs.append(out_2d)

        out = torch.cat(outs, dim=1)  # (B, 4C, H, W)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, 4C)
        out = self.out_proj(out)
        return out.permute(0, 3, 1, 2)  # (B, C, H, W)

    def _scan_hw(self, x):
        """Unfold HW spatial dims into a sequence (top-left → bottom-right)."""
        B, C, H, W = x.shape
        seq = x.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        return seq, (B, C, H, W), None  # no flip for the base direction


# ═══════════════════════════════════════════════════════════════
# VSS Block
# ═══════════════════════════════════════════════════════════════

class VSSBlock(nn.Module):
    """Visual State Space Block: LN → SS2D → LN → FFN (MLP).

    Analogue of a Transformer block with SS2D replacing self-attention.
    """

    def __init__(self, dim: int, d_state: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        # SS2D path
        residual = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = residual + self.ss2d(x_norm)
        # FFN path
        x_norm = self.norm2(x.permute(0, 2, 3, 1))
        x = x + self.mlp(x_norm).permute(0, 3, 1, 2)
        return x


# ═══════════════════════════════════════════════════════════════
# Patch Merge (downsampling)
# ═══════════════════════════════════════════════════════════════

class PatchMerging(nn.Module):
    """2× spatial downsampling with channel doubling.

    Rearranges 2×2 patches into channel dimension, then projects.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H/2, W/2, 4C)
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        return x.permute(0, 3, 1, 2)


# ═══════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════

class MambaEncoder(nn.Module):
    """4-stage Mamba VSS encoder with bottleneck.

    Args:
        in_chans: Input channels (3 for RGB)
        depths: VSS Blocks per stage [2, 2, 8, 2]
        dims: Channels per stage [64, 128, 256, 512]
        d_state: Mamba state dimension
        mlp_ratio: FFN expansion ratio
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 8, 2]
        if dims is None:
            dims = [64, 128, 256, 512]

        # ── Conv Stem: 3 → 64, stride=4 ──
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        # ── Stages ──
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(*[
                VSSBlock(dims[i], d_state, mlp_ratio)
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
            if i < len(depths) - 1:
                self.downs.append(PatchMerging(dims[i]))

        # ── Bottleneck ──
        self.bottleneck = nn.Sequential(*[
            VSSBlock(dims[-1], d_state, mlp_ratio) for _ in range(2)
        ])

        self.dims = dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            bottleneck: (B, 512, H/32, W/32)
            skips: [X1, X2, X3, X4]  at [H/4, H/8, H/16, H/32]
        """
        x = self.stem(x)
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)

        x = self.bottleneck(x)
        return x, skips


# ═══════════════════════════════════════════════════════════════
# S6 Block (shared between SS2D and CCSM)
# ═══════════════════════════════════════════════════════════════

class _S6Block(nn.Module):
    """Mamba S6 selective scan (shared implementation)."""

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        self.dim = dim

        self.in_proj = nn.Linear(dim, dim * 3 + d_state, bias=False)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.ones(dim, d_state) * 0.5))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        A = -torch.exp(self.A_log)
        proj = self.in_proj(x)
        z = proj[..., :C]
        b = proj[..., C:2 * C]
        c = proj[..., 2 * C:3 * C]
        delta = F.softplus(self.dt_proj(proj[..., 3 * C:]))

        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = delta.unsqueeze(-1) * b.unsqueeze(-1)

        h = torch.zeros(B, C, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        for t in range(L):
            A_t = A_bar[:, t, :, :]
            B_t = B_bar[:, t, :, :]
            x_t = x[:, t, :].unsqueeze(-1)
            h = A_t * h + B_t * x_t
            y_t = (h * c[:, t, :].unsqueeze(-1)).sum(dim=-1)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        y = y + self.D * x
        out = y * F.silu(z)
        return self.norm(self.out_proj(out))
