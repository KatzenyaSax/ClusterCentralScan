"""
M2SCAN Decoder: MS-CCSM + Cluster-Gate cascaded over 4 stages.

Each decoder stage:
  (Stage 4: no Patch Expanding — bottleneck output matches X4 resolution)
  Patch Expanding ↑2× → MS-CCSM(K) ─→ F_out ─┐
                              │               ├→ [fusion] → SegHead → p_i
                              └→ Cluster-Gate ┘
                                     ↑
                                    Xi
"""

import torch
import torch.nn as nn

from .ms_ccsm import MSCCSM
from .cluster_gate import ClusterGate
from .patch_expanding import PatchExpanding


class DecoderStage(nn.Module):
    """Single decoder stage.

    Args:
        dim: Input channels (from previous stage or bottleneck)
        skip_dim: Skip connection channels (matching encoder Xi)
        K: Number of cluster centroids for this stage
        d_state: Mamba S6 state dimension
        num_classes: Output classes for segmentation head
        first_stage: If True, skip Patch Expanding (bottleneck→first decoder)
    """

    def __init__(
        self,
        dim: int,
        skip_dim: int,
        K: int,
        d_state: int = 16,
        num_classes: int = 1,
        first_stage: bool = False,
    ):
        super().__init__()
        self.first_stage = first_stage

        if not first_stage:
            self.patch_expand = PatchExpanding(dim, skip_dim)

        # After optional PE + skip fusion, MS-CCSM operates at skip_dim channels
        self.ms_ccsm = MSCCSM(dim=skip_dim, K=K, d_state=d_state)
        self.cluster_gate = ClusterGate()

        # Fusion: global context (skip_dim) + gated skip (skip_dim) → skip_dim
        self.fuse = nn.Conv2d(skip_dim * 2, skip_dim, 1)

        # Segmentation head
        self.seg_head = nn.Conv2d(skip_dim, num_classes, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_skip: torch.Tensor,
        alpha: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C_in, H, W)  previous decoder output
            x_skip: (B, C_skip, H, W)  encoder skip features
            alpha, beta: gating params from MS-CCSM (optional, auto-used if None)
        Returns:
            p_i: (B, num_classes, H, W)  segmentation prediction at this stage
            x_out: (B, C_skip, H, W)  features for next decoder stage
        """
        # ── Patch Expanding (skip for first stage) ──
        if not self.first_stage:
            x = self.patch_expand(x)

        # ── MS-CCSM with centroid extraction ──
        f_out, centroids, centroid_weights = self.ms_ccsm(x, return_params=True)

        # ── Cluster-Gate: gate skip features using MS-CCSM centroids ──
        x_skip_gated = self.cluster_gate(
            x_skip, centroids, centroid_weights,
            alpha=self.ms_ccsm.ccsm.alpha if alpha is None else alpha,
            beta=self.ms_ccsm.ccsm.beta if beta is None else beta,
        )

        # ── Fuse and predict ──
        fused = self.fuse(torch.cat([f_out, x_skip_gated], dim=1))
        p_i = self.seg_head(fused)

        return p_i, fused


class MambaDecoder(nn.Module):
    """4-stage M2SCAN decoder.

    Stage 4 (H/32, K=2) → Stage 3 (H/16, K=4) → Stage 2 (H/8, K=6) → Stage 1 (H/4, K=8)

    Args:
        dims: Encoder channel dimensions [64, 128, 256, 512]
        K_values: Centroids per stage [8, 6, 4, 2] (S1→S4)
        d_state: Mamba state dimension
        num_classes: Output classes
    """

    def __init__(
        self,
        dims: list[int] | None = None,
        K_values: list[int] | None = None,
        d_state: int = 16,
        num_classes: int = 1,
    ):
        super().__init__()
        if dims is None:
            dims = [64, 128, 256, 512]
        if K_values is None:
            K_values = [8, 6, 4, 2]

        self.dims = dims
        self.K_values = K_values

        # Build stages
        # reversed: bottleneck (512) → S4(K=2, dim=512/skip=512) → S3(K=4, dim=256) → ...
        decoder_dims = list(reversed(dims))   # [512, 256, 128, 64]
        decoder_K = list(reversed(K_values))   # [2, 4, 6, 8]

        self.stages = nn.ModuleList()
        for i in range(len(decoder_dims)):
            first = (i == 0)  # Stage 4: no Patch Expanding
            stage = DecoderStage(
                dim=decoder_dims[i],
                skip_dim=decoder_dims[i],   # skip has same channels (matching encoder)
                K=decoder_K[i],
                d_state=d_state,
                num_classes=num_classes,
                first_stage=first,
            )
            self.stages.append(stage)

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 512, H/32, W/32)  bottleneck output
            skips: [X1, X2, X3, X4]  encoder skip features
        Returns:
            predictions: [p4, p3, p2, p1]  stage outputs (low to high resolution)
        """
        predictions = []
        # skips are [X1(H/4), X2(H/8), X3(H/16), X4(H/32)]
        # decoder processes: S4→X4, S3→X3, S2→X2, S1→X1
        skips_rev = list(reversed(skips))  # [X4, X3, X2, X1]

        for i, stage in enumerate(self.stages):
            p_i, x = stage(x, skips_rev[i])
            predictions.append(p_i)

        return predictions  # [p4, p3, p2, p1]
