"""
Cluster-Gate: centroid-reuse gating for skip connections.

Replaces EMCAD's LGAG. Uses the centroids and weights from MS-CCSM
to globally reason about which skip-connection features to keep.
Zero extra learnable parameters beyond what MS-CCSM already provides.

The gate operates as:
  g_p = Σ_k α_{p,k} · w_k    (expected centroid weight for pixel p)
  Xi' = Xi ⊙ σ(g_p)          (sigmoid gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterGate(nn.Module):
    """Skip-connection gating via cluster-centroid reuse.

    This module has NO learnable parameters of its own — it purely
    computes using centroids and weights passed from MS-CCSM.
    """

    def __init__(self):
        super().__init__()
        # No parameters — pure functional computation

    def forward(
        self,
        x_skip: torch.Tensor,
        centroids: torch.Tensor,
        centroid_weights: torch.Tensor,
        alpha: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_skip: (B, C, H, W)  encoder skip features Xi
            centroids: (B, K, C)  from MS-CCSM's FA step
            centroid_weights: (B, K, C)  from MS-CCSM's S6 step
            alpha, beta: optional gating sharpness parameters
        Returns:
            Xi': (B, C, H, W)  gated skip features
        """
        B, C, H, W = x_skip.shape
        K = centroids.shape[1]
        device = x_skip.device

        # Default gating parameters if not provided
        if alpha is None:
            alpha = torch.tensor(2.0, device=device)
        if beta is None:
            beta = torch.tensor(0.0, device=device)

        # Flatten skip features
        Xi_flat = x_skip.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C), N=H*W

        # Cosine similarity between skip pixels and centroids
        centroids_norm = F.normalize(centroids, dim=-1)
        pixels_norm = F.normalize(Xi_flat, dim=-1)
        sim = torch.bmm(pixels_norm, centroids_norm.transpose(1, 2))  # (B, N, K)

        # Assignment probabilities
        assignment = F.softmax(alpha * sim + beta, dim=-1)  # (B, N, K)

        # Gate value = expected centroid weight for each pixel
        g = torch.bmm(assignment, centroid_weights)          # (B, N, C)
        g = g.reshape(B, H, W, C).permute(0, 3, 1, 2)       # (B, C, H, W)

        # Sigmoid gate
        gate = torch.sigmoid(g)
        return x_skip * gate
