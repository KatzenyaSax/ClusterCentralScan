"""
Cluster-Centric Scanning Module (CCSM).

Core module that:
  1. Feature Aggregating (FA): learns K semantic centroids from H×W pixels
  2. Mamba S6 Scan: global reasoning over the K centroids only
  3. Score Diffusing (SD): diffuses centroid-level context back to all pixels

References:
  C2SSM: "Scan Clusters, Not Pixels" (Wu et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCSM(nn.Module):
    """Cluster-Centric Scanning Module.

    Args:
        dim: Input / output channels
        K: Number of cluster centroids (per decoder stage: S4=2, S3=4, S2=6, S1=8)
        d_state: Mamba S6 state dimension
        d_conv: DWConv kernel size for pre-clustering spatial mix
        expand_ratio: MLP expansion ratio
    """

    def __init__(
        self,
        dim: int,
        K: int = 4,
        d_state: int = 16,
        d_conv: int = 3,
        expand_ratio: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.K = K
        inner_dim = int(dim * expand_ratio)

        # ── Pre-clustering channel projection ──
        self.pre_proj = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
        )
        self.dwconv = nn.Conv2d(dim, dim, d_conv, padding=d_conv // 2, groups=dim)
        self.norm_in = nn.LayerNorm(dim)

        # ── Learnable gating parameters (α, β) ──
        self.alpha = nn.Parameter(torch.ones(1) * 2.0)   # similarity sharpness
        self.beta = nn.Parameter(torch.zeros(1))           # activation bias

        # ── Centroid refinement MLPs ──
        self.to_v = nn.Linear(dim, dim, bias=False)       # centroid → value
        self.to_f_hat = nn.Linear(dim, dim, bias=False)    # pixel feature → query

        # ── Mamba S6 over centroids ──
        self.s6 = _S6Block(dim, d_state)

        # ── Output projection ──
        self.out_proj = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

        # ── Gate branch ──
        self.gate = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim),
        )

    def forward(
        self, x: torch.Tensor, return_params: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
            return_params: if True, also return (centroids, weights) for Cluster-Gate
        Returns:
            F_out: (B, C, H, W) global-context-enhanced features
            (Ĉ, W): optional tuple of centroids (B, K, C) and centroid weights (B, K, C)
        """
        B, C, H, W = x.shape
        device = x.device

        # ── 1. Pre-clustering channel projection ──
        residual = x
        x_t = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_t = self.pre_proj(x_t)
        x_t = x_t.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_t = self.dwconv(x_t) + x_t
        x_t = self.norm_in(x_t.permute(0, 2, 3, 1))  # (B, H, W, C)
        F_d = x_t.permute(0, 3, 1, 2)                 # (B, C, H, W)

        # ── 2. Feature Aggregating ──
        centroids = self._feature_aggregating(F_d)
        # centroids: (B, K, C)

        # ── 3. Mamba S6 scan over centroids ──
        W_centroids = self.s6(centroids)
        # W_centroids: (B, K, C)

        # ── 4. Score Diffusing ──
        # Compute assignment probabilities between each pixel and each centroid
        F_d_flat = F_d.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C), N=H*W
        centroids_norm = F.normalize(centroids, dim=-1)
        pixels_norm = F.normalize(F_d_flat, dim=-1)

        sim = torch.bmm(pixels_norm, centroids_norm.transpose(1, 2))  # (B, N, K)
        assignment = F.softmax(self.alpha * sim + self.beta, dim=-1)   # (B, N, K)

        # Diffuse centroid weights to pixels
        W_pixel = torch.bmm(assignment, W_centroids)                    # (B, N, C)
        W_pixel = W_pixel.reshape(B, H, W, C).permute(0, 3, 1, 2)     # (B, C, H, W)
        F_f = self.out_norm(
            self.out_proj(W_pixel.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            .permute(0, 2, 3, 1)
        )  # (B, H, W, C)

        # ── 5. Gate modulation ──
        gate_val = torch.sigmoid(
            self.gate(residual.permute(0, 2, 3, 1))
        )  # (B, H, W, C)
        F_f = gate_val * F_f

        F_out = F_f.permute(0, 3, 1, 2)  # (B, C, H, W)

        if return_params:
            return F_out, centroids, W_centroids
        return F_out

    def _feature_aggregating(self, F_d: torch.Tensor) -> torch.Tensor:
        """Learn K refined cluster centroids from feature pixels.

        Args:
            F_d: (B, C, H, W)
        Returns:
            centroids: (B, K, C) refined centroids
        """
        B, C, H, W = F_d.shape
        N = H * W
        K = self.K
        K = min(K, int(N ** 0.5))  # guard against K > sqrt(N)

        # ── Initialize centroids: uniform sampling ──
        F_flat = F_d.reshape(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # Select K evenly-spaced pixel positions (plus kNN averaging)
        indices = torch.linspace(0, N - 1, K, dtype=torch.long, device=F_d.device)
        c_init = F_flat[:, indices, :]  # (B, K, C)

        # ── Compute cosine similarity distribution ──
        c_norm = F.normalize(c_init, dim=-1)       # (B, K, C)
        f_norm = F.normalize(F_flat, dim=-1)        # (B, N, C)
        sim = torch.bmm(f_norm, c_norm.transpose(1, 2))  # (B, N, K)

        # Normalized similarity (PDF over pixels for each centroid)
        p = F.softmax(sim, dim=1)                    # (B, N, K) — each column sums to 1

        # ── Gated refinement ──
        v = self.to_v(c_init)                        # (B, K, C)
        f_hat = self.to_f_hat(F_flat)                # (B, N, C)

        # Gate: δ(α·p + β), softly select relevant pixels per centroid
        gate_val = torch.sigmoid(self.alpha * p + self.beta)  # (B, N, K)

        # Weighted aggregation: Ĉ_k = (v_k + Σ g_pk · f̂_p) / (1 + Σ g_pk)
        numerator = v + torch.bmm(gate_val.transpose(1, 2), f_hat)  # (B, K, C)
        denominator = 1 + gate_val.sum(dim=1, keepdim=True).transpose(1, 2)  # (B, K, 1)

        centroids = numerator / denominator  # (B, K, C)
        return centroids


class _S6Block(nn.Module):
    """Mamba S6 selective scan block operating on a sequence of centroids.

    This is a simplified pure-PyTorch implementation of the selective
    state-space mechanism from Mamba.

    Args:
        dim: Input/output dimension per centroid
        d_state: State dimension
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        self.dim = dim

        # Input projections (B, C, Δ share base projection)
        self.in_proj = nn.Linear(dim, dim * 3 + d_state, bias=False)

        # Discretization: Δ ← softplus(Linear(x) + bias_param)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        # A is a learnable parameter, discretized per step via Δ
        self.A_log = nn.Parameter(torch.log(torch.ones(dim, d_state) * 0.5))
        # D skip connection
        self.D = nn.Parameter(torch.ones(dim))

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)  where L = K (number of centroids)
        Returns:
            out: (B, L, C)
        """
        B, L, C = x.shape
        A = -torch.exp(self.A_log)                     # (C, N)  → stays positive

        # Input projection
        proj = self.in_proj(x)                          # (B, L, 3C + N)
        z = proj[..., :C]
        b = proj[..., C:2 * C]
        c = proj[..., 2 * C:3 * C]
        delta = torch.nn.functional.softplus(
            self.dt_proj(proj[..., 3 * C:])
        )                                               # (B, L, C)

        # Discretize A: Ā = exp(Δ·A)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, C, N)

        # B̄ = Δ·b
        B_bar = delta.unsqueeze(-1) * b.unsqueeze(-1)   # (B, L, C, N)

        # Selective scan (recurrent form)
        h = torch.zeros(B, C, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        for t in range(L):
            A_t = A_bar[:, t, :, :]                     # (B, C, N)
            B_t = B_bar[:, t, :, :]                     # (B, C, N)
            x_t = x[:, t, :].unsqueeze(-1)               # (B, C, 1)
            h = A_t * h + B_t * x_t                     # (B, C, N)
            y_t = (h * c[:, t, :].unsqueeze(-1)).sum(dim=-1)  # (B, C)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)                   # (B, L, C)
        y = y + self.D * x                              # skip connection

        # Gating (z is the gate signal)
        out = y * torch.nn.functional.silu(z)
        out = self.norm(self.out_proj(out))
        return out
