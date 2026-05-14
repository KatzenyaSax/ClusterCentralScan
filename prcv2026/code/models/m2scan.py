"""
M2SCAN: Multi-scale Mamba Scanning over Cluster Centers.

Full model assembly: Mamba VSS Encoder + MS-CCSM Decoder
for medical image segmentation.

Usage:
    model = M2SCAN(num_classes=1)
    out = model(x)              # final prediction only
    preds = model(x, return_all=True)  # all stage predictions for multi-stage loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MambaEncoder
from .decoder import MambaDecoder


class M2SCAN(nn.Module):
    """M2SCAN: Multi-scale Mamba Scanning over Cluster Centers.

    Args:
        in_chans: Input channels (3 for RGB, 1 for grayscale)
        num_classes: Output segmentation classes
        depths: VSS Blocks per encoder stage [2, 2, 8, 2]
        dims: Channels per stage [64, 128, 256, 512]
        K_values: Centroids per decoder stage [8, 6, 4, 2] (S1→S4)
        d_state: Mamba S6 state dimension
        mlp_ratio: FFN expansion ratio in VSS blocks
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        K_values: list[int] | None = None,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 8, 2]
        if dims is None:
            dims = [64, 128, 256, 512]
        if K_values is None:
            K_values = [8, 6, 4, 2]

        self.encoder = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
        )
        self.decoder = MambaDecoder(
            dims=dims,
            K_values=K_values,
            d_state=d_state,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.dims = dims

    def forward(
        self, x: torch.Tensor, return_all: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)  input image
            return_all: if True, return dict with all stage preds
        Returns:
            If return_all=False: (B, num_classes, H, W) final prediction
            If return_all=True: dict with keys 'p1'..'p4', 'final'
        """
        # Encode
        bottleneck, skips = self.encoder(x)

        # Decode → [p4, p3, p2, p1]
        preds = self.decoder(bottleneck, skips)

        # preds[3] = p1 (H/4 resolution), upsample to full
        final_pred = preds[-1]  # p1 at H/4
        final_pred = F.interpolate(
            final_pred, size=x.shape[2:], mode='bilinear', align_corners=False
        )

        if return_all:
            result = {'final': final_pred}
            for i, p in enumerate(preds):
                result[f'p{4 - i}'] = p  # p4, p3, p2, p1
            return result
        return final_pred


def compute_multi_stage_loss(
    preds: dict[str, torch.Tensor],
    target: torch.Tensor,
    loss_fn: callable,
) -> torch.Tensor:
    """Compute MUTATION-style multi-stage combinatorial loss.

    L = L(p1) + L(p2) + L(p3) + L(p4) + L(p1+p2+p3+p4)

    Args:
        preds: dict with keys 'p1'..'p4' at various resolutions
        target: (B, 1, H, W) ground truth at input resolution
        loss_fn: callable(output, target) → scalar loss
    Returns:
        Total loss scalar
    """
    total_loss = 0.0
    target_size = target.shape[2:]

    # Resize all preds to target size
    resized = {}
    for key in ['p1', 'p2', 'p3', 'p4']:
        p = F.interpolate(preds[key], size=target_size, mode='bilinear',
                          align_corners=False)
        resized[key] = p
        total_loss += loss_fn(p, target)

    # Combinatorial: mean of all 4 predictions
    combined = (resized['p1'] + resized['p2'] + resized['p3'] + resized['p4']) / 4.0
    total_loss += loss_fn(combined, target)

    return total_loss
