"""
M2SCAN training entry point.

Follows EMCAD training protocol:
  - AdamW, lr=1e-4, weight_decay=1e-4
  - Multi-scale training {0.75, 1.0, 1.25}
  - Multi-stage combinatorial loss
  - Gradient clipping 0.5
"""

import os
import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.m2scan import M2SCAN, compute_multi_stage_loss


# ── Loss functions ────────────────────────────────────────────

class BinarySegLoss(nn.Module):
    """Weighted BCE + IoU loss for binary medical segmentation."""

    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        # IoU loss (soft)
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = (pred_sig + target - pred_sig * target).sum(dim=(2, 3))
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1.0 - iou.mean()
        return self.bce_weight * bce + self.iou_weight * iou_loss


# ── Training loop ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 0.5,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        preds = model(images, return_all=True)
        loss = compute_multi_stage_loss(preds, masks, loss_fn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    dice_scores = []
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        preds = model(images, return_all=True)
        loss = compute_multi_stage_loss(preds, masks, loss_fn)
        total_loss += loss.item()

        # DICE for final prediction
        final = torch.sigmoid(preds['final'])
        pred_bin = (final > 0.5).float()
        intersection = (pred_bin * masks).sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (
            pred_bin.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-6
        )
        dice_scores.append(dice.mean().item())

    return {
        'loss': total_loss / len(dataloader),
        'dice': sum(dice_scores) / len(dice_scores),
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='M2SCAN Training')
    parser.add_argument('--config', type=str, default='configs/m2scan.yaml')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model
    model_cfg = cfg['model']
    model = M2SCAN(
        in_chans=model_cfg['in_chans'],
        num_classes=model_cfg['num_classes'],
        depths=model_cfg.get('depths', [2, 2, 8, 2]),
        dims=model_cfg.get('dims', [64, 128, 256, 512]),
        K_values=model_cfg.get('K_values', [8, 6, 4, 2]),
        d_state=model_cfg.get('d_state', 16),
        mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params / 1e6:.2f}M')

    # Optimizer
    train_cfg = cfg['train']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )

    # Loss
    loss_cfg = train_cfg.get('loss', {})
    loss_fn = BinarySegLoss(
        bce_weight=loss_cfg.get('bce_weight', 1.0),
        iou_weight=loss_cfg.get('iou_weight', 1.0),
    )

    # TODO: Replace with actual dataset
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)

    print('Training ready. Replace dataset placeholder to start training.')


if __name__ == '__main__':
    main()
