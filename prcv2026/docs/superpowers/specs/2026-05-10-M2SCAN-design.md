# M2SCAN Design Specification

**Date:** 2026-05-10
**Status:** Draft
**Target:** Medical Image Segmentation

---

## 1. Overview

M2SCAN is a U-Net variant for medical image segmentation that replaces the decoder's convolution-based attention with a multi-scale cluster-centric scanning mechanism. The architecture unifies Mamba (SS2D/S6) across both encoder and decoder for efficient global context modeling.

### 1.1 Core Idea

Replace EMCAD decoder's **MSCAM** (Multi-scale Convolutional Attention Module) with **MS-CCSM** (Multi-scale Cluster-Centric Scanning Module), and replace PVTv2 encoder with a **Mamba SS2D encoder**.

| Component | EMCAD (baseline) | M2SCAN (ours) |
|-----------|-----------------|-------------------|
| Encoder | PVTv2 (Transformer) | SS2D Mamba (linear complexity) |
| Decoder core | MSCAM (DW Conv Attention) | MS-CCSM (Cluster-centric Scanning) |
| Skip gate | LGAG | LGAG (kept) |
| Upsample | EUCB | EUCB (kept) |
| Multi-stage loss | Yes | Yes (kept) |

### 1.2 Innovation Points

1. **Multi-scale Cluster-Centric Scanning:** CCSM extended to multiple scales — different decoder levels use different numbers of cluster centroids (K=[2,4,6,8]), enabling hierarchical global reasoning from coarse semantic to fine structural levels.

2. **Full Mamba Architecture:** SS2D (4-direction selective scan) in encoder for spatial mixing + S6 in decoder for centroid-level global reasoning. First U-Net with end-to-end Mamba spatial modeling.

3. **Efficient Decoder:** Retains EMCAD's LGAG gating and EUCB upsampling for skip connection filtering and efficient feature expansion.

---

## 2. Architecture

### 2.1 Full Network

```
Input (H×W×3)
    │
    ▼
┌────────────────────────────────────┐
│  Conv Stem (stride=4, 3→64)        │
└────────────────────────────────────┘
    │
    ├── Stage 1: SS2D Block ×2  ──── X1 (64ch, H/4)
    │       ↓ Patch Merge
    ├── Stage 2: SS2D Block ×2  ──── X2 (128ch, H/8)
    │       ↓ Patch Merge
    ├── Stage 3: SS2D Block ×8  ──── X3 (256ch, H/16)
    │       ↓ Patch Merge
    ├── Stage 4: SS2D Block ×2  ──── X4 (512ch, H/32)
    │
    ▼
┌────────────────────────────────────┐
│  Bottleneck: SS2D Block ×2 (512ch) │
└────────────────────────────────────┘
    │
    ▼  D4
┌────────────────────────────────────┐
│  Decoder Stage 4:                  │
│    EUCB ↑2× → LGAG(X4) →           │
│    MS-CCSM(K=2) → SegHead → p4     │
│                    ↓                │
│  Decoder Stage 3:                  │
│    EUCB ↑2× → LGAG(X3) →           │
│    MS-CCSM(K=4) → SegHead → p3     │
│                    ↓                │
│  Decoder Stage 2:                  │
│    EUCB ↑2× → LGAG(X2) →           │
│    MS-CCSM(K=6) → SegHead → p2     │
│                    ↓                │
│  Decoder Stage 1:                  │
│    EUCB ↑2× → LGAG(X1) →           │
│    MS-CCSM(K=8) → SegHead → p1     │
└────────────────────────────────────┘
    │
    ▼
Final Segmentation Map (H×W×#classes)
```

### 2.2 Encoder: Mamba SS2D Blocks

Each SS2D Block consists of:
- LayerNorm → SS2D (4-directional selective scan) → LayerNorm → FFN (MLP)

The 4-directional scan unfolds the 2D feature map into sequences along 4 paths (top-left→bottom-right, bottom-right→top-left, top-right→bottom-left, bottom-left→top-right), applies Mamba S6 to each, then merges. This provides global receptive field at linear complexity.

Channel progression: 64 → 128 → 256 → 512
Block distribution: [2, 2, 8, 2] (Stage 3 gets more blocks as H/16 is the efficiency sweet spot)
Downsampling: Patch Merge (2×2→1 with channel doubling)

### 2.3 Decoder: MS-CCSM + EMCAD Components

Each decoder stage: EUCB → LGAG → MS-CCSM(K) → SegHead

**Kept from EMCAD:**
- **EUCB:** Upsample(×2) → 3×3 DWConv → BN → ReLU → 1×1 Conv. Efficient upsampling.
- **LGAG:** Large-kernel grouped attention gate. Filters skip connection features before fusion with upsampled decoder features.
- **SegHead:** 1×1 Conv mapping to #classes.
- **Multi-stage loss:** L = Σ L(p_i) + L(Σ p_i), same MUTATION-style combinatorial loss.

**Replaces MSCAM:**
- **MS-CCSM(K):** CCSM(K centroids) + SCFM in parallel, outputs fused feature map.

### 2.4 MS-CCSM Internal Structure

```
F_in ──┬──► CCSM(K) ──► global_context ──┐
       │                                   ├──► + ──► F_out
       └──► SCFM ──────► detail_features ─┘
```

- **CCSM(K):** Distills H×W features into K cluster centroids via similarity-based soft assignment; runs Mamba S6 only on the K centroids for global reasoning; diffuses centroid-level context back to all pixels via learned similarity distribution. K controls granularity.
- **SCFM:** Parallel spatial + channel convolution attention path that preserves high-frequency details (textures, boundaries) potentially lost in clustering.

### 2.5 Multi-Scale K Strategy

| Decoder Stage | Resolution (vs input) | K | Rationale |
|--------------|----------------------|---|-----------|
| Stage 4 | H/16 | 2 | Coarse: foreground vs background |
| Stage 3 | H/8 | 4 | Medium granularity |
| Stage 2 | H/4 | 6 | Finer details |
| Stage 1 | H/2 | 8 | Finest: small structures, boundaries |

Higher resolution → more centroids needed to represent the richer spatial variation.

---

## 3. Implementation Plan

### 3.1 File Structure

```
prcv2026/
├── models/
│   ├── __init__.py
│   ├── encoder.py          # Mamba SS2D encoder + bottleneck
│   ├── decoder.py          # EMCAD-style decoder with MS-CCSM
│   ├── ms_ccsm.py          # MS-CCSM module (CCSM + SCFM)
│   ├── ccsman.py           # Cluster-Centric Scanning Module
│   ├── scfm.py             # Spatial-Channel Feature Modulator
│   ├── lgag.py             # Large-kernel Grouped Attention Gate
│   ├── eucb.py             # Efficient Up-Convolution Block
│   └── mambamedseg.py      # Full model assembly
├── configs/
│   └── mambamedseg.yaml    # Model hyperparameters
├── train.py                # Training entry point
├── scripts/
│   └── draw_architecture.py
└── docs/superpowers/specs/
    └── 2026-05-10-mambamedseg-design.md
```

### 3.2 Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Encoder blocks | [2, 2, 8, 2] | Mamba SS2D blocks per stage |
| Channels | [64, 128, 256, 512] | Per stage |
| K values | K=[8,6,4,2] for Stage1→Stage4 | Stage1(high-res)=8, Stage4(low-res)=2 |
| Expansion factor (MLP) | 4 | Inside SS2D blocks |
| S6 d_state | 16 | Mamba state dimension |
| Estimated params | ~10-12M | vs EMCAD-B2 26.8M |

### 3.3 Dependencies

- PyTorch >= 2.0
- mamba-ssm (selective scan CUDA kernel) or pure-PyTorch SS2D
- timm (optional, for training utilities)

### 3.4 Training Strategy

Follow EMCAD's training protocol:
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Epochs: 200 (binary), 300-400 (multi-organ)
- Batch size: 16 (binary), 6-12 (multi-organ)
- Input size: 352×352 (binary), 224×224 (multi-organ)
- Multi-scale training: {0.75, 1.0, 1.25}
- Loss: weighted BCE + weighted IoU (binary); CrossEntropy + Dice (multi-class)
- Gradient clip: 0.5

### 3.5 Evaluation

- Binary segmentation: 10 datasets (polyp ×5, skin lesion ×2, cell ×2, breast cancer)
- Multi-organ: Synapse, ACDC
- Metrics: DICE, HD95, mIoU
- Efficiency: #Params, #FLOPs (measured at 224×224 and 256×256)

---

## 4. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| SS2D CUDA kernel unavailable | Implement pure-PyTorch fallback using 4× unfold + scan |
| CCSM with K > H×W degenerates | Guard: K = min(K, sqrt(H×W)) |
| Training instability with Mamba | Use same LR/warmup as VMamba, gradient clip |
| SCFM adds redundant params | Ablation to verify necessity; can be made optional |

---

## 5. Ablation Plan

1. **K values:** Compare [2,4,6,8] vs [4,4,4,4] (single-scale CCSM) vs [1,2,4,8]
2. **SCFM:** With vs without (verify detail compensation effect)
3. **LGAG:** With vs without (verify skip gate benefit)
4. **Encoder depth:** [2,2,8,2] vs [2,2,4,2] vs [2,2,12,2]
5. **vs EMCAD baseline:** Same encoder, compare MSCAM vs MS-CCSM decoder
