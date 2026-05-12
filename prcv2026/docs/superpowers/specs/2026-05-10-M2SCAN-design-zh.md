# M2SCAN 设计规格书

**日期：** 2026-05-10（更新于 2026-05-12）
**状态：** 草案
**目标：** 医学图像分割

---

## 1. 概述

M2SCAN（Multi-scale Mamba Scanning over Cluster Centers）是一种用于医学图像分割的 U-Net 变体。它将解码器中的卷积注意力替换为多尺度聚类中心扫描机制，并在编码器和解码器中统一使用 Mamba（SS2D/S6）实现高效的全局上下文建模。跳跃连接门控也被改造为基于聚类质心的全局门控。

### 1.1 核心思路

用 **MS-CCSM**（多尺度聚类中心扫描模块）替换 EMCAD 解码器中的 **MSCAM**（多尺度卷积注意力模块）；用 **Cluster-Gate**（质心复用门控）替换 **LGAG**（大核分组注意力门控）；用 **Patch Expanding** 替换 **EUCB**（高效上卷积模块）；用 **Mamba SS2D 编码器**替换 PVTv2 编码器。

| 组件 | EMCAD（基线） | M2SCAN（本文） |
|------|-------------|--------------|
| 编码器 | PVTv2 (Transformer) | SS2D Mamba（线性复杂度） |
| 解码器核心 | MSCAM（深度可分离卷积注意力） | MS-CCSM（聚类中心扫描） |
| 跳跃连接门控 | LGAG（大核分组注意力） | Cluster-Gate（质心复用门控） |
| 上采样 | EUCB（高效上卷积） | Patch Expanding（与 Patch Merge 对称） |
| 多级损失 | 是 | 是（保留） |

### 1.2 创新点

1. **多尺度聚类中心扫描：** 将 CCSM 扩展到多尺度——不同解码器层级使用不同数量的聚类质心（K=[2,4,6,8]），实现从粗粒度语义到细粒度结构的层次化全局推理。

2. **全 Mamba 架构：** 编码器使用 SS2D（四方向选择性扫描）进行空间混合，解码器使用 S6 进行质心级全局推理。首个在 U-Net 编解码全链路统一使用 Mamba 空间建模的工作。

3. **质心复用门控（Cluster-Gate）：** 将 MS-CCSM 产出的聚类质心复用于跳跃连接门控——质心不仅用于全局上下文扩散，还指导编码器特征的智能筛选。同一组质心，双重用途。零额外参数替换 LGAG 的大核卷积。

4. **对称编解码设计：** 编码器使用 Patch Merge 下采样，解码器使用 Patch Expanding 上采样，两者互为逆操作，形成对称的层级特征变换。仅保留 EMCAD 的级联多级输出和组合损失作为通用训练策略。

---

## 2. 架构设计

### 2.1 整体网络结构

```
输入 (H×W×3)
    │
    ▼
┌────────────────────────────────────┐
│  Conv Stem (stride=4, 3→64)        │
└────────────────────────────────────┘
    │
    ├── Stage 1: VSS Block ×2  ──── X1 (64ch, H/4)
    │       ↓ Patch Merge
    ├── Stage 2: VSS Block ×2  ──── X2 (128ch, H/8)
    │       ↓ Patch Merge
    ├── Stage 3: VSS Block ×8  ──── X3 (256ch, H/16)
    │       ↓ Patch Merge
    ├── Stage 4: VSS Block ×2  ──── X4 (512ch, H/32)
    │
    ▼
┌────────────────────────────────────┐
│  Bottleneck: VSS Block ×2 (512ch) │
└────────────────────────────────────┘
    │
    ▼  (H/32, 512ch)
┌──────────────────────────────────────────────┐
│  解码器 Stage 4:                              │
│    MS-CCSM(K=2) → Cluster-Gate(X4) → p4     │
│                    ↓                         │
│             Patch Exp ↑2× (H/32 → H/16)      │
│  解码器 Stage 3:                              │
│    MS-CCSM(K=4) → Cluster-Gate(X3) → p3     │
│                    ↓                         │
│             Patch Exp ↑2× (H/16 → H/8)       │
│  解码器 Stage 2:                              │
│    MS-CCSM(K=6) → Cluster-Gate(X2) → p2     │
│                    ↓                         │
│             Patch Exp ↑2× (H/8 → H/4)        │
│  解码器 Stage 1:                              │
│    MS-CCSM(K=8) → Cluster-Gate(X1) → p1     │
└──────────────────────────────────────────────┘
    │
    ▼
最终分割图 (H×W×#classes)
```

### 2.2 编码器：Mamba SS2D 模块

每个 VSS Block 由以下组成：
- LayerNorm → SS2D（四方向选择性扫描）→ LayerNorm → FFN（MLP）

四方向扫描将 2D 特征图按 4 条路径展开为序列（左上→右下、右下→左上、右上→左下、左下→右上），对每条路径分别应用 Mamba S6，然后合并。这在线性复杂度下提供了全局感受野。

- 通道数：64 → 128 → 256 → 512
- Block 分布：[2, 2, 8, 2]（Stage 3 最深，因为 H/16 分辨率是感受野与计算量的最优平衡点）
- 下采样：Patch Merge（2×2→1，通道翻倍）

### 2.3 解码器：MS-CCSM + Cluster-Gate

解码器共 4 级。Stage 4 直接取瓶颈输出（H/32），无需上采样（Xi=X4 也在 H/32）。后续每一级先用 Patch Expanding 翻倍分辨率，再匹配对应的编码器跳跃连接。

每个解码器层级的流程：

```
  上一级输出
       │
       ▼
  ┌──────────────┐   (Stage 4 跳过此步)
  │ Patch Expand │
  └──────┬───────┘
         │
         ▼
  ┌────────────────────────┐
  │     MS-CCSM(K)         │
  │  (内部产出质心参数)      │
  └───────┬────────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
  F_out   Cluster-Gate(Xi)
    │           │
    │           ▼
    │       Xi' (门控后)
    │           │
    └─────┬─────┘
          ▼
   融合 + SegHead → p_i
```

**Patch Expanding（上采样模块）：**
- Linear(C → 2C) → PixelShuffle(H×W → 2H×2W) → Linear(2C → C/2)
- 与编码器的 Patch Merge 互为逆操作，形成对称编解码设计
- 来自 Swin Transformer / VMamba 系列的通用上采样范式

**MS-CCSM(K)：**
- 前置执行，输出特征图 F_out 走主路径；内部产出的质心参数 Ĉ, W 送给 Cluster-Gate 复用（代码内硬编码传递，非架构级数据流）

**Cluster-Gate（质心复用门控）：**
- 输入 Xi（编码器跳跃连接），内部使用 MS-CCSM 的质心参数 Ĉ, W 计算门控系数
- 不产生新特征，只输出 Xi' = Xi ⊙ gate，筛掉噪声像素

对跳跃连接特征 Xi 的每个空间位置 p：
1. 计算 p 与 K 个质心的余弦相似度：`sim(Xi_p, ĉₖ)`
2. Softmax 得到归属概率：`α_{p,k} = softmax(α·sim + β)`（α, β 复用自 MS-CCSM）
3. 质心全局权重加权求和：`g_p = Σₖ α_{p,k} · wₖ`（wₖ 来自 Mamba S6）
4. Sigmoid 门控：`Xi'_p = Xi_p · σ(g_p)`

**直觉：** Mamba S6 输出的 wₖ 代表"第 k 个聚类有多重要"。如果一个 skip 像素属于重要质心，门控权重就大，信息被保留；反之被抑制。门控获得了整张图的全局语义理解。

**SegHead（分割头）：** 1×1 卷积，将特征通道映射到类别数。

**多级损失：** L = Σ L(p_i) + L(Σ p_i)，沿用 EMCAD 的 MUTATION 组合损失。

### 2.4 Cluster-Gate 与 LGAG 对比

| 维度 | 原 LGAG | Cluster-Gate |
|------|--------|-------------|
| 感受野 | 大核卷积（局部） | 全局（质心代表整图） |
| 门控依据 | 卷积特征响应 | 聚类归属 + 质心全局重要性 |
| 额外参数 | 大核卷积权重 | 几乎为零（复用 Ĉ, α, β, W） |
| 与 MS-CCSM 关系 | 独立模块 | 紧密耦合，共享质心 |

### 2.5 MS-CCSM 内部结构

```
F_in ──┬──► CCSM(K) ──► 全局上下文 F_out, 质心 Ĉ, 权重 W ──┐
       │                                                      ├──► 融合 ──► 最终输出
       └──► SCFM ──────► 细节特征 ────────────────────────────┘
```

- **CCSM(K)：** 将 H×W 个像素特征通过相似度软分配提炼为 K 个聚类质心 Ĉ；仅在这 K 个质心上运行 Mamba S6 进行全局推理，输出质心权重 W；将质心级全局上下文通过相似度分布扩散回所有像素。K 控制聚类粒度。**同时输出 Ĉ 和 W 供 Cluster-Gate 复用。**
- **SCFM（空间-通道特征调制器）：** 并行的空间注意力 + 通道注意力纯卷积路径，保留聚类过程中可能丢失的高频细节（纹理、边界）。

### 2.6 多尺度 K 策略与分辨率对齐

| 解码器层级 | 分辨率 | K | 跳跃连接 | 含义 |
|-----------|--------|---|---------|------|
| Stage 4 | H/32 | 2 | X4 (H/32, 512ch) | 粗粒度：前景 vs 背景 |
| Stage 3 | H/16 | 4 | X3 (H/16, 256ch) | 中等粒度 |
| Stage 2 | H/8 | 6 | X2 (H/8, 128ch) | 较细粒度 |
| Stage 1 | H/4 | 8 | X1 (H/4, 64ch) | 最细粒度：小结构、边界 |

- Stage 4 直接从瓶颈（H/32）进入 MS-CCSM，无需 Patch Expanding，因为 X4 也在 H/32
- Stage 3–1 先 Patch Expanding 翻倍分辨率，再匹配对应层的跳跃连接
- K 值随分辨率升高递增：分辨率越高→需要更多质心来表示更丰富的空间变化

---

## 3. 实现计划

### 3.1 文件结构

```
prcv2026/
├── models/
│   ├── __init__.py
│   ├── encoder.py          # Mamba SS2D 编码器 + 瓶颈层
│   ├── decoder.py          # 解码器（含 MS-CCSM + Cluster-Gate + SegHead）
│   ├── ms_ccsm.py          # MS-CCSM 模块（CCSM + SCFM）
│   ├── ccsman.py           # 聚类中心扫描模块
│   ├── scfm.py             # 空间-通道特征调制器
│   ├── cluster_gate.py     # 质心复用门控（替代 LGAG）
│   ├── patch_expanding.py  # Patch Expanding 上采样模块
│   └── m2scan.py           # 完整模型组装
├── configs/
│   └── m2scan.yaml         # 模型超参数
├── train.py                # 训练入口
├── scripts/
│   └── draw_architecture.py
└── docs/superpowers/specs/
    ├── 2026-05-10-M2SCAN-design.md
    └── 2026-05-10-M2SCAN-design-zh.md
```

### 3.2 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| 编码器 Blocks | [2, 2, 8, 2] | 每个 Stage 的 VSS Block 数量 |
| 通道数 | [64, 128, 256, 512] | 各 Stage 通道数 |
| K 值 | S1→S4: [8,6,4,2] | Stage1(H/4)=8, Stage2(H/8)=6, Stage3(H/16)=4, Stage4(H/32)=2 |
| MLP 扩展因子 | 4 | VSS Block 内部 |
| S6 d_state | 16 | Mamba 状态空间维度 |
| 预估参数量 | ~10M | vs EMCAD-B2 的 26.8M；比原方案更低（移除 LGAG 大核卷积） |

### 3.3 依赖

- PyTorch >= 2.0
- mamba-ssm（选择性扫描 CUDA 算子）或纯 PyTorch SS2D 回退实现
- timm（可选，用于训练工具）

### 3.4 训练策略

沿用 EMCAD 的训练协议：
- 优化器：AdamW，lr=1e-4，weight_decay=1e-4
- 训练轮数：200（二分类），300–400（多器官）
- Batch size：16（二分类），6–12（多器官）
- 输入尺寸：352×352（二分类），224×224（多器官）
- 多尺度训练：{0.75, 1.0, 1.25}
- 损失函数：加权 BCE + 加权 IoU（二分类）；交叉熵 + Dice（多类别）
- 梯度裁剪：0.5

### 3.5 评估方案

- 二分类分割：10 个数据集（息肉×5、皮肤病变×2、细胞×2、乳腺癌）
- 多器官分割：Synapse、ACDC
- 指标：DICE、HD95、mIoU
- 效率：参数量、FLOPs（在 224×224 和 256×256 分辨率下测量）

---

## 4. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| SS2D CUDA 算子不可用 | 实现纯 PyTorch 回退：4 方向 unfold + scan |
| CCSM 在 K > H×W 时退化 | 保护逻辑：K = min(K, sqrt(H×W)) |
| Mamba 训练不稳定 | 沿用 VMamba 的学习率和 warmup 策略，梯度裁剪 |
| SCFM 增加冗余参数 | 通过消融实验验证必要性；可设为可选项 |
| Cluster-Gate 质心质量敏感 | 质心由 MS-CCSM 的 FA 精炼，梯度可通过门控反向传播至 FA，形成闭环优化 |

---

## 5. 消融实验计划

1. **K 值消融：** 对比 [2,4,6,8] vs [4,4,4,4]（单尺度 CCSM）vs [1,2,4,8]
2. **SCFM 消融：** 有 vs 无（验证细节补偿作用）
3. **Cluster-Gate 消融：** Cluster-Gate vs 原 LGAG vs 无门控（验证质心复用门控贡献）
4. **编码器深度：** [2,2,8,2] vs [2,2,4,2] vs [2,2,12,2]
5. **vs EMCAD 基线：** 相同编码器下，对比 MSCAM+LGAG vs MS-CCSM+Cluster-Gate 解码器
6. **质心共享消融：** Cluster-Gate 使用独立质心 vs 复用 MS-CCSM 质心
