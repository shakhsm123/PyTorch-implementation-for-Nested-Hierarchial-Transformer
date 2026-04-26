# NeST — Nested Hierarchical Transformer

A from-scratch PyTorch implementation of **NeST** (Nested Hierarchical Transformer), based on the paper:

> *Nested Hierarchical Transformer: Towards Accurate, Data-Efficient and Interpretable Visual Understanding*  
> Zizhao Zhang, Han Zhang, Long Zhao, Ting Chen, Sercan Ö. Arık, Tomas Pfister  
> Google Research, AAAI 2022 — [arXiv:2105.12723](https://arxiv.org/abs/2105.12723)

---

## Results

Training NeST-T from scratch on CIFAR-10 and CIFAR-100 without distillation or pre-training:

| Dataset   | Epochs | Val Accuracy |
|-----------|--------|-------------|
| CIFAR-10  | 50     | ~87%        |
| CIFAR-100 | 50     | ~65%        |

The paper reports 96.04% on CIFAR-10 and 78.69% on CIFAR-100 at 300 epochs with full augmentation — this implementation is a faithful reproduction of the core architecture.

---

## What makes NeST different

Standard ViT runs self-attention between every pair of patches — O(N²) cost, and no built-in sense of locality. NeST fixes this with three ideas:

**1. Block the image.** Patches are divided into non-overlapping spatial groups. Attention stays strictly within each block — patches only attend to spatial neighbours.

**2. Block aggregation.** Between hierarchy levels, blocks are unfolded back to the full image plane, a 3×3 conv + MaxPool is applied (crossing block boundaries), then re-blocked into fewer, larger groups. This is how cross-block information flows.

**3. Nested hierarchy.** The receptive field grows gradually: 16 small blocks → 4 larger blocks → 1 global block. Like a CNN pyramid, but with transformers at each level.

The result is a model that converges faster, needs less data, and produces a natural decision tree that can be interpreted with GradCAT.

---

## Repo structure

```
Nested-Hierarchial-Transformer-PyTorch/
├── configs/
│   ├── cifar10.py          # CIFAR-10 config
│   └── cifar100.py         # CIFAR-100 config
├── models/
│   ├── utils.py            # block() and unblock()
│   ├── patch_embed.py      # PatchEmbedding
│   ├── attention.py        # MHA (multi-head attention from scratch)
│   ├── transformer.py      # TransformerLayer
│   ├── aggregation.py      # BlockAggregation
│   ├── hierarchy.py        # NestHierarchy
│   └── nest.py             # NeST full model
├── data/
│   └── dataset.py          # CIFAR dataloaders
├── engine/
│   └── train.py            # training loop
├── train_script.py                # entry point
├── requirements.txt
└── README.md
```

---

## Install

```bash
git clone https://github.com/shakhsm123/Nested-Hierarchial-Transformer-PyTorch
cd Nested-Hierarchial-Transformer-PyTorch
pip install -r requirements.txt
```

---

## Train

```bash
# CIFAR-10
python train.py --dataset cifar10 --epochs 100 --lr 1e-3

# CIFAR-100
python train.py --dataset cifar100 --epochs 100 --lr 1e-3
```

---



## Architecture overview

```
Image (3, 32, 32)
    ↓ PatchEmbedding (Conv2d, flatten, LayerNorm)
(B, 256, 192)
    ↓ block()
(B, 16, 16, 192)   ← 16 blocks, 16 patches each

    ↓ NestHierarchy (2× TransformerLayer, local attention)
(B, 16, 16, 192)
    ↓ BlockAggregation (unblock → Conv2d → LayerNorm → MaxPool → block)
(B,  4, 16, 192)   ← 4 blocks after aggregation

    ↓ NestHierarchy (2× TransformerLayer)
(B,  4, 16, 192)
    ↓ BlockAggregation
(B,  1, 16, 192)   ← 1 block, sees full image

    ↓ NestHierarchy (2× TransformerLayer)
    ↓ LayerNorm → GlobalAvgPool → Linear
(B, num_classes)
```

---

## Key implementation notes

- `block()` and `unblock()` use `einops.rearrange` to preserve spatial neighbourhoods — a naive reshape would group non-adjacent patches together
- `MHA` folds the block dimension into the batch dimension before attention (`B*num_blocks, seq_len, d`), so all blocks share weights automatically
- `BlockAggregation` operates on the full image plane (not per-block) — this is critical for cross-block information mixing

---
## TODO:
- add GradCat method later as well as general interpretability for nested architecture
## Reference

```bibtex
@article{zhang2021nested,
  title={Nested Hierarchical Transformer: Towards Accurate, Data-Efficient
         and Interpretable Visual Understanding},
  author={Zhang, Zizhao and Zhang, Han and Zhao, Long and Chen, Ting and
          Ar{\i}k, Sercan {\"O} and Pfister, Tomas},
  journal={arXiv preprint arXiv:2105.12723},
  year={2021}
}
```
