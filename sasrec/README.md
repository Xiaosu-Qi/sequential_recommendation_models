# SASRec Models

This folder contains implementations of the **SASRec (Self-Attentive Sequential Recommendation)** model and its variants. These models use self-attention (from Transformers) to model user behavior sequences and predict the next item a user is likely to interact with.

## 📄 Original Paper

> Wang-Cheng Kang and Julian McAuley.  
> **Self-Attentive Sequential Recommendation.**  
> CIKM 2018. [[arXiv]](https://arxiv.org/abs/1808.09781)

## 📦 Variants Implemented

| Subfolder             | Description                                             |
|-----------------------|---------------------------------------------------------|
| `baseline/`           | Reimplementation of the original SASRec model.          |
| `sasrec+time/`        | Time-aware extension that incorporates temporal gaps.   |
| `sasrec+contrastive/` | Contrastive learning enhanced version (coming soon).    |

Each folder includes its own:
- 📁 `model.py`: model architecture
- 🏋️‍♀️ `trainer.py`: training logic
- 🚀 `demo.py`: example usage
- 📘 `README.md`: explanation of the variant and paper it’s based on

## 🚀 Getting Started (SASRec Baseline)

```bash
cd sasrec/baseline
python demo.py
