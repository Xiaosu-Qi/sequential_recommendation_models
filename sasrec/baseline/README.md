# SASRec - Baseline Implementation

This is a PyTorch reimplementation of the original **SASRec** model as described in the paper:

> Wang-Cheng Kang and Julian McAuley.  
> **Self-Attentive Sequential Recommendation.**  
> Proceedings of the 27th ACM International Conference on Information and Knowledge Management (CIKM 2018).  
> [[arXiv]](https://arxiv.org/abs/1808.09781)

## 🧠 Model Overview

SASRec is one of the earliest Transformer-based models for sequential recommendation. It models a user's interaction history using **self-attention** to capture both short-term and long-term preferences, and predicts the next item the user will likely interact with.

### Key Components:
- **Item Embedding**: Learnable embeddings for each item ID.
- **Positional Embedding**: To preserve order in the sequence.
- **Self-Attention Blocks**: Multi-head self-attention + feed-forward layers.
- **Output Layer**: Predicts the likelihood of each candidate item via dot product.

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-placeholder.png" alt="SASRec architecture" width="600"/>
</p>

## 📦 Files

| File         | Description                          |
|--------------|--------------------------------------|
| `model.py`   | Defines the SASRec model architecture |
| `trainer.py` | Training loop, evaluation, metrics   |
| `demo.py`    | Example script to run a small demo   |

## 🚀 Quick Start

```bash
cd sasrec/baseline
python demo.py
