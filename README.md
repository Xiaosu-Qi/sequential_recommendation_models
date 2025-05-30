# Sequential Recommendation Models (PyTorch)

A collection of reproducible PyTorch implementations of **state-of-the-art sequential recommendation algorithms**, including Transformer-based models such as **SASRec**, **BERT4Rec**, **GRU4Rec**, and more.

This repository aims to provide clean, modular, and research-friendly implementations of key papers in the field, enabling easy experimentation and benchmarking.

## 📘 Included Models

| Model       | Paper Title                                      | Year | Link |
|-------------|--------------------------------------------------|------|------|
| SASRec      | Self-Attentive Sequential Recommendation         | 2018 | [arXiv](https://arxiv.org/abs/1808.09781) |
| BERT4Rec    | BERT4Rec: Sequential Recommendation with BERT    | 2019 | [arXiv](https://arxiv.org/abs/1904.06690) |
| GRU4Rec     | Session-based Recommendations with RNNs          | 2016 | [PDF](https://arxiv.org/abs/1511.06939)   |
| (More soon) | ...                                              |      |      |

## 🧠 Why Sequential Recommendation?

Sequential recommenders aim to predict the **next item** a user will interact with, based on their recent behavior history. Unlike traditional models, these architectures capture temporal patterns and evolving preferences using:

- Recurrent Neural Networks (RNNs)
- Convolutional Nets (CNNs)
- Self-Attention (Transformers)

## 🏗️ Repo Structure
sequential-recommendation-models/
│
├── sasrec/                             # 📄 SASRec (Self-Attentive Sequential Recommendation)
│   ├── baseline/                       # Original SASRec implementation
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── demo.py
│   │
│   ├── sasrec+time/                    # Time-aware SASRec variant (example)
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── README.md                   # Explains the modification or paper background
│   │
│   ├── sasrec+contrastive/            # Contrastive learning enhanced variant (example)
│       ├── model.py
│       ├── trainer.py
│       └── README.md
│
├── bert4rec/                           # BERT4Rec implementation (coming soon)
│   └── ...
│
├── data/                               # Sample datasets and preprocessors
│   └── ml-1m/
│
├── common/                             # Shared utilities and layers
│   ├── dataloader.py
│   ├── layers.py
│   └── config.py
│
├── main.py                             # 🚀 Entry script to run different models
├── requirements.txt                    # 📦 Python dependencies
└── README.md                           # 📝 Project description (you’re reading it!)

## 🛠️ Installation

```bash
pip install -r requirements.txt
```
Note: For GPU support, make sure to install the correct PyTorch version from [https://pytorch.org](https://pytorch.org) based on your CUDA version.


## ✨ About the Author
Hi there! 👋 I'm Xiaosu Qi, currently working as a machine learning engineer at Grubhub 🥡, with 4 years of hands-on experience building and scaling ML systems in production — from recommender to ranking models, and a fair share of wrestling with Spark 😅.

This repo is my personal playground 🎮 for reproducing and demystifying some of the coolest papers in sequential recommendation, especially transformer-based ones. I hope it helps others (and future me 🧠) to better understand the inner workings of these models.

If you're also obsessed with clean code, cat memes, or self-attention mechanisms — feel free to connect or drop a star ⭐️!

[📫 LinkedIn](https://www.linkedin.com/in/xiaosu-q-979b67155/) | [🐱 GitHub](https://github.com/Xiaosu-Qi)

