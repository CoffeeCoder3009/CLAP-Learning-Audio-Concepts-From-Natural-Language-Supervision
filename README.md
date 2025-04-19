# CLAP-Learning-Audio-Concepts-From-Natural-Language-Supervision

This repository contains an implementation and benchmarking framework for the **CLAP** model on various public audio datasets.

CLAP (Contrastive Language-Audio Pretraining) learns audio representations directly from natural language supervision, enabling **zero-shot audio classification** without requiring predefined class labels.

---

## 📌 Paper Details

- **Title:** CLAP: Learning Audio Concepts from Natural Language Supervision  
- **Authors:** Yusong Wu, Ke Chen, Takuya Nishimura, et al.  
- **arXiv:** [2206.04769](https://arxiv.org/abs/2206.04769)  


---

## 🚀 How to Run

### 1. Clone the repo & install dependencies

```bash
git clone https://github.com/CoffeeCoder3009/CLAP-Learning-Audio-Concepts-From-Natural-Language-Supervision.git

```
### 2. Install requirements

```bash
pip install -r requirements.txt
```
### 3. Run inference on a dataset

```bash
python esc50_clap_eval.py
```


---

## 📊 Results on Datasets

| Dataset       | Zero-Shot Accuracy (ours) |  Zero-Shot Accuracy (actual) 
|---------------|---------------------|---------------------|
| ESC-50        | 82.10%              |  82.60%             |
| US8K          | 74.62%              |  73.24%             |
| FSD50K          | -              |  0.3024 (mAP)           |
   

---
