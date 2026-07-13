# HMT: Hybrid Mamba–Transformer for Image Captioning

<p align="center">

Official PyTorch implementation of **HMT**, a hybrid Vision Mamba–Transformer architecture for efficient image captioning.

**Authors:** Mohammad Abdian, Sayeh Mirzaei

</p>

---

## ✨ Highlights

- 🚀 Hybrid Vision Mamba–Transformer architecture
- 🔄 Cross-shaped 2D spatial scanning
- 🧠 Gated fusion of CLIP and Vision Mamba features
- ⚡ Causal Mamba decoder with Transformer cross-attention
- 📉 Only **7.3 GFLOPs**
- 🎯 Competitive performance on **MS COCO** and **Flickr30k**

---

## 🏗️ Architecture

<p align="center">
<img src="assets/diagram.png" width="900">
</p>

### Encoder
- Frozen CLIP ViT-B/32
- Vision Mamba
- Horizontal & Vertical Scan
- Gated Feature Fusion

### Decoder
- Causal Mamba
- Cross-Attention
- Linear-time sequence modeling

---

## 📊 Model Statistics

| Item | Value |
|------|------:|
| Backbone | CLIP ViT-B/32 |
| Total Parameters | **138.6M** |
| Trainable Parameters | **50.8M** |
| Frozen Parameters | **87.8M** |
| FLOPs | **7.3G** |

---

## 📈 Results

### MS COCO Karpathy Test Split

| BLEU-4 | CIDEr |
|-------:|------:|
| **38.4** | **122.5** |

### Flickr30k Karpathy Test Split

| Metric | Score |
|--------|------:|
| BLEU-1 | 68.3 |
| BLEU-2 | 52.4 |
| BLEU-3 | **39.4** |
| BLEU-4 | **29.3** |
| METEOR | **25.6** |
| ROUGE-L | **50.1** |
| CIDEr | **63.8** |
| SPICE | **16.6** |

---

## ⚙️ Training

| Item | Value |
|------|------:|
| GPU | RTX 4090 (24GB) |
| Optimizer | AdamW |
| Batch Size | 32 |
| Epochs | 5 |
| Learning Rate | 1e-4 |
| Scheduler | Cosine Annealing |
| Beam Size | 3 |

Training Time

| Dataset | Time |
|---------|------|
| MS COCO | 10.1 h |
| Flickr30k | 2.3 h |

---

## 📂 Datasets

Experiments follow the standard **Karpathy split**.

| Dataset | Train | Val | Test |
|---------|------:|----:|-----:|
| MS COCO | 113,287 | 5,000 | 5,000 |
| Flickr30k | 29,783 | 1,000 | 1,000 |

---

## 📏 Evaluation Metrics

- BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4
- METEOR
- ROUGE-L
- CIDEr
- SPICE

---

## 🚀 Quick Start

```bash
# Feature extraction
python -m HMT.feature_pipeline.run

# Training
python -m HMT.train

# Evaluation
python -m HMT.evaluate

# Inference
python -m HMT.inference
```

---

## 📁 Repository Structure

```
HMT/
├── configs/
├── datasets/
├── feature_pipeline/
├── models/
├── utils/
├── train.py
├── evaluate.py
├── inference.py
└── README.md
```

---

## 📄 Paper

The manuscript is currently under review.

---

## 💻 Code Availability

The complete implementation is available at:

**https://github.com/mohammadabdian/HMT**

---

## 📚 Citation

```bibtex
@article{abdian2026hmt,
  title={HMT: A Hybrid Mamba--Transformer Architecture for Image Captioning},
  author={Mohammad Abdian and Sayeh Mirzaei},
  journal={Under Review},
  year={2026}
}
```

---

## 📜 License

Released for academic research purposes.
