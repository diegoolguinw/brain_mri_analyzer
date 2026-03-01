# 🧠 Brain MRI Tumor Segmentation

An end-to-end **deep-learning project** that trains a U-Net model to segment brain tumors from MRI scans, then deploys it as a web application with automatic PDF report generation.

Built for **educational purposes** — the full pipeline is covered: data exploration, model training, evaluation, web development, and cloud deployment.

> **⚠️ Disclaimer:** This project is for learning and research only. It is **not** a medical device and must not be used for clinical decisions.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Django](https://img.shields.io/badge/Django-4.2-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17+-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [The Notebook — Training Pipeline](#-the-notebook--training-pipeline)
- [The Web App — Django + ONNX Runtime](#-the-web-app--django--onnx-runtime)
- [Getting Started](#-getting-started)
  - [1. Training (Notebook)](#1-training-notebook)
  - [2. Exporting to ONNX](#2-exporting-to-onnx)
  - [3. Running the Web App](#3-running-the-web-app)
- [Deployment on Render](#-deployment-on-render)
- [Architecture Deep Dive](#-architecture-deep-dive)
  - [Model Architecture](#model-architecture)
  - [Training Strategy](#training-strategy)
  - [Inference Optimization](#inference-optimization)
- [Key Concepts Covered](#-key-concepts-covered)
- [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

This project walks through the **complete lifecycle** of a medical image segmentation system:

```
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  1. EXPLORE   │ ──▶ │  2. TRAIN    │ ──▶ │  3. DEPLOY   │ ──▶ │  4. ANALYZE  │
    │  & BASELINE   │     │  U-Net MODEL │     │  WEB APP     │     │  & REPORT    │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
     Otsu threshold       Residual SE U-Net    Django + ONNX RT     Upload MRI →
     IoU/Dice eval        Attention U-Net v2   Gunicorn + Docker    Get PDF report
```

**What you'll learn:**
- How to build and train a **U-Net** for semantic segmentation
- Techniques like **SE blocks**, **attention gates**, **EMA**, **test-time augmentation**, and **deep supervision**
- How to **export a PyTorch model to ONNX** for lightweight deployment
- Building a complete **Django web application** with file upload, model inference, and PDF generation
- **Dockerizing** and deploying to the cloud on a free tier (within 512 MB RAM!)

---

## 📁 Repository Structure

```
brain/
├── segmentation.ipynb          # 📓 Full training notebook (data → model → evaluation)
├── export_onnx.py              # 🔄 PyTorch → ONNX conversion script
├── Dockerfile                  # 🐳 Production container (ONNX Runtime, no PyTorch)
├── render.yaml                 # ☁️  Render.com deployment config
│
├── checkpoints/
│   ├── unet_resse_best.pt      # PyTorch checkpoint (training output)
│   ├── model.onnx              # ONNX model (production inference)
│   └── model_meta.json         # Model metadata (img_size, threshold, etc.)
│
└── brain_app/                  # 🌐 Django web application
    ├── manage.py
    ├── requirements.txt
    ├── brain_app/              # Django project configuration
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    └── analyzer/               # Main application module
        ├── nn_models.py        # PyTorch model definitions (training/export only)
        ├── inference.py        # PyTorch inference engine (development)
        ├── inference_onnx.py   # ONNX Runtime inference engine (production)
        ├── report.py           # PDF report generator (ReportLab)
        ├── forms.py            # Upload form
        ├── views.py            # Upload + result views
        ├── urls.py             # URL routing
        ├── templates/analyzer/ # HTML templates (upload page, result page)
        └── static/analyzer/    # CSS styles
```

---

## 📓 The Notebook — Training Pipeline

The [`segmentation.ipynb`](segmentation.ipynb) notebook is a self-contained, extensively documented walkthrough:

| Section | What You'll Learn |
|---|---|
| **Dataset & Exploration** | Loading the LGG MRI Segmentation dataset from Kaggle, visualizing image-mask pairs |
| **Baseline (Otsu)** | Classical thresholding as a non-ML baseline; why it fails and motivates deep learning |
| **Residual SE U-Net** | Building a U-Net with Residual blocks and Squeeze-and-Excitation (SE) attention, `base_ch=48` |
| **Training Pipeline** | Combined BCE + Focal-Tversky loss, `WeightedRandomSampler` for class imbalance, `AdamW` + `CosineAnnealingLR`, gradient clipping, Exponential Moving Average (EMA) |
| **Evaluation** | Threshold grid search, per-patient IoU/Dice metrics, failure analysis for worst cases |
| **Attention U-Net v2** | Upgraded model with Attention Gates, deeper channels (`base_ch=64`), deep supervision, richer augmentations (rotation, noise, brightness), `OneCycleLR` |
| **Head-to-Head Comparison** | Side-by-side evaluation of both models |
| **Model Card** | Summary of intended use, limitations, and responsible deployment |

### Channel progression

```
Residual SE U-Net:     1 → 48 → 96 → 192 → 384 → 768 (bottleneck) → decode back
Attention U-Net v2:    1 → 64 → 128 → 256 → 512 → 1024 (bottleneck) → decode back
```

---

## 🌐 The Web App — Django + ONNX Runtime

A clean, responsive web interface where users can:

1. **Upload** a brain MRI image (PNG, JPEG, TIFF, BMP)
2. **View** the segmentation overlay with tumor highlighted
3. **Download** a structured PDF report with:
   - Detection status and confidence metrics
   - Tumor area estimate and anatomical quadrant
   - Natural-language findings summary
   - Recommendations based on tumor size
   - Clinical disclaimer

### Application Flow

```
User uploads MRI image
        │
        ▼
┌─────────────────────┐
│  Django view saves   │
│  file to /media/     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ONNX Runtime loads  │
│  model (singleton)   │
│  Runs inference      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Generate overlay    │
│  (matplotlib)        │
│  Generate PDF report │
│  (ReportLab)         │
└────────┬────────────┘
         │
         ▼
   Result page with
   overlay + metrics +
   PDF download link
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- (For training) A GPU is helpful but not required

### 1. Training (Notebook)

```bash
# Install training dependencies
pip install torch torchvision numpy matplotlib Pillow scipy kagglehub

# Open the notebook
jupyter notebook segmentation.ipynb
```

Run all cells to train the model. The checkpoint is saved to `checkpoints/unet_resse_best.pt`.

### 2. Exporting to ONNX

After training (or with the provided checkpoint):

```bash
# Install export dependencies
pip install torch onnx

# Export
python export_onnx.py

# Output:
#   checkpoints/model.onnx       (~71 MB)
#   checkpoints/model_meta.json  (threshold, img_size, etc.)
```

### 3. Running the Web App

```bash
cd brain_app

# Install production dependencies (no PyTorch needed!)
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start the development server
python manage.py runserver
```

Open **http://127.0.0.1:8000** and upload an MRI image.

---

## ☁️ Deployment on Render

The app is configured for **Render's free tier** (512 MB RAM) using Docker:

```bash
# Just push to your Git repo — Render builds from the Dockerfile automatically
git add .
git commit -m "Deploy brain MRI analyzer"
git push
```

### How it fits in 512 MB

| Optimization | Savings |
|---|---|
| **ONNX Runtime** instead of PyTorch | ~260 MB (40 MB vs 300 MB) |
| **1 gunicorn worker** + 2 threads | ~50% less than 2 workers |
| **No TTA** in production (single pass) | ~20 MB peak reduction |
| **ONNX model** (71 MB) vs `.pt` (141 MB) | ~70 MB disk savings |

**Estimated runtime memory: ~150–200 MB** ✅

### Environment Variables

| Variable | Description |
|---|---|
| `DJANGO_SECRET_KEY` | Auto-generated by Render |
| `DJANGO_DEBUG` | `0` for production |
| `ONNX_MODEL_PATH` | Path to `.onnx` file inside the container |
| `CSRF_TRUSTED_ORIGINS` | Your Render URL |

---

## 🏗 Architecture Deep Dive

### Model Architecture

**Residual SE U-Net** — a U-Net backbone enhanced with:

- **Residual connections** — skip connections within each encoder/decoder block to ease gradient flow
- **Squeeze-and-Excitation (SE) blocks** — channel-wise attention that lets the network learn which feature channels are most informative
- **Batch Normalization** + **Dropout** at increasing rates through the encoder

**Attention U-Net v2** adds:

- **Attention Gates** — spatial attention at each skip connection, letting the decoder focus on relevant regions
- **Deep Supervision** — auxiliary loss heads at intermediate decoder stages for better gradient propagation during training

### Training Strategy

| Technique | Why |
|---|---|
| **BCE + Focal-Tversky Loss** | Handles severe class imbalance (most pixels are background) |
| **WeightedRandomSampler** | Oversamples slices that contain tumors |
| **Exponential Moving Average (EMA)** | Stabilizes weights; EMA checkpoint used for inference |
| **Test-Time Augmentation (TTA)** | 4-fold flipping at inference for more robust predictions |
| **Gradient Clipping** | Prevents exploding gradients |
| **Cosine Annealing / OneCycleLR** | Learning rate scheduling for better convergence |
| **Early Stopping** | Saves the best model, stops when validation loss plateaus |

### Inference Optimization

The production deployment swaps out PyTorch entirely:

```
Training:   PyTorch (GPU/CPU) → .pt checkpoint
                                      │
                                      ▼
Export:     export_onnx.py → .onnx model + metadata
                                      │
                                      ▼
Production: ONNX Runtime (CPU only, ~40 MB) → fast, lightweight inference
```

Key differences between training and production inference:

| | Training / Dev | Production |
|---|---|---|
| Runtime | PyTorch (~300 MB) | ONNX Runtime (~40 MB) |
| TTA | 4-fold flip augmentation | Single forward pass |
| Checkpoint | `.pt` (141 MB) | `.onnx` (71 MB) |
| Sigmoid | `torch.sigmoid` | `scipy.special.expit` |

---

## 📚 Key Concepts Covered

This project is a learning resource for:

- **Medical Image Segmentation** — pixel-level classification for tumor detection
- **U-Net Architecture** — the encoder-decoder model that dominates biomedical segmentation
- **Attention Mechanisms** — SE blocks (channel attention) and Attention Gates (spatial attention)
- **Training Best Practices** — loss design, class imbalance handling, EMA, TTA, early stopping
- **Model Export & Optimization** — PyTorch → ONNX for production deployment
- **Web Development** — Django forms, views, templates, PDF generation
- **DevOps** — Dockerization, cloud deployment, memory-efficient serving
- **Responsible AI** — disclaimers, model cards, limitations documentation

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Training** | PyTorch, NumPy, Matplotlib, Kaggle Hub |
| **Model Export** | ONNX |
| **Web Framework** | Django 4.2 |
| **Inference (prod)** | ONNX Runtime |
| **PDF Reports** | ReportLab |
| **Image Processing** | Pillow, SciPy, Matplotlib |
| **Static Files** | WhiteNoise |
| **Server** | Gunicorn |
| **Containerization** | Docker |
| **Deployment** | Render (free tier) |

---

## 📄 License

This project is intended for educational and research purposes.

---

<p align="center">
  Made with ❤️ for learning deep learning applied to healthcare, web development, and deployment.
</p>
