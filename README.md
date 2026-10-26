# DeepShield — AI Fake Image Detection & Grad-CAM Forensics Engine

**DeepShield** is a dual-backbone digital image forensics and explainability platform combining deep convolutional neural networks (**DenseNet-121** and **ResNet-50**) with an advanced multi-tier computer vision forensic pipeline (**Error Level Analysis**, **2D Fourier/DCT Radial Spectrum Analysis**, **Laplacian Sensor Noise**, and **EXIF Hardware Analysis**).

---

## 🛡️ Core Capabilities & Architecture

- **Dual-Backbone Vision Classification**:
  - **DenseNet-121**: Specializes in fine-grained micro-texture reuse and local synthesis artifacts (`target_layer: features.norm5`).
  - **ResNet-50**: Specializes in macro structural coherence and physical edge boundaries (`target_layer: layer4[-1]`).
- **Explainable Grad-CAM Heatmaps**:
  - Gradient-weighted Class Activation Mapping computed targeting the specific predicted class (`AI-Generated` vs. `Authentic`).
  - Supports **Overlay**, **Interactive Split Slider**, **Heatmap Only**, **Original Image**, and dynamic **Heatmap Opacity**.
  - Localized hotspot activation detection and objective spatial descriptions.
- **Forensic Vision Pipeline**:
  - **Error Level Analysis (ELA)**: Quantifies edge-to-flat compression residuals upon JPEG quantization.
  - **2D Fast Fourier Transform (FFT)**: Measures radial power spectrum decay slope and high-frequency energy ratio ($1/f^\alpha$). Detects periodic deconvolution grid spikes.
  - **Laplacian Sensor Shot-Noise Variance**: Measures natural optical sensor noise ($15 \le \sigma^2 \le 250$).
  - **Chromatic Channel Correlation**: Pearson correlation across R, G, and B sensor color channels.
  - **EXIF Hardware Metadata Inspection**: Identifies genuine camera sensor tags (Make, Model, Lens, ISO) vs. AI software tags.
- **3-Outcome Uncertainty Decision Policy**:
  - **Likely Authentic** ($\le 38\%$ AI probability, concordant models).
  - **Likely AI-Generated** ($\ge 62\%$ AI probability, concordant models).
  - **Inconclusive / Uncertain** (ambiguous interval $38\% - 62\%$ or model disagreement $\Delta > 25\%$).
- **Features Included**:
  - **Model Agreement Panel**: Visualizes individual DenseNet and ResNet predictions with architectural divergence warnings.
  - **Downloadable Forensic Report**: Exports full diagnostic payload to formatted JSON report.
  - **Batch Analysis**: Upload up to 10 images with real-time queue, results table, and CSV export.
  - **Image Metadata Viewer**: Inspects dimensions, color mode, container format, and EXIF tags.
  - **Model Diagnostics**: Real-time compute device, CUDA availability, checkpoint integrity, and execution latency.
  - **Benchmark Evaluation Dashboard**: Evaluates held-out test sets with Accuracy, ROC-AUC, F1-Score, and Confusion Matrix.

---

## 🚀 Quickstart & Server Launch

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Launch Backend Server
```bash
python app.py
```
Or double-click `start_server.bat` on Windows.
The server will start on: **http://127.0.0.1:5000**

---

## 🧪 Automated Test Suite

Run the comprehensive unit and integration test suite:
```bash
python test_deepshield.py
```
Covers:
- `/health`, `/diagnostics`, `/models`, and `/benchmarks` endpoints.
- End-to-end image inference and response schemas.
- Error handling (empty files, corrupted data, size limit exceedance).
- Batch analysis pipeline.
- Numerical safety on edge-case images (flat 0-variance images, division by zero).
- Grad-CAM layer extraction and tensor gradient backpropagation.
- Uncertainty decision policy and model disagreement triggers.
- Consecutive upload state isolation.

---

## 🏋️ Model Training & Fine-Tuning Pipeline

DeepShield provides a separate training harness (`train.py`) and evaluation harness (`evaluate.py`) for binary classification between authentic camera photographs and AI-generated synthetic images (e.g., Midjourney, Stable Diffusion, DALL-E, Flux, GANs).

### Dataset Directory Format
Prepare a balanced dataset (such as GenImage or CIFAKE) structured as follows:
```
data/
  train/
    authentic/
    ai_generated/
  val/
    authentic/
    ai_generated/
  test/
    authentic/
    ai_generated/
```

### Run Fine-Tuning
```bash
# Fine-tune both DenseNet-121 and ResNet-50 backbones
python train.py --model ensemble --data_dir data --epochs 10 --batch_size 16 --lr 1e-4

# Fine-tune a specific backbone
python train.py --model densenet121 --data_dir data --epochs 10
python train.py --model resnet50 --data_dir data --epochs 10
```
Trained checkpoints are automatically saved to `checkpoints/densenet121_deepshield.pth` and `checkpoints/resnet50_deepshield.pth`. When present, DeepShield switches automatically from *Heuristic Forensics Mode* to *Production Binary Classifier Mode*.

### Run Test Set Evaluation
```bash
python evaluate.py --data_dir data/test
```
Computes and saves the held-out test metrics (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`, `False Positive Rate`, `Inference Latency`, and `Confusion Matrix`) to `checkpoints/evaluation_results.json`, instantly reflected in the dashboard's **📊 Benchmarks** view.

---

## ⚠️ Important Limitations & Ethical Disclaimers

1. **Grad-CAM Visualizations**: Grad-CAM visualizes spatial regions that contributed most heavily to the model's classification score. It highlights decision attention; it does not independently prove an image is synthetic or authentic.
2. **Web Re-compression & EXIF**: EXIF metadata is routinely stripped by major messaging and social platforms (Twitter/X, WhatsApp, Discord, Instagram). Missing EXIF metadata is standard across the web and is not definitive proof of AI generation.
3. **Forensic Metrics**: Error Level Analysis (ELA) and frequency spectra can vary based on image resolution, re-saving history, and post-processing filters. DeepShield uses an uncertainty-aware policy and reports "Inconclusive" when evidence is ambiguous.
