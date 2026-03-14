# 🎯 Attention-Driven Spatial Recognition using YOLOv8 + CBAM + Grad-CAM

> Improving object detection accuracy and interpretability in cluttered environments by integrating **Convolutional Block Attention Modules (CBAM)** into YOLOv8, with **Grad-CAM** visual explanations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red.svg)](https://pytorch.org/)

---

## 🚀 Overview

Autonomous robots often fail to detect small or partially occluded traffic signs (e.g., "No Entry") in complex warehouse environments. This project demonstrates how **attention mechanisms improve spatial localization and detection confidence**, while maintaining model explainability through Grad-CAM heatmaps.

### Key Contributions

- **CBAM integration** into YOLOv8 backbone and neck layers
- **Multi-scale inference** strategy for robust small-object detection
- **Grad-CAM explainability** for visualizing model attention
- **CPU-feasible** pipeline with minimal latency overhead

---

## 🧠 Methodology

### Baseline Model

- **Architecture:** YOLOv8-s (small variant)
- **Pre-trained on:** COCO dataset (80 classes)
- **Fine-tuned on:** Merged dataset — COCO subset + custom No-Entry sign dataset (81 classes)

### CBAM Integration

CBAM modules were inserted into the **backbone feature blocks** and **neck multi-scale fusion layers**.

```
Feature Map → Channel Attention → Spatial Attention → Refined Feature Map
```

| Sub-Module | Operations |
|---|---|
| **Channel Attention** | Global Avg Pool → Global Max Pool → Shared MLP → Sigmoid |
| **Spatial Attention** | Channel-axis Pool → Concatenation → 7×7 Conv → Sigmoid |

### Training Strategy

A two-stage fine-tuning approach was used to avoid catastrophic forgetting:

| Stage | Script | Purpose |
|---|---|---|
| Stage 1 | `train_cbam_stage1.py` | Stabilize CBAM weights (early backbone layers frozen) |
| Stage 2 | `train_cbam_stage2.py` | Fine-tune the full detector end-to-end |

### Multi-Scale Detection

Inference is performed across multiple resolutions and results are merged via **Non-Maximum Suppression (NMS)**:

```
Scales: 640, 800, 960, 1280, 1536
```

This improves detection of both large foreground objects and small background objects.

---

## 📊 Results

### Detection Performance

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Baseline YOLOv8 | ~0.62 | ~0.57 | ~0.62 | ~0.44 |
| **YOLOv8 + CBAM** | **0.96** | **0.88** | **0.96** | **0.90** |

✅ Improved bounding-box localization  
✅ Better detection of small traffic signs  
⚠️ Slight increase in false positives

### CPU Inference Speed

Measured on a standard laptop CPU:

| Model | Avg Inference Time | FPS |
|---|---|---|
| Baseline | 0.1711 s | 5.84 |
| CBAM Enhanced | 0.1717 s | 5.83 |

> **Latency increase: only 0.31%** — attention modules add negligible overhead.

### Grad-CAM Explainability

Grad-CAM visualizations confirm the model focuses on semantically meaningful regions:

- **Red circular border** of No-Entry signs
- **Text regions** within signs
- Spatial attention improves **object localization precision**

Output includes bounding boxes with heatmap overlays and class-specific attention maps.

---

## 📂 Project Structure

```
├── Scripts/
│   ├── cbam.py                    # CBAM module definition (Channel + Spatial Attention)
│   ├── build_cbam_from_baseline.py # Inject CBAM into a trained baseline model
│   ├── build_cbam_model.py         # Build CBAM-enhanced YOLOv8 from scratch
│   ├── train_cbam_stage1.py        # Stage 1: Stabilize CBAM weights
│   ├── train_cbam_stage2.py        # Stage 2: Full fine-tuning
│   ├── train_cbam.py               # Single-stage CBAM training
│   ├── train_final.py              # Final training on merged dataset
│   ├── test_baseline.py            # Evaluate baseline model
│   ├── cpu_benchmark.py            # CPU inference speed comparison
│   ├── multiscale_detect.py        # Multi-scale detection with NMS
│   ├── gradcam_detect.py           # Grad-CAM visualization
│   └── dataset_final.yaml          # Dataset config (80 COCO + noentry)
│
├── Models/
│   ├── yolov8s.pt                  # Pre-trained YOLOv8-s
│   ├── baseline_trained.pt         # Fine-tuned baseline
│   ├── cbam_stage1.pt              # After Stage 1 training
│   └── cbam_stage2.pt              # After Stage 2 training
│
├── Input Images/                   # Test images for inference
├── Baseline Model Output/          # Baseline detection results (trained / untrained)
├── Baseline_CBAM Output/           # CBAM detection + Grad-CAM outputs
├── CPU Time Comparison/            # CPU benchmark results
├── noentry_dataset/                # Custom No-Entry sign dataset
├── Project Demo.mp4                # Full project demo video
└── README.md
```

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python pytorch-grad-cam torch torchvision
```

### 2. Train Baseline

```bash
python Scripts/test_baseline.py
```

### 3. Inject CBAM & Train (Two-Stage)

```bash
python Scripts/build_cbam_from_baseline.py
python Scripts/train_cbam_stage1.py
python Scripts/train_cbam_stage2.py
```

### 4. Run Multi-Scale Detection

```bash
python Scripts/multiscale_detect.py
```

### 5. Generate Grad-CAM Visualizations

```bash
python Scripts/gradcam_detect.py
```

### 6. CPU Speed Benchmark

```bash
python Scripts/cpu_benchmark.py
```

---

## 🌍 Applications & Impact

This approach improves safety and reliability across:

| Domain | Use Case |
|---|---|
| 🏭 Warehouse Robotics | Detecting obstacle signs in cluttered environments |
| 🚦 Traffic Monitoring | Recognizing small / occluded road signs |
| 📹 Smart Surveillance | Reliable detection in complex scenes |
| ♿ Assistive Navigation | Real-time hazard awareness for visually impaired |

---

## 📌 Key Takeaways

- ✔️ **Attention improves spatial reasoning** — CBAM refines feature maps without architectural overhaul
- ✔️ **Multi-scale inference is critical** — essential for detecting objects at varying sizes
- ✔️ **Explainability is non-negotiable** — Grad-CAM provides trust for safety-critical systems
- ✔️ **Lightweight attention modules** — only 0.31% latency increase on CPU

---

## 📖 References

1. Woo, S. et al. — [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) (ECCV 2018)
2. Selvaraju, R.R. et al. — [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391) (ICCV 2017)
3. Ultralytics — [YOLOv8 Framework](https://github.com/ultralytics/ultralytics)

---
