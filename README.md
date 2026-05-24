# UAVYOLO — UAV Detection with YOLO & Custom Deep Learning Models

> **Freelance project** for **UAV (Unmanned Aerial Vehicle) detection** in aerial imagery. This repository implements and compares state-of-the-art object detection pipelines — including **YOLOv10**, **YOLOv11**, and a custom **HMAY-TSF** architecture — optimized for small-object detection in drone and satellite views.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv11-green.svg)](https://github.com/ultralytics/ultralytics)

---

## Overview

**UAVYOLO** is a computer vision project focused on detecting aerial vehicles — drones, aircraft, and related objects — from overhead and SAR (Synthetic Aperture Radar) imagery. Built as a freelance engagement, it delivers a full end-to-end pipeline: dataset preparation, model training, evaluation, and side-by-side benchmarking of multiple detection architectures.

Whether you are researching **aerial object detection**, building **drone surveillance systems**, or comparing **YOLO vs. Faster R-CNN** for UAV traffic monitoring, this repo provides reproducible scripts and a custom attention-enhanced baseline model.

### Key Features

- **Multi-model support** — Train and evaluate YOLOv10, YOLOv11, Faster R-CNN, RetinaNet, and a custom HMAY-TSF model
- **UAV-optimized architecture** — Hybrid Feature Refinement Module (HFRM) with adaptive channel & spatial attention for small-object detection
- **Super-Resolution Data Augmentation (SRDA)** — Preprocessing pipeline to simulate high-resolution inputs
- **COCO-format evaluation** — Full mAP@0.5, mAP@0.5:0.95, precision, recall, and inference-time benchmarking
- **Roboflow integration** — Easy aerial vehicle dataset download in COCO and YOLO formats
- **Automated checkpointing** — Best-model saving, periodic checkpoints, and CSV metrics logging

---

## Architecture

### HMAY-TSF Baseline Model

**Hybrid Multi-scale Attention + Temporal-Spatial Fusion (HMAY-TSF)** is a custom Faster R-CNN variant designed specifically for UAV traffic domain detection.

```
Input Image → ResNet-50 → FPN → HFRM → ROI Heads → Detections
                ↓           ↓      ↓
            Backbone   Multi-scale  Channel +
                      Refinement   Spatial Attention
```

| Component | Description |
|-----------|-------------|
| **Backbone** | ResNet-50 with pre-trained ImageNet weights |
| **Neck** | Feature Pyramid Network (FPN) with integrated HFRM |
| **HFRM** | Hybrid Feature Refinement Module — SPP-like multi-scale pooling + ACA/ASA attention |
| **Anchors** | Reduced sizes (16–256 px) tuned for small aerial objects |
| **SRDA** | Optional super-resolution augmentation before inference |

See [`models/README.md`](models/README.md) for detailed architecture documentation.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, TorchVision, Ultralytics YOLO |
| Evaluation | pycocotools (COCO mAP) |
| Data | Roboflow, OpenCV, Pillow |
| Visualization | Matplotlib, custom result visualizer |
| Utilities | NumPy, Pandas, scikit-learn, tqdm |

---

## Project Structure

```
UAVYOLO/
├── models/
│   ├── faster_rcnn.py      # HMAY-TSF baseline model
│   ├── nr_module.py        # Hybrid Feature Refinement Module (HFRM)
│   ├── srda_pipeline.py    # Super-Resolution Data Augmentation
│   └── README.md           # Architecture deep-dive
├── datasets/
│   ├── custom_dataset.py   # COCO-format PyTorch Dataset
│   └── eval_dataset.py     # Evaluation dataset loader
├── utils/
│   ├── transforms.py       # Image augmentation & preprocessing
│   ├── inference.py        # Inference time measurement
│   ├── metrics.py          # Evaluation metrics helpers
│   └── visualization.py    # Training history plots
├── train_model.py          # Main training entry point
├── train.py                # Training loop with COCO evaluation
├── trainYolo.py            # YOLOv10 / YOLOv11 training script
├── eval.py                 # Standalone model evaluation
├── compare.py              # Multi-model benchmark comparison
├── results_compare.py      # Visual side-by-side result comparison
├── data.py                 # Roboflow dataset downloader
├── test_training_pipeline.py
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/danishjavedcodes/UAVYOLO.git
cd UAVYOLO
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. GPU support (optional)

For CUDA-accelerated training, install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Dataset Setup

This project uses aerial vehicle datasets exported from [Roboflow](https://roboflow.com/) in COCO format.

### Download via Roboflow

1. Create a free [Roboflow](https://app.roboflow.com/) account.
2. Set your API key as an environment variable:

```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

3. Update `data.py` with your workspace/project details and run:

```bash
python data.py
```

The dataset will be downloaded to a local directory (e.g., `Aerial-Vehicles-1/`) with `train/` and `valid/` splits and `_annotations.coco.json` files.

### Expected directory layout

```
Aerial-Vehicles-1/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
└── valid/
    ├── _annotations.coco.json
    └── *.jpg
```

---

## Usage

### Train the custom HMAY-TSF model

```bash
python train_model.py
```

Configurable hyperparameters in `train_model.py`:

| Parameter | Default |
|-----------|---------|
| `NUM_CLASSES` | 5 |
| `BATCH_SIZE` | 16 |
| `NUM_EPOCHS` | 150 |
| `LEARNING_RATE` | 0.005 |

Training outputs are saved to `results/`:
- `best_model.pth` — highest mAP@0.5 checkpoint
- `model_epoch_N.pth` — checkpoint every 10 epochs
- `training_metrics.csv` — per-epoch loss and mAP logs

### Train YOLO models

```bash
python trainYolo.py
```

Trains both **YOLOv11n** and **YOLOv10n** on your YOLO-format dataset. Update the `data.yaml` path inside the script to match your local dataset location.

### Evaluate a trained model

```bash
python eval.py
```

Reports mAP@0.5, mAP@0.5:0.95, mAP@0.75, and average inference time per image.

### Compare all models

```bash
python compare.py
```

Benchmarks YOLOv10, YOLOv11, HMAY-TSF, Faster R-CNN, and RetinaNet side-by-side. Generates a comparison table and saves `model_comparison.png`.

### Visualize detection results

```bash
python results_compare.py
```

Overlays bounding boxes from multiple models on the same images for qualitative comparison.

### Test the training pipeline

```bash
python test_training_pipeline.py
```

Runs a short 2-epoch smoke test to verify metrics logging and checkpointing.

---

## Model Comparison

| Model | Type | Best For |
|-------|------|----------|
| **YOLOv11** | One-stage (Ultralytics) | Real-time UAV detection |
| **YOLOv10** | One-stage (Ultralytics) | Fast inference with NMS-free head |
| **HMAY-TSF** | Two-stage (Custom Faster R-CNN) | Small-object accuracy in aerial scenes |
| **Faster R-CNN** | Two-stage (TorchVision) | General-purpose baseline |
| **RetinaNet** | One-stage (TorchVision) | Focal-loss balanced detection |

All models are evaluated using the COCO evaluation protocol (IoU thresholds 0.50–0.95).

---

## Metrics & Logging

During training, the following metrics are tracked per epoch and written to `results/training_metrics.csv`:

- Training loss
- Validation loss
- Precision
- Recall
- **mAP@0.5**
- **mAP@0.5:0.95**

Best models are saved automatically when mAP@0.5 improves.

---

## Use Cases

- **Drone traffic monitoring** — Detect and track UAVs in urban airspace
- **Aerial surveillance** — Identify vehicles and aircraft from satellite or drone footage
- **SAR imagery analysis** — Object detection on synthetic aperture radar data
- **Research benchmarking** — Compare YOLO and two-stage detectors on aerial datasets
- **Edge deployment prototyping** — Evaluate inference speed vs. accuracy trade-offs

---

## Freelance Project Context

This repository was developed as a **freelance computer vision project** focused on **UAV detection** for a client requiring:

1. A custom deep learning model optimized for small aerial objects
2. Benchmarking against industry-standard YOLO and R-CNN baselines
3. A reproducible training and evaluation pipeline with full metrics reporting
4. Visual comparison tools for model selection and deployment decisions

---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request on [GitHub](https://github.com/danishjavedcodes/UAVYOLO).

---

## Author

**Danish Javed** — [GitHub](https://github.com/danishjavedcodes)

---

## Keywords

`UAV detection` · `drone detection` · `aerial object detection` · `YOLO` · `YOLOv11` · `YOLOv10` · `Faster R-CNN` · `RetinaNet` · `computer vision` · `deep learning` · `PyTorch` · `small object detection` · `aerial imagery` · `SAR` · `Roboflow` · `COCO mAP` · `object detection benchmark` · `freelance ML project`
