# 🚗 AIC2025 - Fisheye Object Detection

> **AI City Challenge 2025 - Team Tennhom**  
> Real-time object detection on fisheye camera images using DEIM (DETR with Improved Matching)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![DEIM](https://img.shields.io/badge/Model-DEIM-orange.svg)

## 📋 Overview

This repository contains our solution for the **AI City Challenge 2025**, focusing on detecting vehicles and pedestrians in fisheye camera images.

### 🏆 Final Model: DEIM (DETR with Improved Matching)

We use **[DEIM](https://github.com/Intellindust-AI-Lab/DEIM)** as our final detection model. DEIM is an advanced training framework that enhances the matching mechanism in DETRs, enabling faster convergence and improved accuracy. 

**Key advantages of DEIM:**

- ⚡ Fast convergence with improved matching mechanism
- 🎯 State-of-the-art accuracy on COCO benchmark
- 🚀 Real-time inference capability
- 📊 Multiple model sizes (Nano to XLarge)

### 🎯 Target Classes

| ID  | Class      |
| --- | ---------- |
| 0   | Bus        |
| 1   | Bike       |
| 2   | Car        |
| 3   | Pedestrian |
| 4   | Truck      |

## 📁 Project Structure

```
AIC2025_Tennhom/
├── augmentation/           # Data augmentation techniques
│   └── cycle_gan/          # CycleGAN for domain adaptation
├── dataset/                # Dataset files
│   ├── fisheye8k/          # Fisheye8K dataset
│   ├── fisheye_test/       # Test images
│   ├── visdrone/           # VisDrone dataset
│   ├── HIT-UAV.modif/      # Modified HIT-UAV dataset
│   └── synthetic_visdron/  # Synthetic data
├── eda/                    # Exploratory Data Analysis
│   └── FishEye8k.ipynb     # Dataset analysis notebook
├── infer/                  # Inference scripts
│   └── inference.py        # Main inference script
├── submission/             # Competition submissions
│   └── rtdetr.json         # Submission file in COCO format
└── train/                  # Training code
    ├── model/              # Model implementations
    │   ├── DEIM/           # Final model - DEIM 
    │   ├── Drone-DETR/     # Drone-DETR model
    │   ├── RT-DETR/        # RT-DETR base model
    │   ├── RT-DETR-SEA/    # RT-DETR with SEA attention
    │   ├── SO_DETR/        # SO-DETR model
    │   ├── SpotNet/        # SpotNet model
    │   ├── FORT/           # FORT model
    │   ├── weight/         # Pre-trained weights
    │   ├── train.py        # Ultralytics training script
    │   └── train_deim.sh   # 🚀 DEIM training script
    └── tools/              # Utility tools
        ├── yolo2coco.py    # YOLO to COCO format converter
        ├── get_fps.py      # FPS benchmarking tool
        ├── export_onnx.py  # ONNX export utility
        └── infer_edge.py   # Edge device inference
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.11+ recommended
conda create -n deim python=3.11.9
conda activate deim

# Install DEIM requirements
cd train/model/DEIM
pip install -r requirements.txt
```

### Dataset Preparation

Organize your Fisheye8K dataset in COCO format:

```
dataset/fisheye8k/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

---

## 🏆 Training with DEIM (Recommended)

### Quick Start

```bash
cd train/model
chmod +x train_deim.sh
./train_deim.sh
```

### Custom Configuration

```bash
# Train with specific model size (n/s/m/l/x)
MODEL_SIZE=l ./train_deim.sh

# Multi-GPU training
NUM_GPUS=4 MODEL_SIZE=l ./train_deim.sh

# Resume from checkpoint
RESUME=/path/to/checkpoint.pth ./train_deim.sh

# Fine-tune from pretrained weights
TUNING=/path/to/pretrained.pth ./train_deim.sh
```

### Available Model Sizes

| Model          | Params | Latency | GFLOPs | AP (COCO) |
| -------------- | ------ | ------- | ------ | --------- |
| **N** (Nano)   | 4M     | 2.12ms  | 7      | 43.0      |
| **S** (Small)  | 10M    | 3.49ms  | 25     | 49.0      |
| **M** (Medium) | 19M    | 5.62ms  | 57     | 52.7      |
| **L** (Large)  | 31M    | 8.07ms  | 91     | 54.7      |
| **X** (XLarge) | 62M    | 12.89ms | 202    | 56.5      |

### Manual Training

```bash
cd train/model/DEIM

# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py \
    -c configs/deim_dfine/deim_hgnetv2_l_coco.yml \
    --use-amp \
    --seed=0

# Multi-GPU (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --master_port=7777 \
    --nproc_per_node=4 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_l_coco.yml \
    --use-amp \
    --seed=0
```

---

## 🔄 Alternative: Ultralytics Training

For quick experiments with Ultralytics RT-DETR:

```bash
cd train/model
python train.py
```

---

## 📊 Inference

### Using DEIM

```bash
cd train/model/DEIM

# PyTorch inference
python tools/inference/torch_inf.py \
    -c configs/deim_dfine/deim_hgnetv2_l_coco.yml \
    -r model.pth \
    --input image.jpg \
    --device cuda:0

# ONNX inference
python tools/inference/onnx_inf.py \
    --onnx model.onnx \
    --input image.jpg

# TensorRT inference
python tools/inference/trt_inf.py \
    --trt model.engine \
    --input image.jpg
```

### Using Ultralytics

```bash
python infer/inference.py <model_path> <image_path>
```

---

## 🔧 Utility Tools

### YOLO to COCO Format Converter

```bash
python train/tools/yolo2coco.py \
    --images_dir ./dataset/fisheye_test/images \
    --labels_dir ./runs/detect/predict/labels \
    --output ./submission/results.json \
    --conf 1 \
    --submission 1 \
    --is_fisheye8k 1
```

### Export to ONNX (DEIM)

```bash
cd train/model/DEIM
pip install onnx onnxsim
python tools/deployment/export_onnx.py \
    --check \
    -c configs/deim_dfine/deim_hgnetv2_l_coco.yml \
    -r model.pth
```

### Export to TensorRT

```bash
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

### FPS Benchmarking

```bash
cd train/model/DEIM
python tools/benchmark/get_info.py \
    -c configs/deim_dfine/deim_hgnetv2_l_coco.yml
```

---

## 📊 Models Explored

| Model       | Description                             | Status       |
| ----------- | --------------------------------------- | ------------ |
| **DEIM**    | DETR with Improved Matchin              |  **Final** |
| RT-DETR     | Real-Time DEtection TRansformer         | Baseline     |
| Drone-DETR  | Optimized for drone/aerial imagery      | Explored     |
| SO-DETR     | Small Object DETR variant               | Explored     |
| RT-DETR-SEA | RT-DETR with Squeeze-and-Excitation     | Explored     |
| SpotNet     | Specialized for small object detection  | Explored     |
| FORT        | Feature-enhanced Object Recognition     | Explored     |

---

## 📈 Datasets

- **Fisheye8K**: Primary dataset with fisheye camera images
- **VisDrone**: Drone-captured aerial images for pre-training
- **HIT-UAV**: UAV dataset for domain adaptation
- **Synthetic VisDrone**: Synthesized data for augmentation

---

## 🤝 Team

**Team Tennhom** - AI City Challenge 2025

---

## 📄 License

This project is for the AI City Challenge 2025 competition.

---

## 📚 References

- [DEIM Paper](https://arxiv.org/abs/2412.04234)
- [DEIM GitHub](https://github.com/Intellindust-AI-Lab/DEIM)
- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [AI City Challenge](https://www.aicitychallenge.org/)

---
