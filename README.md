# 🚗 Visual-Inertial Odometry (Deep Learning)

A deep learning framework for **Visual-Inertial Odometry (VIO)** using two fundamentally different architectures:

- 🧠 Transformer-based cross-modal fusion  
- 🔁 CNN + LSTM recurrent fusion  

The system estimates **6-DoF relative pose (translation + rotation)** from:

- 📷 Image pairs  
- 📡 IMU measurements  

---

## 🎥 Demo (Transformer Model)

Below is a sample trajectory output from the Transformer-based model:

> ⚠️ GitHub may not autoplay videos — click to view

<video src="smooth_vio.mp4" controls width="100%"></video>

---

## 📌 Overview

Visual-Inertial Odometry is a core component in:

- 🚗 Autonomous driving  
- 🚁 Drones  
- 🤖 Robotics  
- 🥽 AR/VR  

This project explores **two fundamentally different learning paradigms** for VIO:

| Model | Type | Fusion Strategy |
|------|------|----------------|
| Transformer VIO | Attention-based | Cross-modal attention |
| CNN + LSTM VIO | Recurrent | Feature concatenation |

This enables a **direct comparison between attention vs recurrence** in sensor fusion.

---

# 🧠 Model 1: Transformer-based VIO (Proposed)

## 🔍 Key Features

- Cross-modal attention between vision and IMU  
- Sinusoidal positional encoding (temporal + spatial)  
- Pre-layer normalization (stable training)  
- Bidirectional fusion (vision ↔ IMU)  
- Global reasoning via Transformer  

---

## 🏗 Architecture

![Transformer Architecture](images/model_transformer.png)

### Pipeline
Image Pair (6 channels)
│
CNN Stem
│
Patch Embedding
│
Vision Tokens ─────────────┐
│
Cross
│
IMU Sequence → IMU Encoder ┘
│
Transformer Encoder
│
CLS Token
│
Pose Head


---

## ⚙️ Key Improvements Over Standard ViT

- ✅ CNN feature extractor before patching  
- ✅ Temporal Transformer for IMU  
- ✅ Cross-attention fusion (not simple concatenation)  
- ✅ Pre-LN Transformer (more stable training)  
- ✅ Learned CLS positional embedding  
- ✅ Quaternion normalization  

---

# 🔁 Model 2: CNN + LSTM VIO (Baseline)

## 🔍 Key Features

- CNN-based visual feature extraction (FlowNet-style)  
- IMU encoded via MLP  
- Temporal modeling via LSTM  
- Late fusion of visual + IMU features  

---

## 🏗 Architecture

![CNN LSTM Architecture](images/model_cnn_lstm.png)

### Pipeline
Image Pair
│
CNN Encoder
│
Visual Features ─────┐
│
IMU → MLP Encoder ──┘
│
Concatenate
│
LSTM
│
Fully Connected
│
Pose Output



---

# 🧪 Why Two Models?

This project is designed for **scientific comparison**:

| Property | Transformer | CNN + LSTM |
|--------|------------|------------|
| Spatial modeling | Global attention | Local CNN |
| Temporal modeling | Attention | Recurrence |
| Fusion | Cross-attention | Concatenation |
| Parallelism | High | Low |
| Memory usage | High | Moderate |

---

# 📊 Training Objective

The model predicts:
[tx, ty, tz, qw, qx, qy, qz]


### Loss Function


Loss = Translation MSE + 10 × Rotation Loss


- Translation → Mean Squared Error  
- Rotation → Geodesic quaternion loss  

---

# 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/visual-inertial-odometry.git
cd visual-inertial-odometry

pip install -r requirements.txt

🚀 Training
Example (KITTI)
python3 train.py /path/to/kitti --epochs 10 --batch-size 4
Example (EuRoC)
python3 train.py /path/to/euroc --epochs 10 --batch-size 4
💾 Training Output
checkpoints/
   run_TIMESTAMP/
      best_model.pth
      loss_curve.png
🗂 Dataset Support
1️⃣ EuRoC MAV Dataset
dataset/
   euroc/
      V2_03_difficult/
         mav0/
            cam0/data/*.png
            imu0/data.csv
            state_groundtruth_estimate0/data.csv

🔗 https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

2️⃣ KITTI Odometry Dataset
dataset/
   kitti/
      sequences/
         00/image_0/*.png
      poses/00.txt

🔗 http://www.cvlibs.net/datasets/kitti/eval_odometry.php

⚠️ Note: KITTI has no IMU → IMU input is zero-filled.

🧱 Project Structure
visual-inertial-odometry
│
├── dataloader/
│   └── dataloader.py
│
├── model/
│   ├── models.py              # Transformer model
│   ├── model_cnn_lstm.py     # CNN + LSTM model
│
├── utils/
│   └── custom_transforms.py
│
├── images/
│   ├── model_transformer.png
│   ├── model_cnn_lstm.png
│   └── smooth_vio.mp4
│
├── train.py
├── requirements.txt
└── README.md
🔄 Data Processing

The dataloader:

Loads image sequences
Synchronizes IMU data
Computes relative poses
Resizes images → 640 × 192

Each sample contains:

imgs  → image sequence
imus  → IMU data
poses → ground truth
⚙️ Model Configuration
Transformer
Embedding dim: 256
Layers: 6
Heads: 8
Input
Image: 640 × 192
IMU sequence length: 3
📈 Example Training Output
Epoch 1 | Step 0/2317 | Loss 0.92
Epoch 1 | Step 10/2317 | Loss 0.71
Epoch 1 | Step 20/2317 | Loss 0.68
🧪 Suggested Evaluation Metrics

For proper comparison:

Absolute Trajectory Error (ATE)
Relative Pose Error (RPE)
Rotation Error
Inference speed (FPS)
GPU memory usage
🔬 Future Work
Uncertainty-aware training (partially implemented)
Multi-scale visual features
LiDAR + camera fusion
Self-supervised VIO
Loop closure integration
📜 License

MIT License
