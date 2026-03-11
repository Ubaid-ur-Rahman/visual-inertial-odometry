Visual-Inertial Odometry Transformer 🚗

A deep learning framework for Visual-Inertial Odometry (VIO) using a Transformer-based sensor fusion architecture.
The model estimates 6-DoF relative pose (translation + rotation) from:

Image pairs

IMU measurements

The system supports multiple datasets through a universal dataloader, currently including:

EuRoC MAV Dataset

KITTI Odometry Dataset

Overview

Visual-Inertial Odometry is a key component of autonomous systems such as:

self-driving cars

drones

robotics

AR/VR

This project implements a Transformer-based VIO model that fuses:

visual tokens (patch embeddings from images)

IMU tokens

inside a unified transformer encoder.

The model predicts the relative pose between frames.

Architecture

The model consists of:

Vision Patch Embedding

Images are split into patches using a convolutional projection similar to a Vision Transformer.

Image pair (6 channels)
      │
Patch Embedding
      │
Visual Tokens

IMU Embedding

IMU measurements are embedded into tokens using a linear projection.

IMU sequence
      │
Linear Projection
      │
IMU Tokens

Transformer Fusion

Visual and IMU tokens are concatenated and passed into a transformer encoder.

[CLS] + Visual Tokens + IMU Tokens
             │
        Transformer
             │
          CLS token

Pose Head

The CLS token is used to predict:

Translation (x, y, z)
Quaternion (qw, qx, qy, qz)
Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/visual-inertial-odometry.git
cd visual-inertial-odometry

Install dependencies:

pip install -r requirements.txt
Training

Example training command using the KITTI dataset:

python3 train.py /path/to/dataset/kitti --epochs 10 --batch-size 4

Example:

python3 train.py /home/ubaid/Downloads/Autonomous_driving/visual-inertial-odometry/dataset/kitti --epochs 10 --batch-size 4

During training the script will:

split dataset into train / validation

save best model checkpoint

generate training loss plots

Saved outputs:

checkpoints/
   run_TIMESTAMP/
      best_model.pth
      loss_curve.png
Dataset Support

The project currently supports:

EuRoC MAV Dataset
dataset/
   euroc/
      V2_03_difficult/
         mav0/
            cam0/data/*.png
            imu0/data.csv
            state_groundtruth_estimate0/data.csv
      MH_02_easy/
      V1_01_easy/

Dataset link:

https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

KITTI Odometry Dataset

Expected structure:

dataset/
   kitti/
      sequences/
         00/
            image_0/*.png
         01/
         02/
      poses/
         00.txt
         01.txt

Dataset link:

http://www.cvlibs.net/datasets/kitti/eval_odometry.php

Note:

KITTI odometry dataset does not include IMU data, therefore IMU inputs are filled with zeros.

Project Structure
visual-inertial-odometry
│
├── dataloader
│   └── dataloader.py
│
├── model
│   └── models.py
│
├── utils
│   └── custom_transforms.py
│
├── train.py
├── requirements.txt
└── README.md
Data Processing

The dataloader:

loads sequences of images

synchronizes IMU data

computes ground truth poses

resizes images to 640 × 192

Each training sample contains:

imgs  -> image sequence
imus  -> IMU measurements
poses -> ground truth camera poses
Training Objective

The model predicts:

[tx, ty, tz, qw, qx, qy, qz]

Loss function:

Loss = Translation MSE + 10 × Rotation Loss

Rotation loss uses geodesic quaternion distance.

Model Details

Transformer parameters:

Embedding dimension : 256
Transformer layers  : 6
Attention heads     : 8
Patch size          : 16

Image resolution:

640 × 192
Example Training Output
Epoch 1 | Step 0/2317 | Loss 0.92
Epoch 1 | Step 10/2317 | Loss 0.71
Epoch 1 | Step 20/2317 | Loss 0.68