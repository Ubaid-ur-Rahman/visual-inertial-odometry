import os
import argparse
import time
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.data
import custom_transforms

from models import VisionEncoder, ImuEncoder, FusionModule, TemporalModule, PoseRegressor
from utils import save_path_formatter
from data_loader_euroc import EuRoC_Loader


# =========================================================
# Argument Parser
# =========================================================

parser = argparse.ArgumentParser(
    description='Selective Sensor Fusion (Clean Version)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--sequence-length', type=int, default=3)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight-decay', default=0, type=float)
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--seed', default=0, type=int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Main
# =========================================================

def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    train_losses = []
    val_losses = []
    best_epoch = 0

    # -------------------------
    # Save directory
    # -------------------------
    timestamp = time.strftime("%m-%d-%H-%M")
    save_path = Path("checkpoints") / f"run_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"=> Saving to {save_path}")

    # -------------------------
    # Transforms
    # -------------------------
    normalize = custom_transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])
    normalize2 = custom_transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])

    input_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize,
        normalize2
    ])

    # -------------------------
    # Dataset
    # -------------------------
    train_set = EuRoC_Loader(
        root=args.data,
        seed=args.seed,
        train=1,
        sequence_length=args.sequence_length,
        transform=input_transform
    )

    val_set = EuRoC_Loader(
        root=args.data,
        seed=args.seed,
        train=0,
        sequence_length=args.sequence_length,
        transform=input_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=EuRoC_Loader.euroc_collate
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=EuRoC_Loader.euroc_collate
    )

    print(f"{len(train_set)} train samples")
    print(f"{len(val_set)} val samples")

    # -------------------------
    # Model
    # -------------------------
    feature_dim = 256

    vision_encoder = VisionEncoder(feature_dim).to(device)
    imu_encoder = ImuEncoder(feature_dim).to(device)
    fusion_module = FusionModule(feature_dim).to(device)
    temporal_module = TemporalModule(feature_dim).to(device)
    pose_net = PoseRegressor(feature_dim).to(device)

    parameters = (
        list(vision_encoder.parameters()) +
        list(imu_encoder.parameters()) +
        list(fusion_module.parameters()) +
        list(temporal_module.parameters()) +
        list(pose_net.parameters())
    )

    optimizer = torch.optim.Adam(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # -------------------------
    # Training Loop
    # -------------------------
    best_val = float('inf')

    for epoch in range(args.epochs):

        train_loss = train_one_epoch(
            train_loader,
            vision_encoder,
            imu_encoder,
            fusion_module,
            temporal_module,
            pose_net,
            optimizer,
            epoch,
            args
        )

        val_loss = validate(
            val_loader,
            vision_encoder,
            imu_encoder,
            fusion_module,
            temporal_module,
            pose_net,
            epoch,
            args
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.5f} "
              f"Val Loss: {val_loss:.5f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save({
                'vision': vision_encoder.state_dict(),
                'imu': imu_encoder.state_dict(),
                'fusion': fusion_module.state_dict(),
                'temporal': temporal_module.state_dict(),
                'pose': pose_net.state_dict(),
            }, save_path / "best_model.pth")

            print("Best model saved.")

    # -------------------------------------------------
    # Plot Loss Curve
    # -------------------------------------------------
    epochs_range = range(1, args.epochs + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')

    plt.axvline(x=best_epoch, linestyle='--', label=f'Best Epoch ({best_epoch})')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path / "loss_curve.png")
    plt.close()

    print(f"Loss curve saved to {save_path}/loss_curve.png")


# =========================================================
# Training
# =========================================================

def train_one_epoch(loader,
                    vision_encoder,
                    imu_encoder,
                    fusion_module,
                    temporal_module,
                    pose_net,
                    optimizer,
                    epoch,
                    args):

    vision_encoder.train()
    imu_encoder.train()
    fusion_module.train()
    temporal_module.train()
    pose_net.train()

    total_loss = 0
    n_batches = 0

    for i, (imgs, imus, poses) in enumerate(loader):

        seq_features = []

        # Process sequence
        for j in range(len(imgs) - 1):

            tgt = imgs[j + 1].to(device)
            ref = imgs[j].to(device)
            imu = imus[j].to(device)

            vision_feat = vision_encoder(tgt, ref)
            imu_feat = imu_encoder(imu)

            fused = fusion_module(vision_feat, imu_feat)
            seq_features.append(fused.unsqueeze(1))

        seq_features = torch.cat(seq_features, dim=1)

        temporal_feat = temporal_module(seq_features)
        pred_pose = pose_net(temporal_feat)

        # ---- Ground Truth ----
        gt_trans, gt_quat = compute_trans_pose(
            poses[0].numpy().astype(np.float64),
            poses[-1].numpy().astype(np.float64)
        )

        gt_translation = torch.tensor(
            gt_trans,
            dtype=torch.float32,
            device=device
        )

        gt_quat = torch.tensor(
            gt_quat,
            dtype=torch.float32,
            device=device
        )

        # ---- Loss ----
        trans_loss = F.mse_loss(pred_pose[:, :3], gt_translation)
        rot_loss = F.mse_loss(pred_pose[:, 3:], gt_quat)

        loss = trans_loss + 10.0 * rot_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if i % args.print_freq == 0:
            print(f"Epoch {epoch+1} Step {i}/{len(loader)} "
                  f"Loss: {loss.item():.5f}")

    return total_loss / n_batches


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate(loader,
             vision_encoder,
             imu_encoder,
             fusion_module,
             temporal_module,
             pose_net,
             epoch,
             args):

    vision_encoder.eval()
    imu_encoder.eval()
    fusion_module.eval()
    temporal_module.eval()
    pose_net.eval()

    total_loss = 0
    n_batches = 0

    for imgs, imus, poses in loader:

        seq_features = []

        for j in range(len(imgs) - 1):

            tgt = imgs[j + 1].to(device)
            ref = imgs[j].to(device)
            imu = imus[j].to(device)

            vision_feat = vision_encoder(tgt, ref)
            imu_feat = imu_encoder(imu)

            fused = fusion_module(vision_feat, imu_feat)
            seq_features.append(fused.unsqueeze(1))

        seq_features = torch.cat(seq_features, dim=1)

        temporal_feat = temporal_module(seq_features)
        pred_pose = pose_net(temporal_feat)

        gt_trans, gt_quat = compute_trans_pose(
            poses[0].numpy().astype(np.float64),
            poses[-1].numpy().astype(np.float64)
        )

        gt_translation = torch.tensor(
            gt_trans,
            dtype=torch.float32,
            device=device
        )

        gt_quat = torch.tensor(
            gt_quat,
            dtype=torch.float32,
            device=device
        )

        trans_loss = F.mse_loss(pred_pose[:, :3], gt_translation)
        rot_loss = torch.mean(1 - (pred_pose[:, 3:] * gt_quat).sum(dim=1).pow(2))  # Geodesic
        loss = trans_loss + 10.0 * rot_loss

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# =========================================================
# Pose Computation
# =========================================================

from scipy.spatial.transform import Rotation

def compute_trans_pose(ref_pose, tgt_pose):
    tmp_pose = np.copy(tgt_pose)
    tmp_pose[:, :, -1] -= ref_pose[:, :, -1]
    trans_pose = np.linalg.inv(ref_pose[:, :, :3]) @ tmp_pose
    rel_rot_matrix = trans_pose[:, :3, :3]  # Batch of rotation matrices
    translations = trans_pose[:, :, -1]  # Batch of translations
    
    # Convert each rotation matrix to quaternion [w, x, y, z]
    quats = []
    for i in range(rel_rot_matrix.shape[0]):
        rot = Rotation.from_matrix(rel_rot_matrix[i])
        quat = rot.as_quat()  # [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # To [w, x, y, z]
        quats.append(quat)
    quats = np.array(quats)
    
    return translations, quats  # Return both


if __name__ == '__main__':
    main()