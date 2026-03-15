import os
import argparse
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.data
import utils.custom_transforms as custom_transforms

from model.model_transformer import VIO_TRANSFORMER
from model.model_cnn_lstm import VIO_CNN_LSTM
from dataloader.dataloader import UniversalVIODataset as VIO_Dataset

from scipy.spatial.transform import Rotation


# =========================================================
# Argument Parser
# =========================================================
parser = argparse.ArgumentParser(
    description='Visual-Inertial Odometry Training',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('data', metavar='DIR', help='path to dataset')

parser.add_argument(
    '--dataset',
    default='kitti',
    choices=['kitti', 'euroc', 'tum'],
    help='dataset type'
)

parser.add_argument(
    '--model',
    default='cnn_lstm',
    choices=['cnn_lstm', 'transformer'],
    help='model architecture'
)

parser.add_argument('--sequence-length', type=int, default=3)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight-decay', default=1e-5, type=float)
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
    best_val = float('inf')

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
    dataset_root = args.data
    dataset_type = args.dataset

    print(f"Using dataset: {dataset_type}")
    print(f"Dataset path: {dataset_root}")

    train_set = VIO_Dataset(
        root=dataset_root,
        dataset_type=dataset_type,
        sequence_length=args.sequence_length,
        train=True
    )

    val_set = VIO_Dataset(
        root=dataset_root,
        dataset_type=dataset_type,
        sequence_length=args.sequence_length,
        train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    print(f"{len(train_set)} train samples")
    print(f"{len(val_set)} val samples")

    # -------------------------
    # Model Selection
    # -------------------------
    if args.model == "cnn_lstm":
        print("Using CNN-LSTM model")
        model = VIO_CNN_LSTM().to(device)

    elif args.model == "transformer":
        print("Using Transformer VIO model")
        model = VIO_TRANSFORMER().to(device)

    else:
        raise ValueError("Unknown model type")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # =====================================================
    # Training Loop
    # =====================================================
    for epoch in range(args.epochs):

        train_loss = train_one_epoch(
            train_loader, model, optimizer, epoch, args
        )

        val_loss = validate(
            val_loader, model, epoch
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train: {train_loss:.5f}  "
              f"Val: {val_loss:.5f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1

            torch.save(model.state_dict(), save_path / "best_model.pth")
            print("✓ Best model saved.")

    # =====================================================
    # Plot Loss Curve
    # =====================================================
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val')
    plt.axvline(best_epoch, linestyle='--', label=f'Best ({best_epoch})')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path / "loss_curve.png")
    plt.close()

    print("Training complete.")


# =========================================================
# Training Step
# =========================================================
def train_one_epoch(loader, model, optimizer, epoch, args):

    model.train()
    total_loss = 0

    for i, (imgs, imus, poses) in enumerate(loader):

        tgt = imgs[:, -1].to(device)
        ref = imgs[:, 0].to(device)

        img_pair = torch.cat([tgt, ref], dim=1)

        imu_seq = imus.to(device)

        pred_pose = model(img_pair, imu_seq)
        poses_np = poses.numpy().astype(np.float64)

        gt_trans, gt_quat = compute_trans_pose(
            poses_np[:, 0],
            poses_np[:, -1]
        )

        gt_translation = torch.from_numpy(gt_trans).float().to(device)
        gt_quat = torch.from_numpy(gt_quat).float().to(device)

        trans_loss = F.mse_loss(pred_pose[:, :3], gt_translation)

        # Geodesic quaternion loss
        dot = torch.sum(pred_pose[:, 3:] * gt_quat, dim=1)
        rot_loss = torch.mean(1 - dot.pow(2))

        loss = trans_loss + 10.0 * rot_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % args.print_freq == 0:
            print(f"Epoch {epoch+1} | Step {i}/{len(loader)} | "
                  f"Loss {loss.item():.5f}")

    return total_loss / len(loader)


# =========================================================
# Validation
# =========================================================
@torch.no_grad()
def validate(loader, model, epoch):

    model.eval()
    total_loss = 0

    for imgs, imus, poses in loader:

        tgt = imgs[:, -1].to(device)
        ref = imgs[:, 0].to(device)
        imu_seq = imus.to(device)

        img_pair = torch.cat([tgt, ref], dim=1)
        pred_pose = model(img_pair, imu_seq)

        # Compute GT for the entire batch
        poses_np = poses.numpy().astype(np.float64)
        gt_trans, gt_quat = compute_trans_pose(
            poses_np[:, 0],
            poses_np[:, -1]
        )

        gt_translation = torch.from_numpy(gt_trans).float().to(device)
        gt_quat = torch.from_numpy(gt_quat).float().to(device)

        # Compute losses
        trans_loss = F.mse_loss(pred_pose[:, :3], gt_translation)
        dot = torch.sum(pred_pose[:, 3:] * gt_quat, dim=1)
        rot_loss = torch.mean(1 - dot.pow(2))

        loss = trans_loss + 10.0 * rot_loss
        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# Relative Pose Computation
# =========================================================
def compute_trans_pose(ref_pose, tgt_pose):

    tmp_pose = np.copy(tgt_pose)
    tmp_pose[:, :, -1] -= ref_pose[:, :, -1]

    trans_pose = np.linalg.inv(ref_pose[:, :, :3]) @ tmp_pose

    rel_rot_matrix = trans_pose[:, :3, :3]
    translations = trans_pose[:, :, -1]

    quats = []
    for i in range(rel_rot_matrix.shape[0]):
        rot = Rotation.from_matrix(rel_rot_matrix[i])
        quat = rot.as_quat()
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        quats.append(quat)

    quats = np.array(quats)

    return translations, quats


if __name__ == '__main__':
    main()