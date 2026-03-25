import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation

from model.model_transformer import VIO_TRANSFORMER
from model.model_cnn_lstm import VIO_CNN_LSTM
from dataloader.dataloader import UniversalVIODataset as VIO_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Pose integration (relative → global)
# =========================================================
def integrate_pose(prev_pose, rel_pose):
    t = rel_pose[:3]
    q = rel_pose[3:]

    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return prev_pose @ T


def compute_relative_pose(ref, tgt):
    rel = np.linalg.inv(ref) @ tgt

    R = rel[:3, :3]
    t = rel[:3, 3]

    quat = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # → [w,x,y,z]

    return np.concatenate([t, quat])


def to_4x4(p):
    if p.shape == (3, 4):
        T = np.eye(4)
        T[:3, :] = p
        return T
    return p


# =========================================================
# Smooth trajectory (important for visualization)
# =========================================================
def smooth_traj(traj, alpha=0.9):
    if len(traj) == 0:
        return traj
    smoothed = []
    prev = traj[0]
    for p in traj:
        prev = alpha * prev + (1 - alpha) * p
        smoothed.append(prev)
    return np.array(smoothed)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="transformer",
                        choices=["transformer", "cnn_lstm"])
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=15)

    args = parser.parse_args()

    dataset = VIO_Dataset(
        root=args.data,
        dataset_type="kitti",
        sequence_length=3,
        train=False
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -------------------------
    # Model
    # -------------------------
    if args.model == "transformer":
        model = VIO_TRANSFORMER().to(device)
    else:
        model = VIO_CNN_LSTM().to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print("✅ Model loaded")

    pred_traj = []
    gt_traj = []

    pred_pose = np.eye(4)
    gt_pose = np.eye(4)

    max_frames = args.fps * args.seconds
    frames = []

    fig, ax = plt.subplots(figsize=(5, 5))

    prev_img_tensor = None  # 🔥 KEY FIX: ensures temporal continuity

    with torch.no_grad():
        for i, (imgs, imus, poses) in enumerate(loader):

            if i >= max_frames:
                break

            # -------------------------
            # Use sequential frames
            # -------------------------
            curr_img = imgs[:, 0].to(device)  # FIX: use consistent frame

            if prev_img_tensor is None:
                prev_img_tensor = curr_img
                continue

            img_pair = torch.cat([curr_img, prev_img_tensor], dim=1)
            prev_img_tensor = curr_img

            imu_seq = imus.to(device)

            pred = model(img_pair, imu_seq)[0].cpu().numpy()

            # -------------------------
            # Quaternion fix
            # -------------------------
            t = pred[:3]
            q_model = pred[3:]
            q_fixed = np.array([q_model[3], q_model[0], q_model[1], q_model[2]])
            pred_rel = np.concatenate([t, q_fixed])

            pred_pose = integrate_pose(pred_pose, pred_rel)
            pred_traj.append(pred_pose[:3, 3])

            # -------------------------
            # Ground truth
            # -------------------------
            poses_np = poses.numpy()

            ref_pose = to_4x4(poses_np[0, 0])
            tgt_pose = to_4x4(poses_np[0, -1])

            rel_gt = compute_relative_pose(ref_pose, tgt_pose)
            gt_pose = integrate_pose(gt_pose, rel_gt)

            gt_traj.append(gt_pose[:3, 3])

            # -------------------------
            # Image
            # -------------------------
            img = curr_img[0].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)

            # -------------------------
            # Smooth trajectories
            # -------------------------
            pred_np = smooth_traj(np.array(pred_traj))
            gt_np = smooth_traj(np.array(gt_traj))

            # -------------------------
            # Plot
            # -------------------------
            ax.clear()

            ax.plot(gt_np[:, 0], gt_np[:, 2], 'g-', label='GT')
            ax.plot(pred_np[:, 0], pred_np[:, 2], 'b-', label='Pred')

            ax.scatter(gt_np[-1, 0], gt_np[-1, 2], c='g', s=60)
            ax.scatter(pred_np[-1, 0], pred_np[-1, 2], c='b', s=60)

            ax.legend()
            ax.set_title("Trajectory (X-Z)")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box')

            fig.canvas.draw()
            traj_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

            traj_img = cv2.resize(traj_img, (img.shape[1], img.shape[0]))

            combined = np.hstack((img, traj_img))
            frames.append(combined)

            if i % 20 == 0:
                print(f"Frame {i}/{max_frames}")

    plt.close(fig)

    if len(frames) == 0:
        print("❌ No frames generated!")
        return

    # -------------------------
    # Save video
    # -------------------------
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps,
        (w, h)
    )

    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    out.release()

    print(f"🎬 Video saved: {args.output}")


if __name__ == "__main__":
    main()