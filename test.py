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
# Pose integration
# =========================================================
def integrate_pose(prev_pose, rel_pose):
    t = rel_pose[:3]
    q = rel_pose[3:]

    qw, qx, qy, qz = q
    R_rel = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])

    new_pose = np.eye(4)
    new_pose[:3, :3] = prev_pose[:3, :3] @ R_rel
    new_pose[:3, 3] = prev_pose[:3, 3] + prev_pose[:3, :3] @ t

    return new_pose


def compute_relative_pose(ref, tgt):
    rel = np.linalg.inv(ref) @ tgt
    R = rel[:3, :3]
    t = rel[:3, 3]

    quat = Rotation.from_matrix(R).as_quat()
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])

    return np.concatenate([t, quat])


def to_4x4(p):
    if p.shape == (3, 4):
        T = np.eye(4)
        T[:3, :] = p
        return T
    return p


# =========================================================
# Smooth trajectory
# =========================================================
def smooth_traj(traj, alpha=0.6):
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

    prev_img_tensor = None

    with torch.no_grad():
        for i, (imgs, imus, poses) in enumerate(loader):

            if i >= max_frames:
                break

            curr_img = imgs[:, -1].to(device)

            if prev_img_tensor is None:
                prev_img_tensor = curr_img
                continue

            # Instead of prev_img_tensor logic
            ref_img = imgs[:, 0].to(device)
            tgt_img = imgs[:, -1].to(device)

            img_pair = torch.cat([ref_img, tgt_img], dim=1)
            prev_img_tensor = curr_img

            imu_seq = imus.to(device)

            pred = model(img_pair, imu_seq)[0].cpu().numpy()

            # -------------------------
            # GT pose (FIRST!)
            # -------------------------
            poses_np = poses.numpy()
            ref_pose = to_4x4(poses_np[0, -2])
            tgt_pose = to_4x4(poses_np[0, -1])

            rel_gt = compute_relative_pose(ref_pose, tgt_pose)

            gt_pose = integrate_pose(gt_pose, rel_gt)
            gt_traj.append(gt_pose[:3, 3])

            # -------------------------
            # Pose integration (prediction)
            # -------------------------
            gt_scale = np.linalg.norm(rel_gt[:3])

            pred_dir = pred[:3] / (np.linalg.norm(pred[:3]) + 1e-8)
            t = pred_dir * gt_scale

            q_model = pred[3:]
            # 🔥 NORMALIZE quaternion (VERY IMPORTANT)
            q_model = q_model / (np.linalg.norm(q_model) + 1e-8)

            # Optional: enforce consistent hemisphere
            if q_model[0] < 0:
                q_model = -q_model
            pred_rel = np.concatenate([t, q_model])

            pred_pose = integrate_pose(pred_pose, pred_rel)
            pred_traj.append(pred_pose[:3, 3])

            # -------------------------
            # Metrics
            # -------------------------
            if len(pred_traj) > 1:
                vel = np.linalg.norm(pred_traj[-1] - pred_traj[-2]) * args.fps
            else:
                vel = 0.0

            error = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
            print("Pred t:", np.linalg.norm(t), "GT t:", np.linalg.norm(rel_gt[:3]), "Error:", error)
            # -------------------------
            # Image
            # -------------------------
            img = curr_img[0].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)

            overlay = img.copy()

            cv2.putText(overlay, f"Frame: {i}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(overlay, f"Velocity: {vel:.2f} m/s", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(overlay, f"Error: {error:.2f} m", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # -------------------------
            # Trajectory
            # -------------------------
            pred_np = smooth_traj(np.array(pred_traj))
            gt_np = smooth_traj(np.array(gt_traj))

            ax.clear()

            ax.plot(gt_np[:, 0], gt_np[:, 2],
                    color='lime', linewidth=2, label='GT')

            ax.plot(pred_np[:, 0], pred_np[:, 2],
                    color='dodgerblue', linewidth=2, linestyle='--', label='Pred')

            # fading trail
            for k in range(1, len(pred_np)):
                alpha = k / len(pred_np)
                ax.plot(pred_np[k-1:k+1, 0],
                        pred_np[k-1:k+1, 2],
                        color='blue', alpha=alpha)

            # direction arrow
            if len(pred_np) > 1:
                dx = pred_np[-1, 0] - pred_np[-2, 0]
                dz = pred_np[-1, 2] - pred_np[-2, 2]
                ax.arrow(pred_np[-1, 0], pred_np[-1, 2],
                         dx, dz, head_width=0.5, color='blue')

            # center view
            cx, cz = gt_np[-1, 0], gt_np[-1, 2]
            ax.set_xlim(cx - 20, cx + 20)
            ax.set_ylim(cz - 20, cz + 20)

            ax.legend()
            ax.set_title("Trajectory (Top View)")
            ax.grid(True)
            ax.set_aspect('equal')

            fig.canvas.draw()
            traj_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            traj_img = cv2.resize(traj_img, (img.shape[1], img.shape[0]))

            # -------------------------
            # Layout
            # -------------------------
            overlay = cv2.copyMakeBorder(overlay, 10,10,10,10,
                                        cv2.BORDER_CONSTANT, value=(255,255,255))
            traj_img = cv2.copyMakeBorder(traj_img, 10,10,10,10,
                                         cv2.BORDER_CONSTANT, value=(255,255,255))

            combined = np.hstack((overlay, traj_img))

            banner = np.ones((50, combined.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(banner,
                        f"Visual-Inertial Odometry | {args.model.upper()}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

            combined = np.vstack((banner, combined))
            frames.append(combined)

            if i % 20 == 0:
                print(f"Frame {i}/{max_frames}")

    plt.close(fig)

    if len(frames) == 0:
        print("❌ No frames generated!")
        return

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