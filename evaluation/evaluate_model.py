import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from model.model_transformer import VIO_TRANSFORMER
from model.model_cnn_lstm import VIO_CNN_LSTM
from dataloader.dataloader import UniversalVIODataset as VIO_Dataset
import utils.custom_transforms as custom_transforms


# ────────────────────────────────────────────────
# Pose utilities
# ────────────────────────────────────────────────

def pose_vec_to_SE3(p):
    """[tx ty tz qx qy qz qw] → 4x4 matrix"""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(p[3:]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def SE3_to_pose_vec(T):
    """4x4 matrix → [tx ty tz qx qy qz qw]"""
    t = T[:3, 3]
    q = R.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([t, q])


def integrate_trajectory(rel_poses):
    """
    Convert relative poses → global trajectory
    rel_poses: [N,7]
    """
    global_poses = []

    T_global = np.eye(4)
    global_poses.append(SE3_to_pose_vec(T_global))

    for p in rel_poses:
        T_rel = pose_vec_to_SE3(p)
        T_global = T_global @ T_rel
        global_poses.append(SE3_to_pose_vec(T_global))

    return np.array(global_poses)


# ────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────

def compute_ate(est, gt):
    """Absolute Trajectory Error (RMSE)"""
    est_t = est[:, :3]
    gt_t = gt[:, :3]

    diff = est_t - gt_t
    rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmse


def compute_rpe(est, gt):
    """
    Relative Pose Error
    """
    errors_t = []
    errors_r = []

    for i in range(len(est) - 1):

        T_est1 = pose_vec_to_SE3(est[i])
        T_est2 = pose_vec_to_SE3(est[i+1])

        T_gt1 = pose_vec_to_SE3(gt[i])
        T_gt2 = pose_vec_to_SE3(gt[i+1])

        delta_est = np.linalg.inv(T_est1) @ T_est2
        delta_gt = np.linalg.inv(T_gt1) @ T_gt2

        error = np.linalg.inv(delta_gt) @ delta_est

        trans_error = np.linalg.norm(error[:3, 3])
        rot_error = R.from_matrix(error[:3, :3]).magnitude()

        errors_t.append(trans_error)
        errors_r.append(rot_error)

    return np.mean(errors_t), np.mean(errors_r)


def compute_kitti_errors(est, gt, lengths=[100,200,300,400,500,600,700,800]):

    est_T = np.array([pose_vec_to_SE3(p) for p in est])
    gt_T  = np.array([pose_vec_to_SE3(p) for p in gt])

    t_errs = []
    r_errs = []

    for length in lengths:

        for i in range(len(gt_T)):

            for j in range(i+1, len(gt_T)):

                dist = np.linalg.norm(gt_T[j,:3,3] - gt_T[i,:3,3])

                if abs(dist - length) < 3:

                    delta_gt = np.linalg.inv(gt_T[i]) @ gt_T[j]
                    delta_est = np.linalg.inv(est_T[i]) @ est_T[j]

                    error = np.linalg.inv(delta_gt) @ delta_est

                    t_err = np.linalg.norm(error[:3,3]) / length * 100
                    r_err = R.from_matrix(error[:3,:3]).magnitude() * 180 / np.pi / length

                    t_errs.append(t_err)
                    r_errs.append(r_err)

                    break

    if len(t_errs) == 0:
        return 0,0

    return np.mean(t_errs), np.mean(r_errs)


# ────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────

def inference_one_epoch(loader, model, device):

    model.eval()

    pred_rel = []
    gt_rel = []

    with torch.no_grad():

        for imgs, imus, poses_gt in tqdm(loader):

            tgt = imgs[:, -1].to(device)
            ref = imgs[:, 0].to(device)
            imu_seq = imus.to(device)

            img_pair = torch.cat([tgt, ref], dim=1)

            pred = model(img_pair, imu_seq)

            q = F.normalize(pred[:,3:], dim=1)
            pred_pose = torch.cat([pred[:,:3], q], dim=1).cpu().numpy()

            poses_np = poses_gt.numpy().astype(np.float64)

            gt_trans, gt_quat = compute_trans_pose(poses_np[:,0], poses_np[:,-1])
            gt_pose = np.concatenate([gt_trans, gt_quat], axis=1)

            pred_rel.append(pred_pose)
            gt_rel.append(gt_pose)

    pred_rel = np.concatenate(pred_rel)
    gt_rel = np.concatenate(gt_rel)

    return pred_rel, gt_rel


# ────────────────────────────────────────────────
# TUM export
# ────────────────────────────────────────────────

def save_tum(path, poses):

    with open(path, "w") as f:

        for i,p in enumerate(poses):

            ts = i * 0.1
            t = p[:3]
            q = p[3:]

            f.write(
                f"{ts:.6f} {t[0]} {t[1]} {t[2]} "
                f"{q[0]} {q[1]} {q[2]} {q[3]}\n"
            )


# ────────────────────────────────────────────────
# compute_trans_pose (same as training)
# ────────────────────────────────────────────────

def compute_trans_pose(ref_pose, tgt_pose):

    tmp_pose = np.copy(tgt_pose)
    tmp_pose[:, :, -1] -= ref_pose[:, :, -1]

    trans_pose = np.linalg.inv(ref_pose[:, :, :3]) @ tmp_pose

    rel_rot_matrix = trans_pose[:, :3, :3]
    translations = trans_pose[:, :, -1]

    quats = []

    for i in range(rel_rot_matrix.shape[0]):
        quat = R.from_matrix(rel_rot_matrix[i]).as_quat()
        quats.append(quat)

    quats = np.array(quats)

    return translations, quats


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("data")
    parser.add_argument("--dataset", default="kitti")
    parser.add_argument("--model", default="transformer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="./eval_results")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = VIO_Dataset(
        root=args.data,
        dataset_type=args.dataset,
        sequence_length=3,
        train=False
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    if args.model == "transformer":
        model = VIO_TRANSFORMER().to(device)
    else:
        model = VIO_CNN_LSTM().to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("Running inference...")

    pred_rel, gt_rel = inference_one_epoch(loader, model, device)

    # ─────────────────────────────
    # Trajectory reconstruction
    # ─────────────────────────────

    pred_global = integrate_trajectory(pred_rel)
    gt_global = integrate_trajectory(gt_rel)

    # ─────────────────────────────
    # Metrics
    # ─────────────────────────────

    ate = compute_ate(pred_global, gt_global)
    rpe_t, rpe_r = compute_rpe(pred_global, gt_global)

    print(f"ATE RMSE: {ate:.4f} m")
    print(f"RPE trans: {rpe_t:.4f} m")
    print(f"RPE rot: {rpe_r:.4f} rad")

    if args.dataset == "kitti":

        t_err, r_err = compute_kitti_errors(pred_global, gt_global)

        print(f"KITTI translational error: {t_err:.3f}%")
        print(f"KITTI rotational error: {r_err:.4f} deg/m")

    # ─────────────────────────────
    # Save trajectories
    # ─────────────────────────────

    save_tum(out_dir / "est_poses.txt", pred_global)
    save_tum(out_dir / "gt_poses.txt", gt_global)

    # ─────────────────────────────
    # Plot trajectory
    # ─────────────────────────────

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_global[:,0], gt_global[:,1], gt_global[:,2], label="GT")
    ax.plot(pred_global[:,0], pred_global[:,1], pred_global[:,2], label="Pred")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.savefig(out_dir / "trajectory.png", dpi=300)
    plt.close()

    print("Results saved to", out_dir)


if __name__ == "__main__":
    main()

# Example – KITTI
#python evaluate.py /path/to/kitti/dataset  --dataset kitti --model transformer --checkpoint checkpoints/run_XX-XX-XX-XX/best_model.pth --output-dir eval_transformer_kitt