import numpy as np
import glob
import os

def evaluate_epoch(epoch, seq=5):

    result = np.loadtxt(f"results/result_seq{seq}_{epoch}.csv", delimiter=",")
    truth_pose = np.loadtxt(f"results/truth_pose_seq{seq}_{epoch}.csv", delimiter=",")
    truth_euler = np.loadtxt(f"results/truth_euler_seq{seq}_{epoch}.csv", delimiter=",")

    # Translation RMSE
    trans_rmse = np.sqrt(np.mean((result[:, :3] - truth_pose) ** 2))

    # Rotation RMSE
    rot_rmse = np.sqrt(np.mean((result[:, 3:] - truth_euler) ** 2))

    print(f"Epoch {epoch} | Seq {seq}")
    print(f"Translation RMSE: {trans_rmse:.6f}")
    print(f"Rotation RMSE: {rot_rmse:.6f}")
    print("-" * 40)


if __name__ == "__main__":
    for epoch in range(10):
        evaluate_epoch(epoch, seq=5)