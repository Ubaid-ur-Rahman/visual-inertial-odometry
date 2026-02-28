import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R
import random


class EuRoC_Loader(data.Dataset):

    def __init__(self, root, seed=None, train=0,
                 sequence_length=3, transform=None,
                 data_degradation=0, data_random=True):

        np.random.seed(seed)
        random.seed(seed)

        self.root = Path(root)
        self.scenes = [str(self.root)]
        self.sequence_length = sequence_length
        self.transform = transform
        self.random = data_random

        self.img_folder = self.root / "cam0/data"
        self.imu_file = self.root / "imu0/data.csv"
        self.gt_file = self.root / "state_groundtruth_estimate0/data.csv"

        self.load_data()
        self.build_sequences()

    def load_data(self):

        # Load image timestamps
        self.img_files = sorted(self.img_folder.glob("*.png"))
        self.img_timestamps = np.array(
            [int(f.stem) for f in self.img_files]
        )

        # Load IMU
        imu_data = pd.read_csv(self.imu_file, comment='#')
        self.imu_timestamps = imu_data.iloc[:, 0].values
        self.imu_values = imu_data.iloc[:, 1:7].values.astype(np.float32)

        # Load ground truth
        gt_data = pd.read_csv(self.gt_file, comment='#')
        self.gt_timestamps = gt_data.iloc[:, 0].values
        self.gt_pos = gt_data.iloc[:, 1:4].values
        self.gt_quat = gt_data.iloc[:, 4:8].values

    def build_sequences(self):

        self.samples = []
        demi = (self.sequence_length - 1) // 2

        for i in range(demi, len(self.img_files) - demi):

            sample = {"imgs": [], "imus": [], "poses": []}

            for j in range(-demi, demi + 1):

                img_idx = i + j
                img_ts = self.img_timestamps[img_idx]

                sample["imgs"].append(self.img_files[img_idx])

                # Find closest ground truth
                gt_idx = np.argmin(np.abs(self.gt_timestamps - img_ts))

                pose = self.create_pose_matrix(
                    self.gt_pos[gt_idx],
                    self.gt_quat[gt_idx]
                )

                sample["poses"].append(pose)

                # IMU segment until next frame
                if img_idx < len(self.img_files) - 1:

                    next_ts = self.img_timestamps[img_idx + 1]

                    mask = (self.imu_timestamps >= img_ts) & \
                           (self.imu_timestamps < next_ts)

                    imu_segment = self.imu_values[mask]

                    if len(imu_segment) == 0:
                        imu_segment = np.zeros((1, 6), dtype=np.float32)

                    # Average IMU over the frame interval
                    imu_avg = np.mean(imu_segment, axis=0, keepdims=True)

                    sample["imus"].append(imu_avg)

                else:
                    sample["imus"].append(
                        np.zeros((1, 6), dtype=np.float32)
                    )

            self.samples.append(sample)

        if self.random:
            random.shuffle(self.samples)

    def create_pose_matrix(self, position, quat):

        rot = R.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_matrix()

        pose = np.zeros((3, 4))
        pose[:, :3] = rot
        pose[:, 3] = position

        return pose.astype(np.float64)

    def __getitem__(self, index):

        sample = self.samples[index]

        imgs = [cv2.imread(str(p)).astype(np.float32)
                for p in sample["imgs"]]

        poses = sample["poses"]
        imus = sample["imus"]

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, imus, poses
    
    def euroc_collate(batch):
        # batch is list of (imgs_list, imus_list, poses_list)
        imgs = [item[0] for item in batch]          # list of list-of-tensors
        imus = [item[1] for item in batch]
        poses = [item[2] for item in batch]

        # Stack images: want [seq_len, batch, C, H, W]
        imgs_stacked = []
        for timestep in range(len(imgs[0])):
            timestep_imgs = torch.stack([seq[timestep] for seq in imgs], dim=0)  # [batch, C, H, W]
            imgs_stacked.append(timestep_imgs)

        # IMU: each is list of (1,6) arrays → convert to tensor
        imus_stacked = []
        for timestep in range(len(imus[0])):
            timestep_imus = torch.stack([torch.from_numpy(seq[timestep]) for seq in imus], dim=0)  # [batch, 1, 6]
            imus_stacked.append(timestep_imus)

        poses_stacked = []
        for timestep in range(len(poses[0])):
            timestep_pose = torch.stack(
                [torch.from_numpy(seq[timestep]) for seq in poses],
                dim=0
            )
            poses_stacked.append(timestep_pose)

        return imgs_stacked, imus_stacked, poses_stacked

    def __len__(self):
        return len(self.samples)
