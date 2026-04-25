import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pandas as pd
import cv2
import random
from scipy.spatial.transform import Rotation


class UniversalVIODataset(Dataset):

    def __init__(
        self,
        root,
        dataset_type="euroc",
        sequence_length=3,
        train=True,
        train_split=0.8,
        transform=None,
        seed=0
    ):

        random.seed(seed)
        np.random.seed(seed)

        self.root = Path(root)
        self.dataset_type = dataset_type.lower()
        self.sequence_length = sequence_length
        self.transform = transform

        if self.dataset_type == "euroc":
            self.samples = self.load_euroc()

        elif self.dataset_type == "kitti":
            self.samples = self.load_kitti()
        
        elif self.dataset_type == "tum":
            self.samples = self.load_tum()
        else:
            raise ValueError("Dataset not supported")

        # shuffle
        if train:
            random.shuffle(self.samples)

        # train val split
        split = int(len(self.samples) * train_split)

        if train:
            self.samples = self.samples[:split]
        else:
            self.samples = self.samples[split:]

        print(f"{len(self.samples)} samples loaded")

    # --------------------------------------------------
    # EUROC
    # --------------------------------------------------

    def load_euroc(self):

        samples = []
        demi = (self.sequence_length - 1) // 2

        sequences = sorted(self.root.glob("*"))

        for seq in sequences:

            mav = seq / "mav0"

            img_dir = mav / "cam0/data"
            imu_file = mav / "imu0/data.csv"
            gt_file = mav / "state_groundtruth_estimate0/data.csv"

            if not img_dir.exists():
                continue

            imgs = sorted(img_dir.glob("*.png"))
            img_ts = np.array([int(p.stem) for p in imgs])

            imu_data = pd.read_csv(imu_file, comment="#")
            imu_ts = imu_data.iloc[:,0].values
            imu_val = imu_data.iloc[:,1:7].values.astype(np.float32)

            gt = pd.read_csv(gt_file, comment="#")
            gt_ts = gt.iloc[:,0].values
            gt_pos = gt.iloc[:,1:4].values
            gt_quat = gt.iloc[:,4:8].values

            for i in range(demi, len(imgs) - demi):

                sample = {"imgs":[], "imus":[], "poses":[]}

                for j in range(-demi, demi+1):

                    idx = i+j
                    ts = img_ts[idx]

                    sample["imgs"].append(imgs[idx])

                    # pose
                    gt_idx = np.argmin(np.abs(gt_ts-ts))
                    pose = self.pose_matrix(gt_pos[gt_idx], gt_quat[gt_idx])
                    sample["poses"].append(pose)

                    if idx < len(imgs)-1:

                        next_ts = img_ts[idx+1]

                        mask = (imu_ts>=ts) & (imu_ts<next_ts)
                        imu_seq = imu_val[mask]

                        if len(imu_seq)==0:
                            imu_seq=np.zeros((1,6),dtype=np.float32)

                        imu_avg=np.mean(imu_seq,axis=0)
                        sample["imus"].append(imu_avg)

                    else:

                        sample["imus"].append(
                            np.zeros(6,dtype=np.float32)
                        )

                samples.append(sample)

        return samples

    # --------------------------------------------------
    # KITTI
    # --------------------------------------------------

    def load_kitti(self):

        samples = []
        demi = (self.sequence_length-1)//2

        seq_root = self.root/"sequences"
        pose_root = self.root/"poses"

        sequences = sorted(seq_root.glob("*"))

        for seq in sequences:

            img_dir = seq/"image_0"  
            imgs = sorted(img_dir.glob("*.png"))

            if len(imgs) == 0:
                continue

            pose_file = pose_root/(seq.name+".txt")

            if not pose_file.exists():
                continue

            poses = np.loadtxt(pose_file)
            poses = poses.reshape(-1,3,4)

            for i in range(demi,len(imgs)-demi):

                sample={"imgs":[],"imus":[],"poses":[]}

                for j in range(-demi,demi+1):

                    idx=i+j

                    sample["imgs"].append(imgs[idx])
                    sample["poses"].append(poses[idx])

                    # KITTI odometry has no IMU
                    imu = np.zeros(6,dtype=np.float32)
                    sample["imus"].append(imu)

                samples.append(sample)

        return samples
    

    # --------------------------------------------------
    # TUM VI
    # --------------------------------------------------
    def load_tum(self):

        samples = []
        demi = (self.sequence_length - 1) // 2

        cam0_dir = self.root / "cam0/data"
        cam1_dir = self.root / "cam1/data"

        ts0_file = self.root / "cam0/timestamps.txt"
        ts1_file = self.root / "cam1/timestamps.txt"

        imu_file = self.root / "imu0/data.csv"

        if not cam0_dir.exists():
            raise RuntimeError("cam0 folder not found")

        imgs0 = sorted(cam0_dir.glob("*.png"))
        imgs1 = sorted(cam1_dir.glob("*.png"))

        ts0 = np.loadtxt(ts0_file)
        ts1 = np.loadtxt(ts1_file)

        imu_data = pd.read_csv(imu_file)

        imu_ts = imu_data.iloc[:,0].values
        imu_val = imu_data.iloc[:,1:7].values.astype(np.float32)

        n = min(len(imgs0), len(imgs1))

        for i in range(demi, n - demi):

            sample = {"imgs":[], "imus":[], "poses":[]}

            for j in range(-demi, demi+1):

                idx = i + j
                ts = ts0[idx]

                sample["imgs"].append(imgs0[idx])

                # TUM VI has no ground truth pose
                pose = np.eye(3,4,dtype=np.float32)
                sample["poses"].append(pose)

                if idx < n - 1:

                    next_ts = ts0[idx+1]

                    mask = (imu_ts >= ts) & (imu_ts < next_ts)
                    imu_seq = imu_val[mask]

                    if len(imu_seq) == 0:
                        imu_seq = np.zeros((1,6),dtype=np.float32)

                    imu_avg = np.mean(imu_seq, axis=0)
                    sample["imus"].append(imu_avg)

                else:

                    sample["imus"].append(
                        np.zeros(6,dtype=np.float32)
                    )

            samples.append(sample)

        return samples

    # --------------------------------------------------
    # POSE
    # --------------------------------------------------

    def pose_matrix(self,pos,quat):

        rot=Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]]).as_matrix()

        pose=np.zeros((3,4))
        pose[:,:3]=rot
        pose[:,3]=pos

        return pose.astype(np.float32)

    # --------------------------------------------------
    # GET ITEM
    # --------------------------------------------------

    def __getitem__(self,index):

        s=self.samples[index]

        imgs=[]
        for p in s["imgs"]:

            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img,(640,192)) 

            img = img.astype(np.float32)/255.0
            img = torch.from_numpy(img).permute(2,0,1)

            imgs.append(img)

        imgs=torch.stack(imgs)

        imus=torch.from_numpy(np.array(s["imus"],dtype=np.float32))
        poses=torch.from_numpy(np.array(s["poses"],dtype=np.float32))

        return imgs,imus,poses

    def __len__(self):
        return len(self.samples)