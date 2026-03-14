import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================================
# CNN Visual Encoder (FlowNet style)
# ===============================================
class VisualEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(6, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)

        return x.view(x.size(0), -1)  # [B,512]


# ===============================================
# IMU Encoder
# ===============================================
class IMUEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU()
        )

    def forward(self, imu):

        B,T,_ = imu.shape

        imu = imu.view(B*T,6)
        imu = self.net(imu)

        imu = imu.view(B,T,-1)

        return imu
        

# ===============================================
# CNN + LSTM Visual Inertial Odometry
# ===============================================
class VIO_CNN_LSTM(nn.Module):

    def __init__(self, hidden_dim=256):

        super().__init__()

        self.visual_encoder = VisualEncoder()

        self.imu_encoder = IMUEncoder()

        self.lstm = nn.LSTM(
            input_size=512+128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim,128),
            nn.ReLU(),
            nn.Linear(128,7)
        )

    def forward(self, img_pair, imu_seq):

        """
        img_pair : [B,6,H,W]
        imu_seq  : [B,T,6]
        """

        B = img_pair.size(0)

        # Visual features
        vis_feat = self.visual_encoder(img_pair)  # [B,512]

        # Expand for sequence
        T = imu_seq.size(1)
        vis_feat = vis_feat.unsqueeze(1).repeat(1,T,1)

        # IMU features
        imu_feat = self.imu_encoder(imu_seq)  # [B,T,128]

        # Fuse
        fused = torch.cat([vis_feat, imu_feat], dim=2)

        # LSTM
        out,_ = self.lstm(fused)

        last = out[:,-1]

        pose = self.head(last)

        t = pose[:,:3]
        q = F.normalize(pose[:,3:], dim=1)

        return torch.cat([t,q], dim=1)