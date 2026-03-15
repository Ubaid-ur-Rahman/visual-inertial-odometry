import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================
# Residual CNN Encoder for visual input
# ===============================================
class ResidualVisualEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.res_block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.res_block3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )

        # 1x1 conv shortcuts
        self.shortcut1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.shortcut2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        self.shortcut3 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        r1 = F.relu(self.res_block1(x) + self.shortcut1(x))
        r2 = F.relu(self.res_block2(r1) + self.shortcut2(r1))
        r3 = F.relu(self.res_block3(r2) + self.shortcut3(r2))
        x = self.pool(r3).flatten(1)
        x = self.fc(x)
        return x

# ===============================================
# Temporal IMU Encoder
# ===============================================
class IMUTemporalEncoder(nn.Module):
    def __init__(self, imu_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(6, imu_dim)
        self.gru = nn.GRU(
            input_size=imu_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
    def forward(self, imu_seq):
        x = self.input_proj(imu_seq)
        out, _ = self.gru(x)
        return out[:, -1]  # last timestep representation

# ===============================================
# Dual-Path Fusion + LSTM
# ===============================================
class VIO_CNN_LSTM(nn.Module):
    def __init__(self, vis_dim=512, imu_dim=128, hidden_dim=256, lstm_layers=2):
        super().__init__()
        self.visual_encoder = ResidualVisualEncoder(vis_dim)
        self.imu_encoder = IMUTemporalEncoder(imu_dim, imu_dim, num_layers=2)

        # Fusion LSTM
        self.lstm = nn.LSTM(
            input_size=vis_dim + 2*imu_dim,  # 512 + 256 = 768
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers>1 else 0.0,
            bidirectional=True
        )

        # Pose regression head with residual MLP
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim*2, 256),  # bidirectional
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,7)  # t(3)+q(4)
        )

        # Optional uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim*2,6)
        )

    def forward(self, img_pair, imu_seq):
        B,T,_ = imu_seq.shape
        vis_feat = self.visual_encoder(img_pair)
        imu_feat = self.imu_encoder(imu_seq)

        fused = torch.cat([vis_feat, imu_feat], dim=-1).unsqueeze(1).repeat(1,T,1)

        lstm_out,_ = self.lstm(fused)
        last_hidden = lstm_out[:,-1,:]

        pose = self.pose_head(last_hidden)
        t = pose[:,:3]
        q = F.normalize(pose[:,3:], dim=-1)
        pose = torch.cat([t,q],dim=-1)

        return pose