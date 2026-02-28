import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Vision Backbone (Pretrained ResNet18)
# =========================================================

class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(VisionEncoder, self).__init__()

        resnet = models.resnet18(pretrained=True)

        # Modify first conv to accept 6 channels (target + reference)
        self.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Copy pretrained weights for first 3 channels
        self.conv1.weight.data[:, :3] = resnet.conv1.weight.data
        self.conv1.weight.data[:, 3:] = resnet.conv1.weight.data

        # Use pretrained layers
        self.backbone = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(512, output_dim)

    def forward(self, tgt_img, ref_img):
        x = torch.cat([tgt_img, ref_img], dim=1)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # [batch, 256]


# =========================================================
# IMU Encoder (1D CNN instead of LSTM)
# =========================================================

class ImuEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, imu):  # [batch, seq_len, 6]
        _, (h, _) = self.lstm(imu)
        return self.fc(h[-1])  # Last hidden state


# =========================================================
# Fusion Module
# =========================================================

class FusionModule(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, vision_feat, imu_feat):
        feature_dim = vision_feat.size(1)
        q = self.query(vision_feat).unsqueeze(1)
        k = self.key(imu_feat).unsqueeze(1)
        v = self.value(imu_feat).unsqueeze(1)
        attn = F.softmax(torch.bmm(q, k.transpose(1,2)) / (feature_dim ** 0.5), dim=-1)
        fused = torch.bmm(attn, v).squeeze(1) + vision_feat  # Residual
        return self.fc(fused)

# =========================================================
# Temporal Module (GRU)
# =========================================================
class TemporalModule(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x):  # [batch, seq_len, feature_dim]
        x = self.transformer(x)  # Seq-first for Transformer
        return self.fc(x.mean(dim=1))  # Global average instead of last


# =========================================================
# Pose Regressor (Translation + Quaternion)
# =========================================================

class PoseRegressor(nn.Module):
    def __init__(self, input_dim=256):
        super(PoseRegressor, self).__init__()

        self.translation = nn.Linear(input_dim, 3)
        self.rotation = nn.Linear(input_dim, 4)  # quaternion

    def forward(self, x):
        t = self.translation(x)
        q = self.rotation(x)

        # Normalize quaternion
        q = F.normalize(q, dim=1)

        pose = torch.cat([t, q], dim=1)  # [batch, 7]
        return pose