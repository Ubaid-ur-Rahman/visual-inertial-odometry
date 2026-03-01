import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================================
# Positional Encoding
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =========================================================
# Vision Patch Embedding (ViT-style)
# =========================================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=6, embed_dim=256, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


# =========================================================
# IMU Token Embedding
# =========================================================
class IMUEmbedding(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(6, embed_dim)

    def forward(self, imu):
        return self.proj(imu)  # [B, T, D]


# =========================================================
# Unified Transformer VIO Model
# =========================================================
class VIOTransformer(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 depth=6,
                 n_heads=8,
                 patch_size=16):

        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=6,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.imu_embed = IMUEmbedding(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True,
            dropout=0.1
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, img_pair, imu_seq):
        """
        img_pair: [B, 6, H, W]
        imu_seq:  [B, T, 6]
        """

        B = img_pair.size(0)

        img_tokens = self.patch_embed(img_pair)
        imu_tokens = self.imu_embed(imu_seq)

        tokens = torch.cat([img_tokens, imu_tokens], dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = self.pos_encoding(tokens)

        encoded = self.transformer(tokens)

        cls_out = encoded[:, 0]

        pose = self.mlp_head(cls_out)

        t = pose[:, :3]
        q = F.normalize(pose[:, 3:], dim=1)

        return torch.cat([t, q], dim=1)