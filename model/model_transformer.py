import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================================
# SINUSOIDAL POSITIONAL ENCODING (essential for sequence order)
# =========================================================
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (works for any sequence length,
    no learned parameters needed). Used for vision patches (spatial order)
    and IMU tokens (temporal order).
    """
    def __init__(self, embed_dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) → adds positional encoding up to current length.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# =========================================================
# CNN STEM (better visual features)
# =========================================================
class CNNStem(nn.Module):
    """
    Lightweight CNN stem for dual-frame input (6 channels = two RGB images).
    Produces high-level feature maps suitable for patching.
    """
    def __init__(self, in_channels: int = 6, embed_dim: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =========================================================
# PATCH EMBEDDING FROM CNN FEATURE MAP
# =========================================================
class PatchEmbedding(nn.Module):
    """
    Converts CNN feature map into sequence of patch tokens (ViT-style projection).
    """
    def __init__(self, embed_dim: int = 256, patch_size: int = 4):
        super().__init__()

        self.proj = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)    # (B, N_patches, D)
        return x


# =========================================================
# IMU TEMPORAL ENCODER (with positional encoding)
# =========================================================
class IMUEncoder(nn.Module):
    """
    Projects raw IMU (accel + gyro) and encodes temporal dependencies
    using a shallow Transformer + sinusoidal positional encoding.
    """
    def __init__(self, embed_dim: int = 256, heads: int = 4, layers: int = 2):
        super().__init__()

        self.input_proj = nn.Linear(6, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN for training stability
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, imu_seq: torch.Tensor) -> torch.Tensor:
        """
        imu_seq: (B, T, 6)
        """
        tokens = self.input_proj(imu_seq)               # (B, T, D)
        tokens = tokens + self.pos_encoding(tokens)     # add temporal PE
        tokens = self.encoder(tokens)
        return tokens


# =========================================================
# BIDIRECTIONAL CROSS-MODAL FUSION BLOCK (paper-grade)
# =========================================================
class CrossModalFusion(nn.Module):
    """
    Full pre-norm cross-attention block with FFNs (inspired by modern
    multimodal Transformers). Includes LayerNorm, residual connections,
    GELU, and dropout for robust training.
    """
    def __init__(self, embed_dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Cross-attention
        self.v_to_i_attn = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )
        self.i_to_v_attn = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )

        # Pre-norm for attention
        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.norm_i1 = nn.LayerNorm(embed_dim)

        # Feed-forward networks
        self.ff_v = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ff_i = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Pre-norm for FFN
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.norm_i2 = nn.LayerNorm(embed_dim)

    def forward(self, vision: torch.Tensor, imu: torch.Tensor):
        """
        vision: (B, N_v, D)
        imu:    (B, T, D)
        """
        # ----- Cross-attention (pre-norm) -----
        v_norm = self.norm_v1(vision)
        i_norm = self.norm_i1(imu)

        # Vision attends to IMU
        vision_attn, _ = self.v_to_i_attn(v_norm, i_norm, i_norm)
        vision = vision + vision_attn

        # IMU attends to Vision (using original norms for parallelism)
        imu_attn, _ = self.i_to_v_attn(i_norm, v_norm, v_norm)
        imu = imu + imu_attn

        # ----- Feed-forward (pre-norm) -----
        vision = vision + self.ff_v(self.norm_v2(vision))
        imu = imu + self.ff_i(self.norm_i2(imu))

        return vision, imu


# =========================================================
# MAIN VIO TRANSFORMER (improved & paper-ready)
# =========================================================
class VIO_TRANSFORMER(nn.Module):
    """
    Visual-Inertial Odometry Transformer with early cross-modal fusion.
    Key improvements for robustness:
      • Sinusoidal positional encodings (vision + IMU)
      • Full pre-norm cross-modal fusion with FFNs
      • Pre-LN Transformer layers + GELU
      • Dedicated CLS positional embedding
      • Explicit dropout everywhere
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
    ):
        super().__init__()

        # ================== Vision branch ==================
        self.cnn = CNNStem()
        self.patch_embed = PatchEmbedding(embed_dim)
        self.vision_pos_encoding = PositionalEncoding(embed_dim)

        # ================== IMU branch ==================
        self.imu_encoder = IMUEncoder(embed_dim)

        # ================== Cross-modal fusion ==================
        self.fusion = CrossModalFusion(embed_dim, heads)

        # ================== Joint reasoning ==================
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # ================== Heads ==================
        self.pose_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

        # Uncertainty head (e.g., 6-DoF covariance or log-variance)
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 6),
        )

    def forward(self, img_pair: torch.Tensor, imu_seq: torch.Tensor):
        """
        img_pair: (B, 6, H, W)   # two stacked RGB frames
        imu_seq:  (B, T, 6)
        Returns:
            pose:       (B, 7)   # [tx, ty, tz, qx, qy, qz, qw] (quat normalized)
            uncertainty:(B, 6)
        """
        B = img_pair.size(0)

        # ----- Vision tokens -----
        feat = self.cnn(img_pair)
        vision_tokens = self.patch_embed(feat)
        vision_tokens = vision_tokens + self.vision_pos_encoding(vision_tokens)

        # ----- IMU tokens -----
        imu_tokens = self.imu_encoder(imu_seq)

        # ----- Cross-modal fusion -----
        vision_tokens, imu_tokens = self.fusion(vision_tokens, imu_tokens)

        # ----- Concatenate modalities -----
        tokens = torch.cat([vision_tokens, imu_tokens], dim=1)

        # ----- CLS token -----
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed
        tokens = torch.cat([cls, tokens], dim=1)

        # ----- Joint Transformer reasoning -----
        encoded = self.transformer(tokens)
        cls_out = encoded[:, 0]

        # ----- Pose head -----
        pose = self.pose_head(cls_out)
        t = pose[:, :3]
        q = F.normalize(pose[:, 3:], dim=1)
        pose = torch.cat([t, q], dim=1)

        return pose  # <<< ONLY return pose for train.py compatibility