# =============================================================================
# Cell 10: SOTA Baseline Comparison (2D-CNN, 3D-CNN, HybridSN) - 5 SEEDS × 6 DATASETS
# =============================================================================
# (Note: Functions from previous cells are already globally available in Kaggle)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import numpy as np
import gc

# --------------- Baseline Model Definitions ---------------

class Baseline2DCNN(nn.Module):
    """Standard 2D-CNN baseline for HSI classification."""
    def __init__(self, num_classes, num_bands, window_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class Baseline3DCNN(nn.Module):
    """3D-CNN baseline using 3D convolutions over the spectral-spatial cube."""
    def __init__(self, num_classes, num_bands, window_size):
        super().__init__()
        # Input: (B, 1, num_bands, H, W)
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, H, W, bands) -> (B, 1, bands, H, W)
        x = x.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class HybridSN(nn.Module):
    """HybridSN: 3D-CNN followed by 2D-CNN (Roy et al., 2020)."""
    def __init__(self, num_classes, num_bands, window_size):
        super().__init__()
        # 3D part: (B, 1, bands, H, W)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        
        # 2D part: flatten spectral from 3D output
        self.conv2d = nn.Sequential(
            nn.Conv2d(32 * num_bands, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, H, W, bands) -> (B, 1, bands, H, W)
        x = x.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        # Reshape: (B, 32, bands, H, W) -> (B, 32*bands, H, W)
        B, C, D, H, W = x.shape
        x = x.view(B, C * D, H, W)
        x = self.conv2d(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# --------------- Runner ---------------

device = get_device()
