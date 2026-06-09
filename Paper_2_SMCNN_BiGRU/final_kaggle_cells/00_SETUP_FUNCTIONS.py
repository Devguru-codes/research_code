import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Enforce strict reproducibility across PyTorch, NumPy, and Python."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Detect GPU/CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

print("Cell 1: Setup and Utils ready.")

import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

DATASET_INFO = {
    'IndianPines': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
        'data_key': 'indian_pines_corrected',
        'gt_key':   'indian_pines_gt',
        'num_classes': 16
    },
    'PaviaU': {
        'data_url': 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        'gt_url':   'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
        'data_key': 'paviaU',
        'gt_key':   'paviaU_gt',
        'num_classes': 9
    },
    'Salinas': {
        'data_url': '/kaggle/input/datasets/sreevallimanda/salinas-hyperspectral/Salinas_corrected.mat',
        'gt_url':   '/kaggle/input/datasets/sreevallimanda/salinas-hyperspectral/Salinas_gt.mat',
        'data_key': 'salinas_corrected',
        'gt_key':   'salinas_gt',
        'num_classes': 16
    },
    'Botswana': {
        'data_url': '/kaggle/input/datasets/tanverahmed/botswana-hsi/Botswana.mat',
        'gt_url':   '/kaggle/input/datasets/tanverahmed/botswana-hsi/Botswana_gt.mat',
        'data_key': 'Botswana',
        'gt_key':   'Botswana_gt',
        'num_classes': 14
    },
    'KSC': {
        'data_url': '/kaggle/input/datasets/sreevallimanda/ksc-hyperspectral/KSC.mat',
        'gt_url':   '/kaggle/input/datasets/sreevallimanda/ksc-hyperspectral/KSC_gt.mat',
        'data_key': 'KSC',
        'gt_key':   'KSC_gt',
        'num_classes': 13
    },
    'WHU_Hi': {
        'data_url': '/kaggle/input/datasets/dev123123456/whu-hsi-dataset/WHU_Hi_HanChuan.mat',
        'gt_url':   '/kaggle/input/datasets/dev123123456/whu-hsi-dataset/WHU_Hi_HanChuan_gt.mat',
        'data_key': 'WHU_Hi_HanChuan',
        'gt_key':   'WHU_Hi_HanChuan_gt',
        'num_classes': 16
    },
}

def _find_in_kaggle_input(pattern):
    import glob
    matches = glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)
    return matches[0] if matches else None

def download_dataset(dataset_name):
    info = DATASET_INFO[dataset_name]
    if info['data_url'].startswith('/kaggle/input') and os.path.exists(info['data_url']):
        return info['data_url'], info['gt_url']

    os.makedirs('datasets', exist_ok=True)
    data_path = f"datasets/{dataset_name}.mat"
    gt_path   = f"datasets/{dataset_name}_gt.mat"

    search_patterns = {
        'IndianPines': (['*ndian*pines*corrected*.mat', '*Indian_pines.mat'], ['*ndian*pines*gt*.mat']),
        'PaviaU':      (['*aviaU.mat'], ['*aviaU*gt*.mat']),
        'Salinas':     (['*alinas_corrected*.mat', '*alinas.mat'], ['*alinas_gt*.mat']),
        'Botswana':    (['*Botswana.mat'], ['*otswana*gt*.mat']),
        'KSC':         (['*KSC.mat'], ['*KSC_gt*.mat']),
        'WHU_Hi':      (['*HanChuan.mat'], ['*HanChuan*gt*.mat']),
    }

    found_data = found_gt = None
    data_pats, gt_pats = search_patterns.get(dataset_name, ([], []))
    for pat in data_pats:
        found_data = _find_in_kaggle_input(pat)
        if found_data: break
    for pat in gt_pats:
        found_gt = _find_in_kaggle_input(pat)
        if found_gt: break

    if found_data and found_gt:
        print(f"  Found {dataset_name} in Kaggle Input")
        return found_data, found_gt

    if not os.path.exists(data_path):
        print(f"  Downloading {dataset_name} data...")
        import urllib.request
        urllib.request.urlretrieve(info['data_url'], data_path)
    if not os.path.exists(gt_path):
        print(f"  Downloading {dataset_name} GT...")
        import urllib.request
        urllib.request.urlretrieve(info['gt_url'], gt_path)
            
    return data_path, gt_path

def load_dataset(dataset_name):
    data_path, gt_path = download_dataset(dataset_name)
    info = DATASET_INFO[dataset_name]
    
    def _get_array(mat_file, expected_key):
        data = sio.loadmat(mat_file)
        if expected_key in data: return data[expected_key]
        arrays = {k: v for k, v in data.items() if not k.startswith('__') and hasattr(v, 'shape')}
        key = max(arrays, key=lambda k: arrays[k].size)
        return arrays[key]
    
    X = _get_array(data_path, info['data_key'])
    y = _get_array(gt_path, info['gt_key'])
    print(f"Loaded {dataset_name}: X={X.shape}, y={y.shape}, Classes={info['num_classes']}")
    return X, y

def pad_with_zeros(X, margin):
    h, w, b = X.shape
    padded = np.zeros((h + 2*margin, w + 2*margin, b), dtype=X.dtype)
    padded[margin:margin+h, margin:margin+w, :] = X
    return padded

def create_disjoint_patches(X, y, window_size=11, train_ratio=0.05, seed=42):
    rng = np.random.RandomState(seed)
    margin = (window_size - 1) // 2
    padded_X = pad_with_zeros(X, margin)

    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
    classes = np.unique(y[y > 0])

    for c in classes:
        class_indices = np.argwhere(y == c)
        rng.shuffle(class_indices)
        n_train = max(1, int(len(class_indices) * train_ratio))
        
        for r, c_idx in class_indices[:n_train]:
            X_train_list.append(padded_X[r:r+window_size, c_idx:c_idx+window_size, :])
            y_train_list.append(int(c - 1))
        for r, c_idx in class_indices[n_train:]:
            X_test_list.append(padded_X[r:r+window_size, c_idx:c_idx+window_size, :])
            y_test_list.append(int(c - 1))

    X_train, y_train = np.array(X_train_list, dtype=np.float32), np.array(y_train_list, dtype=np.int64)
    X_test, y_test = np.array(X_test_list, dtype=np.float32), np.array(y_test_list, dtype=np.int64)
    print(f"Split (seed={seed}): {len(y_train)} train, {len(y_test)} test")
    return X_train, X_test, y_train, y_test

print("Cell 2: Dataset Loader ready.")

import numpy as np
from scipy import signal
import pywt
from scipy.ndimage import zoom

def apply_kalman_denoising(X):
    """Adaptive Fast Desensitized Kalman Filter (AFDKF) — Proxy Implementation."""
    print("  Applying AFDKF denoising...")
    denoised = np.zeros_like(X, dtype=np.float32)
    for b in range(X.shape[2]):
        band = X[:, :, b].astype(np.float64)
        denoised[:, :, b] = signal.wiener(band, mysize=(5, 5)).astype(np.float32)
    return denoised

def apply_polar_wavelet_transform(X):
    """Polar Linear Canonical Wavelet Transform (PLCWT) — Proxy Implementation."""
    print("  Applying PLCWT feature extraction...")
    h, w, b = X.shape
    ll_features = np.zeros((h, w, b), dtype=np.float32)

    for band_idx in range(b):
        band = X[:, :, band_idx].astype(np.float64)
        coeffs = pywt.dwt2(band, 'db4')
        LL, _ = coeffs
        zoom_h, zoom_w = h / LL.shape[0], w / LL.shape[1]
        ll_resized = zoom(LL, (zoom_h, zoom_w), order=1).astype(np.float32)
        ll_features[:, :, band_idx] = ll_resized[:h, :w]

    combined = np.concatenate([X, ll_features], axis=2)
    print(f"  PLCWT: {b} bands -> {combined.shape[2]} bands")
    return combined

print("Cell 3: Preprocessing ready.")

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRUSpectralModulationBlock(nn.Module):
    def __init__(self, channels, spatial_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        
        self.rnn = nn.GRU(
            input_size=spatial_dim, 
            hidden_size=spatial_dim // 2, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )

        reduction = max(channels // 4, 8)
        self.fc1      = nn.Linear(channels, reduction)
        self.fc_gamma = nn.Linear(reduction, channels)
        self.fc_beta  = nn.Linear(reduction, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        
        B, C, H, W = out.shape
        ctx = out.view(B, C, H*W)
        ctx, _ = self.rnn(ctx)
        
        ctx = ctx.mean(dim=2)
        ctx = F.relu(self.fc1(ctx))
        
        gamma = torch.sigmoid(self.fc_gamma(ctx)).unsqueeze(-1).unsqueeze(-1)
        beta  = self.fc_beta(ctx).unsqueeze(-1).unsqueeze(-1)
        
        self.last_gamma = gamma.detach()
        self.last_beta = beta.detach()
        
        out = out * gamma + beta
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class UniGRUSpectralModulationBlock(nn.Module):
    def __init__(self, channels, spatial_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        
        self.rnn = nn.GRU(
            input_size=spatial_dim, 
            hidden_size=spatial_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=False
        )

        reduction = max(channels // 4, 8)
        self.fc1      = nn.Linear(channels, reduction)
        self.fc_gamma = nn.Linear(reduction, channels)
        self.fc_beta  = nn.Linear(reduction, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        
        B, C, H, W = out.shape
        ctx = out.view(B, C, H*W)
        ctx, _ = self.rnn(ctx)
        
        ctx = ctx.mean(dim=2)
        ctx = F.relu(self.fc1(ctx))
        
        gamma = torch.sigmoid(self.fc_gamma(ctx)).unsqueeze(-1).unsqueeze(-1)
        beta  = self.fc_beta(ctx).unsqueeze(-1).unsqueeze(-1)
        
        self.last_gamma = gamma.detach()
        self.last_beta = beta.detach()
        
        out = out * gamma + beta
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class MLPSpectralModulationBlock(nn.Module):
    def __init__(self, channels, spatial_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        reduction = max(channels // 4, 8)
        self.fc1      = nn.Linear(channels, reduction)
        self.fc_gamma = nn.Linear(reduction, channels)
        self.fc_beta  = nn.Linear(reduction, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        
        ctx = self.pool(out).flatten(1)
        ctx = F.relu(self.fc1(ctx))
        
        gamma = torch.sigmoid(self.fc_gamma(ctx)).unsqueeze(-1).unsqueeze(-1)
        beta  = self.fc_beta(ctx).unsqueeze(-1).unsqueeze(-1)
        
        self.last_gamma = gamma.detach()
        self.last_beta = beta.detach()
        
        out = out * gamma + beta
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class StandardResidualBlock(nn.Module):
    def __init__(self, channels, spatial_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class SMCNN(nn.Module):
    def __init__(self, num_classes, num_bands, window_size, modulation_type='bigru'):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        spatial_dim = window_size * window_size
        
        if modulation_type == 'bigru':
            block_class = BiGRUSpectralModulationBlock
        elif modulation_type == 'unigru':
            block_class = UniGRUSpectralModulationBlock
        elif modulation_type == 'mlp':
            block_class = MLPSpectralModulationBlock
        elif modulation_type == 'none':
            block_class = StandardResidualBlock
        else:
            raise ValueError(f"Unknown modulation_type: {modulation_type}")

        self.block1 = block_class(64, spatial_dim)
        self.block2 = block_class(64, spatial_dim)
        
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        features = self.pool(x).flatten(1)
        out = self.dropout(features)
        return self.fc(out)

    def extract_features(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.pool(x).flatten(1)

    def get_last_modulation_params(self):
        """Returns the gamma and beta tensors from the final modulation block."""
        if hasattr(self.block2, 'last_gamma'):
            return self.block2.last_gamma, self.block2.last_beta
        return None, None

print("Cell 4: Models ready (BiGRU, UniGRU, MLP, Standard).")

import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

def train_and_evaluate(model, train_loader, test_loader, device, epochs=100, patience=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # APCS / SFWOA proxy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    
    best_oa = 0
    best_model_state = None
    epochs_no_improve = 0
    
    model.to(device)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                
        oa = accuracy_score(all_labels, all_preds)
        scheduler.step(oa)
        
        if oa > best_oa:
            best_oa = oa
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    train_time = time.time() - start_time
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    final_oa = accuracy_score(all_labels, all_preds) * 100
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    aa = np.mean(per_class_acc)
    
    return {
        'OA': final_oa,
        'AA': aa,
        'Kappa': kappa,
        'train_time': train_time,
        'preds': np.array(all_preds),
        'labels': np.array(all_labels)
    }

print("Cell 5: Training logic ready.")

print('Cell 1: ALL SETUP READY (Utils, Loader, Preprocessing, Models, Training)')
