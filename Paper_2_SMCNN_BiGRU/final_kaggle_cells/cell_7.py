# =============================================================================
# Cell 13: Classification Map Generation
# =============================================================================
# (Note: Functions from previous cells are already globally available in Kaggle)
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import gc

device = get_device()
datasets = ['IndianPines', 'PaviaU', 'Salinas', 'Botswana', 'KSC', 'WHU_Hi']

for dataset_name in datasets:
    print(f"\n{'='*60}")
    print(f"  Generating Classification Map for {dataset_name}")
    print(f"{'='*60}")
    
    set_seed(42)
    try:
        X, y = load_dataset(dataset_name)
    except Exception as e:
        print(f"Could not load {dataset_name}: {e}. Skipping...")
        continue
    
    num_classes = len(np.unique(y[y > 0]))
    h, w, b = X.shape
    
    pca = PCA(n_components=30, whiten=True)
    X_pca = pca.fit_transform(X.reshape(-1, b)).reshape(h, w, 30)
    X_denoised = apply_kalman_denoising(X_pca)
    X_final = apply_polar_wavelet_transform(X_denoised)
    final_bands = X_final.shape[2]
    
    # Train on 5% split
    X_tr, X_te, y_tr, y_te = create_disjoint_patches(X_final, y, window_size=11, train_ratio=0.05, seed=42)
    
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    test_ds  = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)
    
    print("  Training SMCNN-MLP for classification map...")
    model = SMCNN(num_classes, final_bands, window_size=11, modulation_type='mlp')
    metrics = train_and_evaluate(model, train_loader, test_loader, device, epochs=100, patience=20)
    print(f"  OA: {metrics['OA']:.2f}%")
    
    # Now predict on EVERY labeled pixel to build the full map
    window_size = 11
    margin = window_size // 2
    padded = np.zeros((h + 2*margin, w + 2*margin, final_bands), dtype=np.float32)
    padded[margin:margin+h, margin:margin+w, :] = X_final
    
    pred_map = np.zeros((h, w), dtype=np.int32)
    labeled_pixels = np.argwhere(y > 0)
    
    # Batch prediction for speed
    batch_patches = []
    batch_coords = []
    BATCH = 256
    
    model.eval()
    model.to(device)
    
    for idx, (r, c) in enumerate(labeled_pixels):
        patch = padded[r:r+window_size, c:c+window_size, :]
        batch_patches.append(patch)
        batch_coords.append((r, c))
        
        if len(batch_patches) == BATCH or idx == len(labeled_pixels) - 1:
            batch_tensor = torch.tensor(np.array(batch_patches, dtype=np.float32)).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)
                _, preds = torch.max(outputs, 1)
            preds_np = preds.cpu().numpy()
            
            for i, (br, bc) in enumerate(batch_coords):
                pred_map[br, bc] = preds_np[i] + 1  # +1 since classes are 1-indexed in GT
            
            batch_patches = []
            batch_coords = []
    
    # Plot Ground Truth vs Prediction side-by-side
    cmap = plt.cm.get_cmap('tab20', num_classes + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(y, cmap=cmap, vmin=0, vmax=num_classes)
    axes[0].set_title(f'{dataset_name} — Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(pred_map, cmap=cmap, vmin=0, vmax=num_classes)
    axes[1].set_title(f'{dataset_name} — SMCNN-MLP Prediction (OA={metrics["OA"]:.1f}%)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'classification_map_{dataset_name}.pdf', dpi=300)
    print(f"  Saved classification_map_{dataset_name}.pdf")
    plt.close()
    
    del model, metrics, padded, pred_map, batch_tensor, batch_patches, batch_coords
    del X, y, X_pca, X_denoised, X_final, X_tr, X_te, y_tr, y_te, train_ds, test_ds, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    