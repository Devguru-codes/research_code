# =============================================================================
# Cell 12: Computational Efficiency Profiling
# =============================================================================
# (Note: Functions from previous cells are already globally available in Kaggle)
import torch
import torch.nn as nn
import numpy as np
import time

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_shape, device, num_runs=100):
    """Measure average inference time over num_runs forward passes."""
    model.eval()
    model.to(device)
    dummy = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_runs * 1000  # ms
    return elapsed

device = get_device()
num_classes = 13  # KSC as reference
num_bands = 60    # After PLCWT
window_size = 11
batch_size = 1    # Single sample for inference timing

input_shape = (batch_size, window_size, window_size, num_bands)

model_builders = {
    '2D-CNN':      lambda: Baseline2DCNN(num_classes, num_bands, window_size),
    '3D-CNN':      lambda: Baseline3DCNN(num_classes, num_bands, window_size),
    'HybridSN':    lambda: HybridSN(num_classes, num_bands, window_size),
    'SMCNN-MLP':   lambda: SMCNN(num_classes, num_bands, window_size, modulation_type='mlp'),
    'SMCNN-UniGRU':lambda: SMCNN(num_classes, num_bands, window_size, modulation_type='unigru'),
    'SMCNN-BiGRU': lambda: SMCNN(num_classes, num_bands, window_size, modulation_type='bigru'),
    'SMCNN-None':  lambda: SMCNN(num_classes, num_bands, window_size, modulation_type='none'),
}

print(f"\n{'='*70}")
print(f"  Computational Efficiency Profiling (KSC reference, 60 bands)")
print(f"{'='*70}")
print(f"{'Model':<16} {'Params':>10} {'Params (M)':>10} {'Infer (ms)':>11} {'GPU Mem (MB)':>12}")
print(f"{'-'*60}")

for name, builder in model_builders.items():
    model = builder()
    params = count_parameters(model)
    params_m = params / 1e6
    
    # Measure inference
    infer_ms = measure_inference_time(model, input_shape, device)
    
    # Measure GPU memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        model.to(device)
        dummy = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            _ = model(dummy)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6
    else:
        gpu_mem = 0
    
    print(f"{name:<16} {params:>10,d} {params_m:>10.3f} {infer_ms:>10.2f} {gpu_mem:>11.1f}")
    
    del model
    torch.cuda.empty_cache()
