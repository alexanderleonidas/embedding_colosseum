import random

import numpy as onp  # ordinary NumPy
import torch
from pennylane import numpy as np

from src.dataset.DataManager import DataManager
from src.embeddings.NEQR_PennyLane import NEQR

torch.manual_seed(0)
onp.random.seed(0)
random.seed(0)

dm = DataManager(batch_size=2, seed=0, dataset="mnist", pixel_size=8)
train_loader, _ = dm.get_loaders()
X_batch, y_batch = next(iter(train_loader))

print(f"Pixel tensor shape: {X_batch.shape}")

# Flatten a single image to 1D vector of length num_pixels
pixels = X_batch[0].view(-1).numpy()
print("Pixel range:", pixels.min(), pixels.max())
print("Num of pixels:", len(pixels))

# Verify flattening order by reshaping back to 2D
# (This is just for verification - the quantum encoding uses the flattened array)
if len(pixels) == 64:  # 8x8 image
    img_2d = pixels.reshape(8, 8)
    print("Reshaped to 2D - first row (first 8 pixels):", img_2d[0][:8])


neqr = NEQR(num_pixels=len(pixels), bit_depth=8)
intensities = neqr.preprocess(pixels)
print("NEQR intensities (first 10):", intensities[:10])

circuit = neqr.encode(intensities)
state = circuit()

print(f"NEQR num_qubits: {neqr.num_qubits}")
print(f"NEQR state length (2^num_qubits): {len(state)}")

# Show a few amplitudes just to verify it's not all zero
print("Sample of NEQR amplitudes (first 16):")
print(np.round(state[:16], 3))

# Check normalization
norm = np.sum(np.abs(state) ** 2)
print("Amplitude square sum (should be 1):", norm)

# img = X_batch[1].view(8,8).numpy()
# print(img)
