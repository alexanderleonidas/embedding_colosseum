import random
import numpy as onp
import torch
from pennylane import numpy as np

from src.dataset.DataManager import DataManager
from src.embeddings.OQIM_PennyLane import OQIM

torch.manual_seed(0)
onp.random.seed(0)
random.seed(0)

dm = DataManager(batch_size=2, seed=0, dataset="mnist", pixel_size=8)
train_loader, val_loader, test_loader = dm.get_loaders(train_split=0.8, val_split=0.1, test_split=0.1)
X_batch, y_batch = next(iter(train_loader))

pixels = X_batch[0].view(-1).numpy()
print("Pixel range:", pixels.min(), pixels.max())
print("Num pixels:", len(pixels))

oqim = OQIM(num_pixels=len(pixels))
angles_color, angles_coord = oqim.preprocess(pixels)

circuit = oqim.encode(angles_color, angles_coord)
state = circuit()

print(f"OQIM num_qubits: {oqim.num_qubits}")
print(f"State length: {len(state)}")

print("First 16 amplitudes:", np.round(state[:16], 3))
print("Norm check:", np.sum(np.abs(state) ** 2))
