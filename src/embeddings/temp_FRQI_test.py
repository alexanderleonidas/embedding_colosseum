import torch
import random
import numpy as np
from src.dataset.DataManager import DataManager
from src.embeddings.FRQI_PennyLane import FRQI
from src.model.VariationalClassifier import VariationalClassifier

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

dm = DataManager(batch_size=2, seed=0, dataset="mnist", pixel_size=4)
train_loader, _ = dm.get_loaders()
X_batch, y_batch = next(iter(train_loader))
print(f"Pixel tensor shape: {X_batch.shape}")

#img = X_batch[0].view(4,4).numpy()
#print("Image:\n", np.round(img, 2))

# FRQI expects a flattened vector, the DataManager flattens the data in resize_images but they are wrapped again during loading and batching
pixels = X_batch[0].view(-1).numpy()             
print("Pixel range:", pixels.min(), pixels.max())
print("Num of pixels:", len(pixels))

# embedding
frqi = FRQI(num_pixels=len(pixels))
angles = frqi.preprocess(pixels)
circuit = frqi.encode(angles)
state = circuit()

print(f"FRQI state length - should be 2^(log2(num_pixels)+1): {len(state)}")
print("FRQI final amplitudes:\n", np.round(state, 3))

print("Amplitude square sum:", np.sum(np.abs(state)**2))  # check whether the statize is normalized, amplitude square and sum to 1
