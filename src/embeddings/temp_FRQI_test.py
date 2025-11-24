import random

import numpy as np
import tensorflow as tf
import torch
from qiskit.quantum_info import Statevector

from old.FRQI import FRQI as FRQI_Qiskit
from src.dataset.DataManager import DataManager
from src.embeddings.FRQI_PennyLane import FRQI as FRQI_PL

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

NUM_SAMPLES = 10
SHOW_FIRST_K = 8

dm = DataManager(batch_size=NUM_SAMPLES, seed=0, dataset="mnist", pixel_size=2)
train_loader, _, test_loader = dm.get_loaders()

X_batch, y_batch = next(iter(train_loader))

print("Batch shape from loader:", X_batch.shape)
print("Pixel-size each:", X_batch[0].shape)

X_batch = X_batch.view(NUM_SAMPLES, -1).numpy()  # flattening after batching

num_pixels = X_batch.shape[1]
frqi_pl = FRQI_PL(num_pixels=num_pixels)
frqi_qiskit = FRQI_Qiskit()


def amplitude_fidelity(
    pl_state, q_state
):  # compares how much the amplitudes of the two embeddings differ
    pl_abs = np.sort(np.abs(pl_state))
    q_abs = np.sort(np.abs(q_state))

    diff = np.linalg.norm(pl_abs - q_abs)
    return 1 - diff / np.sqrt(2)


for idx in range(NUM_SAMPLES):
    pixels = X_batch[idx]

    print(f"\n=================== Sample {idx} ==================\n")

    angles_pl = frqi_pl.preprocess(pixels)
    state_pl = frqi_pl.encode(angles_pl)()

    pixels_q = (pixels * 255).astype(
        np.float64
    )  # Qiskit expects non-normalized pixel values 0-255
    angles_q = frqi_qiskit.preprocess(pixels_q)
    circ_q = frqi_qiskit._create_FRQI_circ(angles_q, measure_bool=False)
    state_q = Statevector.from_instruction(circ_q).data

    print("Pennylane Normalization:", np.sum(np.abs(state_pl) ** 2))
    print("Qiskit Normalization:", np.sum(np.abs(state_q) ** 2))

    print("\n--- Full Absolute Amplitudes ---")
    print("PennyLane:", np.round(np.abs(state_pl), 3))
    print("Qiskit:   ", np.round(np.abs(state_q), 3))

    fidelity = amplitude_fidelity(state_pl, state_q)

    print("\nAmplitude Fidelity (ordering not considered)")
    print("Value:", fidelity)
