import pennylane as qml
import torch
from pennylane import numpy as np


device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


class ZZFeatureMapEmbedding:
    def __init__(self, num_features=6):
        # PQCs/Feature maps require preprocessing in the form of dimensionlaity reduction, as they dont have a clever approahc like QIR, but pass
        # thorugh and encode all the pixels, so either encoding 1 pixel per qubit, or one rotation per pixel on a few qubits, in any way the circuits
        # will be deep and likely not feasible to run
        # So this PQC is designed to take a reduced vector of featurs after preprocessing (likely tied to PCA always), and maps each feature to qubit
        # num_features = dimensionality of the classical feature vector (PCA output dimension)
        
        self.num_features = num_features
        self.num_qubits = num_features
        self.device = qml.device("lightning.qubit", wires=self.num_qubits)

    def state_preparation(self, x: torch.Tensor):
        # applies a fixed ZZ Feature Map embedding, consists of single-qubit RX rotations and ZZ entangling rotations for pairwise interactions

        x = x.to(device)

        batch_mode = x.dim() == 2    # check wheter processing batch or a single sample

        # encoding layer with RX(x_i) rotation
        for i in range(self.num_qubits):
            angle = x[:, i] if batch_mode else x[i]      # if batched gives one angle per sample, othewrise just a single angle
            qml.RX(angle, wires=i)

        # ZZ Entanglement Layer
        for i in range(self.num_qubits - 1):
            angle = (
                x[:, i] * x[:, i + 1]
                if batch_mode
                else x[i] * x[i + 1]
            )
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(angle, wires=i + 1)
            qml.CNOT(wires=[i, i + 1])

        pass