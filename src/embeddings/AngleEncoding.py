import pennylane as qml
import torch

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


class AngleEncodingEmbedding:
    def __init__(self, num_features=6, rotation="RY"):
        # Simple angle encoding embedding, each output feature from PCA mapped to a single qubit rotation, baseline for PQC style embeddings
        # num_features = dimensionality of the input feature vector (PCA pre-processing)
        # rotation = rotation gate to use ("RX"/"RY" or "RZ") - by default RY, RX would also make sense like in ZZ, RZ does not add much expressivity

        self.num_features = num_features
        self.num_qubits = num_features
        self.rotation = rotation

    def state_preparation(self, x: torch.Tensor):
        x = x.to(device)
        batch_mode = x.dim() == 2  # check for batching

        for i in range(self.num_qubits):
            angle = (
                x[:, i] if batch_mode else x[i]
            )  # workaround around batching splitting the angle between the batches

            if self.rotation == "RX":
                qml.RX(angle, wires=i)
            elif self.rotation == "RY":
                qml.RY(angle, wires=i)
            elif self.rotation == "RZ":
                qml.RZ(angle, wires=i)
            else:
                raise ValueError("Rotation must be 'RX', 'RY', or 'RZ'")
