import pennylane as qml
import torch

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


class RMPEmbedding:
    def __init__(self, num_features=6, alpha=0.5, rotation="RY"):
        """
        Reverse Map Projection (RMP) embedding prototype - instead of mapping features directly to rotation angles (like ZZFeatMap and Angle Encoding
        in RMP we apply a nonlinear projection that acts a "geometric treansformation" into a new feature space. The general idea of this prototype
        is to see how nonlinear projections before quantum encoding affect the performance.

        RMP is normally described as structure-preserving, however in our pipeline it preserves/reshapes the structure of the PCA feature output geometry
        rather then preserving the spatial image structure like QIR methods, so it is more similar and comparable to PQC based methods
        """
        self.num_features = num_features
        self.num_qubits = num_features
        self.alpha = alpha
        self.rotation = rotation

    def _rmp_projection(self, x: torch.Tensor):
        # each feature is transformed via the projection function, defined as the fearture value to the power of the hyperparameter a
        # this projection with a parameter a<1 compresses larger values and amplifies smaller ones
        # applying the nonlinear scaling x^alpha taking into account the sign
        x_proj = torch.sign(x) * torch.abs(x) ** self.alpha

        if x_proj.dim() == 2:
            # normalization to the unit length, with regard to batching
            norm = torch.linalg.norm(x_proj, dim=1, keepdim=True) + 1e-8
        else:
            norm = torch.linalg.norm(x_proj) + 1e-8

        return x_proj / norm

    def state_preparation(self, x: torch.Tensor):
        x = x.to(device)
        batch_mode = x.dim() == 2

        x = self._rmp_projection(x)  # applying the projection function

        for i in range(self.num_qubits):  # and regular angle encoding follows
            angle = x[:, i] if batch_mode else x[i]

            if self.rotation == "RX":
                qml.RX(angle, wires=i)
            elif self.rotation == "RY":
                qml.RY(angle, wires=i)
            elif self.rotation == "RZ":
                qml.RZ(angle, wires=i)
            else:
                raise ValueError("Rotation must be 'RX', 'RY', or 'RZ'")
