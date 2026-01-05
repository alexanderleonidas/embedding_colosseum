import logging

import matplotlib as mpl
import pennylane as qml
import torch

log = logging.getLogger(__name__)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


class VariationalClassifier:
    def __init__(
        self,
        state_preparation,
        num_qubits: int,
        num_layers: int,
        num_classes: int,
        num_pixels: int = 32,
    ):
        assert state_preparation is not None, (
            "State preparation function must be provided"
        )
        self.state_preparation = state_preparation
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_pixels = num_pixels
        self.device = device

        # Calculate correct num qubits based on embedding qubits and num classes
        self.num_qubits = max(self.num_classes, num_qubits)

        # Initialize weights and bias
        torch.random.manual_seed(0)
        self.weights = torch.rand(
            (num_layers, self.num_qubits, 3), requires_grad=True, device=device
        )
        self.bias = torch.tensor(0.0, requires_grad=True, device=device)

        # If linux based GPU setup is available, install lightning.gpu
        #  (pip install custatevec_cu12 pennylane-lightning-gpu)
        # Otherwise, use the accelerated CPU version: lightning.qubit
        q_device = qml.device("lightning.gpu")

        # Use Torch as interface for PennyLane
        #   -> Gives compatibility with Torch optimizers and autograd (under circumstances)
        #
        # Note: If diff_method="adjoint", is not done by pytorch but by PennyLane Accelerators
        #   -> Requires compatible device (e.g. lightning.gpu or lightning.qubit)
        @qml.qnode(q_device, interface="torch", diff_method="adjoint")
        def circuit(weights, x):
            self.state_preparation(x)

            # Create variational layers based on weights shape
            qml.StronglyEntanglingLayers(weights=weights, wires=range(self.num_qubits))

            return tuple(qml.expval(qml.PauliZ(i)) for i in range(self.num_classes))

        self.circuit = circuit
        pass

    def classify(self, X):
        """Classifies a batch of inputs X"""
        return torch.stack(
            [
                torch.stack(self.circuit(weights=self.weights, x=x)) + self.bias
                for x in X
            ]
        )

    def save_svg(self, path: str = "circuit.svg", decimals: int = 2, level="top"):
        """Prints the circuit to a svg using qml.draw_mpl

        Args:
            path (str, optional): Path of the svg file. Defaults to "circuit.svg".
            decimals (int, optional): Number of decimals to display for rotations. Defaults to 2.
            level (str, optional): "top" shows the circuit as defined using pennylane. To show all operations choose "device". Defaults to "top". See https://docs.pennylane.ai/en/stable/code/api/pennylane.workflow.get_transform_program.html for more details.
        """
        mpl.rcParams["svg.fonttype"] = "none"  # To keep text as text in SVG
        qml.drawer.use_style("sketch")

        fig, ax = qml.draw_mpl(self.circuit, decimals=decimals, level=level)(
            self.weights,
            torch.tensor(
                [0 for _ in range(self.num_pixels)],
                dtype=torch.float64,
            ).to(device),
        )

        fig.savefig(path, transparent=True, bbox_inches="tight")
