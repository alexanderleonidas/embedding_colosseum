import logging

import pennylane as qml
import torch
from pennylane import numpy as np
from rich import print

log = logging.getLogger(__name__)

q_device = qml.device("default.qubit")
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def _layer(layer_weights, num_qubits):
    for wire in range(num_qubits):
        qml.Rot(*layer_weights[wire], wires=wire)

    for wires in [(i, (i + 1) % num_qubits) for i in range(num_qubits)]:
        qml.CNOT(wires)


def _state_preparation(x):
    qml.BasisState(x, wires=[0, 1, 2, 3])


# Add torch and backprop to be able to run it on a GPU using cuda
# This also enables us to use the standard pytorch optimizers, and all the other stuff
@qml.qnode(q_device, interface="torch", diff_method="backprop")
def _circuit(weights, x, num_qubits):
    _state_preparation(x)

    for layer_weights in weights:
        _layer(layer_weights, num_qubits)

    return qml.expval(qml.PauliZ(0))


class VariationalClassifier:
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        torch.random.manual_seed(0)
        self.weights = torch.rand((num_layers, num_qubits, 3), requires_grad=True)
        self.bias = torch.tensor(0.0, requires_grad=True)
        self.loss_fn = torch.nn.MSELoss()

    def classify(self, x):
        return (
            _circuit(weights=self.weights, x=x, num_qubits=self.num_qubits) + self.bias
        )

    def cost(self, X, Y):
        predictions = torch.stack([self.classify(x) for x in X])
        return self.loss_fn(predictions, Y)
