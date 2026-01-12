import math

import pennylane as qml
import torch
from pennylane import numpy as np

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


class OQIM:
    """
    OQIM-style embedding (Order-encoded Quantum Image Model).

    High-level design (faithful to the paper’s description):
      - Use a basis-state register to index the ascending order (rank) of pixels.
      - Use amplitude probabilities of dedicated qubits to store:
          (a) "color" information (pixel intensity of that rank),
          (b) "coordinate/order-position" information (a scalar derived from the rank).

    Reference: Xu et al., "Order-encoded quantum image model and parallel histogram specification" (2019). (Paywalled full text)
    """

    def __init__(self, num_pixels: int):
        self.num_pixels = num_pixels
        self.num_rank_qubits = int(math.log2(num_pixels))
        assert 2**self.num_rank_qubits == num_pixels, "num_pixels must be a power of 2"

        # 2 extra qubits:
        #   wire 0: color qubit (stores intensity via amplitude prob)
        #   wire 1: coord/order-position qubit (stores rank-derived scalar via amplitude prob)
        #   wires 2.. : rank qubits store |rank> in computational basis
        self.num_qubits = 2 + self.num_rank_qubits
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

    def preprocess(self, pixels):
        """
        Input: pixels as array-like, either normalized [0,1] or [0,255].
        Output:
          - angles_color[k] in [0, pi/2] for the k-th smallest pixel
          - angles_coord[k] in [0, pi/2] encoding rank position k/(N-1)
        """
        pixels = np.array(pixels, dtype=float).flatten()

        # normalize if needed
        if pixels.max() > 1.5:
            pixels = np.clip(pixels, 0, 255) / 255.0

        # sort pixels ascending (order-encoded core idea)
        sorted_pixels = np.sort(pixels)  # length N

        # map intensity -> angle in [0, pi/2]
        angles_color = np.interp(sorted_pixels, (0.0, 1.0), (0.0, np.pi / 2))

        # encode rank position (0..N-1) as a scalar into a qubit’s amplitude prob
        if self.num_pixels > 1:
            ranks = np.arange(self.num_pixels) / (self.num_pixels - 1)
        else:
            ranks = np.array([0.0])

        angles_coord = np.interp(ranks, (0.0, 1.0), (0.0, np.pi / 2))

        return angles_color, angles_coord

    def encode(self, angles_color, angles_coord):
        """
        Returns a QNode that prepares the OQIM state and outputs the full statevector.
        """
        angles_color = np.array(angles_color, dtype=float)
        angles_coord = np.array(angles_coord, dtype=float)
        assert len(angles_color) == self.num_pixels
        assert len(angles_coord) == self.num_pixels

        color_wire = 0
        coord_wire = 1
        rank_wires = list(range(2, self.num_qubits))

        @qml.qnode(self.dev)
        def circuit():
            # uniform superposition over rank basis states
            for w in rank_wires:
                qml.Hadamard(wires=w)

            # for each rank k, rotate color/coord qubits controlled on |rank=k>
            for k in range(self.num_pixels):
                bits = np.binary_repr(k, width=self.num_rank_qubits)

                # flip controls where bit==0 so control activates on all-ones
                for wire, bit in zip(rank_wires, bits):
                    if bit == "0":
                        qml.PauliX(wires=wire)

                qml.ctrl(qml.RY, control=rank_wires)(
                    2 * angles_color[k], wires=color_wire
                )
                qml.ctrl(qml.RY, control=rank_wires)(
                    2 * angles_coord[k], wires=coord_wire
                )

                # unflip
                for wire, bit in zip(rank_wires, bits):
                    if bit == "0":
                        qml.PauliX(wires=wire)

            return qml.state()

        return circuit

    def state_preparation(self, X: torch.Tensor):
        """
        Same preparation as encode(), but callable inside a larger QNode (e.g., your VQC).
        Expects X as torch tensor, flattened image in [0,1] (your DataManager style).
        """
        pixels = X.flatten().to(device)

        # we assume your DataManager already normalizes to [0,1]
        # but clamp for safety
        pixels = torch.clamp(pixels, 0.0, 1.0)

        # order-encoding: sort intensities
        sorted_pixels, _ = torch.sort(pixels)

        # intensity -> angle
        pi2 = torch.tensor(
            math.pi / 2, dtype=sorted_pixels.dtype, device=sorted_pixels.device
        )
        angles_color = sorted_pixels * pi2

        # rank position -> angle
        if self.num_pixels > 1:
            ranks = torch.linspace(
                0.0,
                1.0,
                steps=self.num_pixels,
                device=sorted_pixels.device,
                dtype=sorted_pixels.dtype,
            )
        else:
            ranks = torch.tensor(
                [0.0], device=sorted_pixels.device, dtype=sorted_pixels.dtype
            )
        angles_coord = ranks * pi2

        color_wire = 0
        coord_wire = 1
        rank_wires = list(range(2, self.num_qubits))

        # superposition over ranks
        for w in rank_wires:
            qml.Hadamard(wires=w)

        # controlled rotations
        for k in range(self.num_pixels):
            bits = format(k, f"0{self.num_rank_qubits}b")

            for wire, bit in zip(rank_wires, bits):
                if bit == "0":
                    qml.PauliX(wires=wire)

            qml.ctrl(qml.RY, control=rank_wires)(2 * angles_color[k], wires=color_wire)
            qml.ctrl(qml.RY, control=rank_wires)(2 * angles_coord[k], wires=coord_wire)

            for wire, bit in zip(rank_wires, bits):
                if bit == "0":
                    qml.PauliX(wires=wire)
