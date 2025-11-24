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


class NEQR:
    # Novel Enhanced Quantum Representation (NEQR) in PennyLane, inspired by the old one
    # Position qubits: encode pixel index in binary (|pos>)
    # Intensity qubits: encode 8-bit grayscale value (|color>)
    # State: (1/sqrt(N)) \sm_i |pos_i> |color_i>

    def __init__(
        self, num_pixels, bit_depth: int = 8
    ):  # bit_depth (int): number of bits for grayscale (e.g. 8 for 0–255)
        self.num_pixels = num_pixels
        self.num_pos_qubits = int(math.log2(num_pixels))
        assert 2**self.num_pos_qubits == num_pixels, "num_pixels must be a power of 2"

        self.bit_depth = bit_depth
        self.num_color_qubits = bit_depth
        self.num_qubits = self.num_pos_qubits + self.num_color_qubits

        self.device = qml.device("default.qubit", wires=self.num_qubits)

    def preprocess(self, pixels):
        # Convert pixels to integer grayscale values in [0, 2^bit_depth - 1].

        pixels = np.array(pixels, dtype=float)

        # If they look like 0–255, normalize to [0,1] first
        if pixels.max() > 1.5:
            pixels = np.clip(pixels, 0, 255)
            pixels = pixels / 255.0

        # Now pixels in [0,1] → scale to [0, 2^bit_depth - 1]
        max_val = (2**self.bit_depth) - 1
        pixels_scaled = np.rint(pixels * max_val).astype(int)
        pixels_scaled = np.clip(pixels_scaled, 0, max_val)

        return pixels_scaled

    def encode(self, pixel_values):
        pixel_values = np.array(pixel_values, dtype=int)
        assert len(pixel_values) == self.num_pixels, (
            "pixel_values length must match num_pixels."
        )

        # Define qubit indices: [0..num_pos_qubits-1] = position, [..] = color
        pos_wires = list(range(self.num_pos_qubits))
        color_wires = list(range(self.num_pos_qubits, self.num_qubits))

        @qml.qnode(self.device)
        def circuit():
            # Create superposition
            for w in pos_wires:
                qml.Hadamard(wires=w)

            # For each pixel i, encode its grayscale into color register
            #
            # follow the same logic as the old Qiskit code for each pixel index i:
            #       a) Compute binary representation of i (position)
            #       b) Flip those position qubits to turn |i> into |11..1>
            #       c) For each '1' bit in the pixel value, apply multi-controlled X with all position qubits as controls
            #       d) Uncompute (flip back) the position qubits
            #
            for i, val in enumerate(pixel_values):
                pos_bits = np.binary_repr(i, width=self.num_pos_qubits)

                for bit_idx, bit in enumerate(pos_bits):
                    if bit == "1":
                        qml.PauliX(wires=pos_wires[bit_idx])

                color_bits = np.binary_repr(val, width=self.num_color_qubits)

                for color_idx, bit in enumerate(color_bits[::-1]):
                    if bit == "1":
                        qml.ctrl(qml.PauliX, control=pos_wires)(
                            wires=color_wires[color_idx]
                        )

                for bit_idx, bit in enumerate(pos_bits):
                    if bit == "1":
                        qml.PauliX(wires=pos_wires[bit_idx])

            return qml.state()

        return circuit

    # I add here also the state_preparation to use for the VQC
    # This state_preparation was copied from encode()
    # to be used inside a VQC (same structure as FRQI.state_preparation)
    def state_preparation(self, X):
        pixels = X.flatten()  # torch tensor in [0,1] or [0,255]
        pixels = pixels.to(device)

        # --- PREPROCESS (Torch version of preprocess()) ---
        # Normalize if needed
        if torch.max(pixels) > 1.5:
            pixels = torch.clamp(pixels, 0.0, 255.0)
            pixels = pixels / 255.0

        # Scale to integer grayscale in [0, 2^bit_depth - 1]
        max_val = (2**self.bit_depth) - 1
        pixels = torch.clamp(pixels, 0.0, 1.0)
        pixel_values = torch.round(pixels * max_val).to(torch.int64)

        num_pos_qubits = self.num_pos_qubits
        num_color_qubits = self.num_color_qubits

        pos_wires = list(range(num_pos_qubits))
        color_wires = list(range(num_pos_qubits, self.num_qubits))

        # --- SUPERPOSITION OVER POSITION QUBITS ---
        for w in pos_wires:
            qml.Hadamard(wires=w)

        # --- ENCODE EACH PIXEL EXACTLY LIKE encode() ---
        for i in range(self.num_pixels):
            val = int(pixel_values[i].item())

            # position bits for index i
            pos_bits = format(i, f"0{num_pos_qubits}b")

            # flip pos qubits where bit == "1"
            for wire, bit in zip(pos_wires, pos_bits):
                if bit == "1":
                    qml.PauliX(wires=wire)

            # encode grayscale bits
            color_bits = format(val, f"0{num_color_qubits}b")

            # apply controlled-X for each '1' bit
            # reverse order to match encode() convention
            for color_idx, bit in enumerate(color_bits[::-1]):
                if bit == "1":
                    qml.ctrl(qml.PauliX, control=pos_wires)(
                        wires=color_wires[color_idx]
                    )

            # unflip
            for wire, bit in zip(pos_wires, pos_bits):
                if bit == "1":
                    qml.PauliX(wires=wire)
        pass
