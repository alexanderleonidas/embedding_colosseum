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


class FRQI:
    def __init__(self, num_pixels):
        self.num_pixels = num_pixels
        self.num_qubits = (
            int(math.log2(num_pixels)) + 1
        )  # for a (2^N*2^N) image, the #_of_qbits = log2(num_pixels) +1 for color qubit
        self.device = qml.device("default.qubit", wires=self.num_qubits)

    # Maps the normalized pixel values ([0,1]) into rotation angles [0, π/2]
    def preprocess(self, pixels):
        pixels = np.array(pixels, dtype=float)

        # the DataManager should handle normalization in resize_images, but just in case it is not always so
        if pixels.max() > 1.5:
            pixels = np.clip(pixels, 0, 255)
            pixels = pixels / 255.0

        return np.interp(pixels, (0, 1), (0, np.pi / 2))

    # Creates and returns the FRQI encoding circuit
    def encode(self, angles):
        num_pos_qubits = self.num_qubits - 1
        color_wire = 0  # wire index of the control qubit in the circuit
        pos_wires = list(
            range(1, self.num_qubits)
        )  # wire indexes of the position qubits in the circuit
        # pos_wires = list(range(1, self.num_qubits))[::-1]

        @qml.qnode(self.device)
        def circuit():
            # creating a superposition of basis states over all position qubits
            for w in pos_wires:
                qml.Hadamard(wires=w)

            # encoding each pixels brightness into the color qubit
            # now matches the Qiskit implementation more closely, differs only in which bits are flipped before/after the rotation
            for i, theta in enumerate(angles):
                binary = np.binary_repr(
                    i, width=num_pos_qubits
                )  # converting i to binary/basis states, to iterate over each state's position qubits

                # the controlled rotations activate when all position qubits are activated
                # in the Qiskit impl. mcry() was used with the qubits being flipped before/after when their the corr. bit string == 1,
                # so the assumption is mcry() rotates when all control qubits == 0
                # In this implementation we invert the logic and flip the qubits when their value == 0, so the rotation is done with the state being all 1s
                # this results in equivalent embedding amplitudes as the old qiskit version, just the order of basis states is reversed

                for wire, bit in zip(
                    pos_wires, binary
                ):  # flipping the qubits which bit string == 0
                    if bit == "0":
                        qml.PauliX(wires=wire)

                # the state is in all 1s when the controlled RY is applied
                qml.ctrl(qml.RY, control=pos_wires)(2 * theta, wires=color_wire)

                # the qubits corresponding to 0 bits in the bitstring are unflipped
                for wire, bit in zip(pos_wires, binary):
                    if bit == "0":
                        qml.PauliX(wires=wire)

            return qml.state()  # the final state - quantum embedding of the image

        return circuit  # returns the compiled circuit as a QNode object that holds the final state

    # This state_preparation was copied from the circuit above
    # to be used in the VQC
    def state_preparation(self, X: torch.Tensor):
        pixels = X

        # Allow both single-sample (num_pixels,) and batched (B, num_pixels) inputs
        if pixels.dim() == 1:
            pixels = pixels.flatten()
        else:
            # keep batch dim, flatten remaining dims
            batch_size = pixels.shape[0]
            pixels = pixels.view(batch_size, -1)

        pixels = pixels.to(device)

        # the DataManager should handle normalization in resize_images, but just in case it is not always so
        # if pixels.max() > 1.5:
        #     pixels = np.clip(pixels, 0, 255)
        #     pixels = pixels / 255.0

        # angles = np.interp(pixels, (0, 1), (0, np.pi / 2))
        # Ensure lerp endpoints have the same dtype and device as `pixels`
        zero = torch.tensor(0.0, dtype=pixels.dtype, device=device)
        pi2 = torch.tensor(math.pi / 2, dtype=pixels.dtype, device=device)
        angles = torch.lerp(zero, pi2, pixels)

        batch_mode = angles.dim() == 2  # True if shape (B, num_pixels)

        num_pos_qubits = self.num_qubits - 1
        color_wire = 0  # wire index of the control qubit in the circuit
        pos_wires = list(
            range(1, self.num_qubits)
        )  # wire indexes of the position qubits in the circuit
        # pos_wires = list(range(1, self.num_qubits))[::-1]

        # creating a superposition of basis states over all position qubits
        for w in pos_wires:
            qml.Hadamard(wires=w)

        # encoding each pixels brightness into the color qubit
        # now matches the Qiskit implementation more closely, differs only in which bits are flipped before/after the rotation
        for i, theta in enumerate(angles):
            # converting i to binary/basis states, to iterate over each state's position qubits
            binary = np.binary_repr(i, width=num_pos_qubits)

            # the controlled rotations activate when all position qubits are activated
            # in the Qiskit impl. mcry() was used with the qubits being flipped before/after when their the corr. bit string == 1,
            # so the assumption is mcry() rotates when all control qubits == 0
            # In this implementation we invert the logic and flip the qubits when their value == 0, so the rotation is done with the state being all 1s
            # this results in equivalent embedding amplitudes as the old qiskit version, just the order of basis states is reversed

            for wire, bit in zip(
                pos_wires, binary
            ):  # flipping the qubits which bit string == 0
                if bit == "0":
                    qml.PauliX(wires=wire)

            # select theta for single or batched inputs
            if batch_mode:
                theta = angles[:, i]  # shape (B,)
            else:
                theta = angles[i]  # scalar

            # the state is in all 1s when the controlled RY is applied
            qml.ctrl(qml.RY, control=pos_wires)(2 * theta, wires=color_wire)

            # the qubits corresponding to 0 bits in the bitstring are unflipped
            for wire, bit in zip(pos_wires, binary):
                if bit == "0":
                    qml.PauliX(wires=wire)
        pass
