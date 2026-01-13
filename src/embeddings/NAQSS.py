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


class NAQSS:
    """
    NAQSS – Normal Arbitrary Quantum Superposition State (prototype implementation).

    This embedding extends NEQR by introducing an additional segmentation qubit
    that encodes segmentation (regional) information for each pixel.

    Like NEQR, position qubits encode the pixel index in binary, color qubits encode the grayscale intensity in binary.

    NAQSS Extension - segmentation:
    The initial idea was a simple foreground/background categorization based on an intensity threshold (or the mean). However, this reduces the
    to a classical bit, so to take advantage of the qubit, the pixel value deviation from the mean is encoded using a proportional, 
    mirrored rotation. Pixels above the mean rotate in one direction, and pixels below rotate symmetrically in the opposite direction.

    State: (1/sqrt(N)) SUM_i |pos_i> |color_i> |seg_i>
    """

    def __init__(self, num_pixels, bit_depth=8, seg_scale=math.pi):
        self.num_pixels = num_pixels
        self.num_pos_qubits = int(math.log2(num_pixels))
        assert 2**self.num_pos_qubits == num_pixels, "num_pixels must be power of 2"

        self.bit_depth = bit_depth
        self.num_color_qubits = bit_depth
        self.seg_scale = seg_scale  # scales how strongly pixel deviations from the image mean are mapped to rotation angles

        self.num_qubits = self.num_pos_qubits + self.num_color_qubits + 1  #+1 qubit for segmentation

    def state_preparation(self, X: torch.Tensor):
        pixels = X.flatten().to(device).float()

        if pixels.min() < 0.0 or pixels.max() > 1.0:   # sanity check for normalization
            raise ValueError(
                f"NAQSS expects input pixels in [0,1], but got min={pixels.min().item():.3f}, max={pixels.max().item():.3f}"
            )

        # converting pixel values to integer grayscale just as in NEQR, now implemented using torch operations and without clipping, normalization should be enforced by the DataManager now
        max_val = (2**self.bit_depth) - 1
        pixel_values = torch.round(pixels * max_val).to(torch.int64)

        # qubit wire positions definition, position + color qubits just like in NEQR, +1 additional segmentation qubit
        pos_wires = list(range(self.num_pos_qubits))
        color_wires = list(range(self.num_pos_qubits, self.num_pos_qubits + self.num_color_qubits))
        seg_wire = self.num_qubits - 1

        mean_intensity = pixels.mean()

        # rest of state preparation follows NEQR - creates a uniform superposition over pixel position and encoding grayscale values conditionally,
        # using multi-controlled operations, but also additionaly encodes segmentation info - pixel intensity deviation from mean
        for w in pos_wires:
            qml.Hadamard(wires=w)

        for i in range(self.num_pixels):
            pos_bits = format(i, f"0{self.num_pos_qubits}b")

            for wire, bit in zip(pos_wires, pos_bits):
                if bit == "1":
                    qml.PauliX(wires=wire)

            color_bits = format(int(pixel_values[i].item()), f"0{self.num_color_qubits}b")
            for idx, bit in enumerate(color_bits[::-1]):
                if bit == "1":
                    qml.ctrl(qml.PauliX, control=pos_wires)(
                        wires=color_wires[idx]
                    )

            # Segmentation encoding (NAQSS extension from NEQR)
            # applies a continuous rotation on the segmentation qubit proportional to the pixel value's deviation from the mean, mirrored encoding
            # clamping to pi/2 so that rotations are within pi and to avoid wrap around
            seg_angle = torch.clamp(self.seg_scale * (pixels[i] - mean_intensity), -math.pi / 2, math.pi / 2)
            qml.ctrl(qml.RY, control=pos_wires)(2 * seg_angle, wires=seg_wire)

            for wire, bit in zip(pos_wires, pos_bits):
                if bit == "1":
                    qml.PauliX(wires=wire)