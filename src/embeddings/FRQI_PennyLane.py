import math
import torch
import pennylane as qml
from pennylane import numpy as np

class FRQI:
    def __init__(self, num_pixels):
        self.num_pixels = num_pixels
        self.num_qubits = int(math.log2(num_pixels)) + 1  # for a (2^N*2^N) image, the #_of_qbits = log2(num_pixels) +1 for color qubit
        self.device = qml.device("default.qubit", wires=self.num_qubits)

    # Maps the normalized pixel values ([0,1]) into rotation angles [0, π/2]
    def preprocess(self, pixels):
        pixels = np.array(pixels, dtype=float)

        # the DataManager should handle normalization in resize_images, but just in case it is not always so
        if pixels.max() > 1.5:  
            pixels = np.clip(pixels, 0, 255)       
            pixels = pixels / 255.0

        return np.interp(pixels, (0, 1), (0, np.pi/2))
    
    # Creates and returns the FRQI encoding circuit
    def encode(self, angles):
        num_pos_qubits = self.num_qubits - 1
        color_wire = 0      # wire index of the control qubit in the circuit
        pos_wires = list(range(1, self.num_qubits)) # wire indexes of the position qubits in the circuit

        @qml.qnode(self.device)
        def circuit():
            # creating a superposition of basis states over all position qubits
            for w in pos_wires:
                qml.Hadamard(wires=w)

            # encoding each pixels brightness into the color qubit, slightly differs from old Qiskit implementation by using qml.ctrl() on control qubits for rotations instead of X-flips + mcry 
            # the resulting embeddings between qml/qiskit are equivalent between, but the wire indexing differs when comparing the amplitudes
            for i, theta in enumerate(angles):

                binary = np.binary_repr(i, width=num_pos_qubits)  # converting i to binary/basis states, to iterate over each state's position qubits
                control_wires = []                 # and identify the position of control qubits within that will act on rotations
                for j, bit in enumerate(binary):
                    if bit == "1":
                        control_wires.append(pos_wires[j])

                # applying controlled RY rotations on the color qubit, entangling it with position qubits of each basis state based on the corresponding angles
                if control_wires:   # rotates the color qubit entangling it with the position qubits that act as control for the given basis state
                    qml.ctrl(qml.RY, control=control_wires)(2 * theta, wires=color_wire) 
                else:  
                    qml.RY(2 * theta, wires=color_wire)   # plain RY rotation case for the first pixel with no control qubits (index 0/basis state 00..)

            return qml.state()  # the final state - quantum embedding of the image

        return circuit    # returns the compiled circuit as a QNode object that holds the final state
