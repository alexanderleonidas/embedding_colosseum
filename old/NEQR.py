import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from math import log2, ceil

class NEQR:
    """
    Class for implementing Novel Enhanced Quantum Representation (NEQR) algorithm.
    """

    @staticmethod
    def preprocess(pixels):
        """
        Preprocess the image pixels.

        Args:
            pixels (numpy.ndarray): Input image pixels.

        Returns:
            tuple: Tuple containing the number of bits for width and height.
        """
        w_bits = 0
        h_bits = 0

        num_pixels = int(np.sqrt(pixels.shape[0]))
        image = pixels.reshape(num_pixels, num_pixels)

        w_bits = int(ceil(log2(image.shape[1])))
        h_bits = int(ceil(log2(image.shape[0])))
        if not w_bits:
            w_bits = 1
        if not h_bits:
            h_bits = 1

        return w_bits, h_bits

    def _create_NEQR_circ(self, pixels, measure_bool=True, printTime=True):
        """
        Create NEQR circuit.

        Args:
            circs (list): List to append created circuits.
            pixels (numpy.ndarray): Input image pixels.
            measure_bool (bool, optional): Flag to determine if measurements should be performed.
            printTime (bool, optional): Flag to print execution time.
            args (dict, optional): Additional arguments (not used).

        Returns:
            QuantumCircuit: The created quantum circuit.
        """
        start = time.time()
        dictionary = {}

        w_bits, h_bits = self.preprocess(pixels)

        pos = QuantumRegister(w_bits + h_bits, 'pos')
        intensity = QuantumRegister(8, 'intensity')
        cr = ClassicalRegister(len(pos) + len(intensity), 'cr')
        circ = QuantumCircuit(intensity, pos, cr)

        circ.i(intensity)
        circ.h(pos)

        for i, pixel in enumerate(pixels):

            pixel_bin = "{0:b}".format(int(pixel)).zfill(len(intensity))
            position = "{0:b}".format(i).zfill(len(pos))

            for j, coord in enumerate(position):
                if int(coord):
                    circ.x(circ.num_qubits - j - 1)
            circ.barrier()
            for idx, px_value in enumerate(pixel_bin[::-1]):
                if px_value == '1':
                    control_qubits = list(range(intensity.size, intensity.size + pos.size))
                    target_qubit = intensity[idx]
                    circ.mcx(control_qubits, target_qubit)

            if i != len(pixels) - 1:
                for j, coord in enumerate(position):
                    if int(coord):
                        circ.x(circ.num_qubits - j - 1)

            circ.barrier()

        if measure_bool:
            circ.measure(range(circ.num_qubits), range(cr.size))
        
            #print('No measurements')
        end = time.time()

        if printTime:
            print('Time needed: {:5.3f}s'.format(end - start), 'for creating circuit via NEQR')

        return circ
