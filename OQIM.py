import time
import numpy as np
from math import log2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

class OQIM:
    """
    Class for implementing Ordered Quantum Image Model (OQIM) algorithm.
    """

    @staticmethod
    def preprocess(pixels):
        """
        Preprocess the image pixels.

        Args:
            pixels (numpy.ndarray): Input image pixels.

        Returns:
            tuple: Tuple containing thetas and phis.
        """
        size = len(pixels)
        L = 256  # Maximum gray value
        
        # Initialize arrays
        order = np.zeros((2, size), dtype=object)
        temp = np.zeros((3, L+1), dtype=object)

        for position in range(len(pixels)):
            gray_level = int(pixels[position])
            temp[0, gray_level] += 1
            if temp[1, gray_level] == 0:
                temp[1, gray_level] = [gray_level]
            else:
                temp[1, gray_level].append(gray_level)
            if temp[2, gray_level] == 0:
                temp[2, gray_level] = [position]
            else:
                temp[2, gray_level].append(position)

        index = 0
        for l in range(L):
            if temp[0, l] > 0:
                for m in range(temp[0, l]):
                    order[0, index] = temp[1, l][m]
                    order[1, index] = temp[2, l][m]
                    index += 1

        color = np.array(order[0]).astype('float64')
        real_position = np.array(order[1]).astype('float64')

        thetas = np.interp(color, (0, 256), (0, np.pi/2))
        phis = np.interp(real_position, (0, len(pixels)), (0, np.pi/2))

        return thetas, phis

    def create_OQIM_circ(self, pixels, measure_bool=True, printTime=True):
        """
        Create OQIM circuit.

        Args:
            pixels (numpy.ndarray): Input image pixels.
            measure_bool (bool, optional): Flag to determine if measurements should be performed.
            printTime (bool, optional): Flag to print execution time.

        Returns:
            QuantumCircuit: The created quantum circuit.
        """
        angles, positions = self.preprocess(pixels)

        start = time.time()

        num_pixels = np.sqrt(pixels.shape[0])
        N = int(log2(num_pixels))
        pos_qubits = 2 * N

        O = QuantumRegister(pos_qubits, 'o_reg')
        c = QuantumRegister(1, 'c_reg')
        p = QuantumRegister(1, 'p_reg')
        cr = ClassicalRegister(O.size + c.size + p.size, "cl_reg")

        circ = QuantumCircuit(c, p, O, cr)

        circ.id(c)
        circ.h(p)
        circ.h(O)

        controls_ = [O[i] for i in range(len(O))] + [p[i] for i in range(len(p))]

        for i, (phi, theta) in enumerate(zip(positions, angles)):
            qubit_index_bin = "{0:b}".format(i).zfill(pos_qubits)

            for k, qub_ind in enumerate(qubit_index_bin):
                if int(qub_ind):
                    circ.x(O[k])
            
           
            for coord_or_intns in (0,1):
                if not coord_or_intns:
                    circ.mcry(theta=2*theta,
                                q_controls=controls_,
                                q_target=c[0],
                                use_basis_gates=True)
                                
                else:
                    circ.x(p)
                    circ.mcry(theta=2*phi,
                                q_controls=controls_,
                                q_target=c[0],
                                use_basis_gates=True)
                    if i!=len(pixels) - 1:
                        circ.x(p)

            if  i!=len(pixels) - 1:
                for k, qub_ind in enumerate(qubit_index_bin):
                    if int(qub_ind):
                        circ.x(O[k])
            circ.barrier()

        circ = circ.reverse_bits()
        if measure_bool:
            circ.measure(range(circ.num_qubits), range(cr.size))
        #else:
            #print('No measurements')
        end = time.time()

        if printTime:
            print('Time needed: {:5.3f}s'.format(end - start), 'for creating circuit via OQIM')

        
        #print("depth: {}, #qubits: {}".format(circ.depth(), circ.num_qubits))

        return circ
