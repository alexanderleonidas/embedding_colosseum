import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import tensorflow as tf


import numpy as np


class FRQI:
    """
    Class for implementing Fast Randomized Quantum Image (FRQI) algorithm.
    """

    @staticmethod
    def preprocess(pixels):
        """
        Preprocess the image pixels.
        
        Args:
            pixels (numpy.ndarray): Input image pixels.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        image = pixels.astype('float64')
        angle = np.interp(image, (0, 256), (0, np.pi/2))
        return angle

    def _create_FRQI_circ(self,angles, measure_bool=False, printTime=False):
        """
        Create FRQI circuit.
        
        Args:
            circs (list): List to append created circuits.
            pixels (numpy.ndarray): Input image pixels.
            measure_bool (bool, optional): Flag to determine if measurements should be performed.
            printTime (bool, optional): Flag to print execution time.
            args (dict, optional): Additional arguments (not used).
        """
        start = time.time()
        #angles = self.preprocess(pixels)
        angles = tf.convert_to_tensor(angles)
        num_pixels = np.sqrt(angles.shape[0])
        N = int(np.log2(num_pixels))
        pos_qubits = 2 * N

        pos = QuantumRegister(pos_qubits, 'coordinates')
        color = QuantumRegister(1,'c_reg')
        cr = ClassicalRegister(pos.size+color.size, "cl_reg")

        circ = QuantumCircuit(color, pos, cr)
        circ.id(color)
        circ.h(pos)

        controls_ = []
        for i, _ in enumerate(pos):
            controls_.extend([pos[i]])

        for i, theta in enumerate(angles):
            qubit_index_bin = "{0:b}".format(i).zfill(pos_qubits)
            
            for k, qub_ind in enumerate(qubit_index_bin):
                if int(qub_ind):
                    circ.x(pos[k])
                    
            # qc_image.barrier()
            
            circ.mcry(theta = 2*theta.numpy(),
                        q_controls=controls_,
                        q_target=color[0],
                        mode = "noancilla",
                        use_basis_gates=True)
            
            
            
            # qc_image.barrier()
            for k, qub_ind in enumerate(qubit_index_bin):
                if int(qub_ind):
                    circ.x(pos[k])

        
        if measure_bool:
            circ = circ.reverse_bits()
            circ.measure(range(circ.num_qubits), range(cr.size))
        #else:
            #print('No measurements')

        end = time.time()

        if printTime:
            print('Time needed: {:5.3f}s'.format(end - start), 'for creating circuit via FRQI')

        # print("depth: {}, #qubits: {}".format(circ.depth(), circ.num_qubits))
        return circ
