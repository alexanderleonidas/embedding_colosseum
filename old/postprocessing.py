import numpy as np
from math import pi, acos, sqrt, log2, ceil

class Postprocessing:
    """
    Class for postprocessing quantum image processing results.
    """

    @staticmethod
    def postprocess(method, counts, image):
        """
        Perform postprocessing based on the specified method.

        Args:
            method (str): The method used for quantum image processing.
            counts (dict): Dictionary containing measurement outcomes and their counts.
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Processed output image.
        """
        if method == 'FRQI':
            return Postprocessing.FRQI_post(counts, image)
        elif method == 'NEQR':
            return Postprocessing.NEQR_post(counts, image)
        elif method == 'OQIM':
            return Postprocessing.OQIM_post(counts, image)
        elif method == 'QPIE':
            return Postprocessing.QPIE_post(counts,image)
        else:
            raise ValueError("Invalid method specified.")

    @staticmethod
    def FRQI_post(counts, image):
        """
        Perform postprocessing for the FRQI method.

        Args:
            counts (dict): Dictionary containing measurement outcomes and their counts.
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Processed output image.
        """
        num_pixels = np.sqrt(image.shape[0])
        N = int(log2(num_pixels))

        num_pos_qubits = 2 * N
        
        classical_colors = []
        for i in range(0, len(image)):
            color_list = []
            for item in counts.items():
                key = item[0]
                amount = item[1]
                bin_coord = key[1:]
                int_coord = int(bin_coord, 2)
                if int_coord == i:
                    color_list.append((key[0], amount))
            color_amount = 0
            for color, amount in color_list:
                if not int(color):
                    color_amount=color_amount+amount
            try:
                color = np.arccos((color_amount/sum(n for _, n in color_list))**(1/2))
                classical_colors.append(color)
            except ZeroDivisionError:
                classical_colors.append(0)
        classical_colors = list(reversed(np.interp(classical_colors, (0, np.pi/2), (0, 256)).astype(int)))
        classical_colors = np.array(classical_colors)
        out_img = classical_colors.reshape(classical_colors.shape[0],1)
        
        return out_img

    @staticmethod
    def NEQR_post(counts, image):
        """
        Perform postprocessing for the NEQR method.

        Args:
            counts (dict): Dictionary containing measurement outcomes and their counts.
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Processed output image.
        """
        w_bits = 0
        h_bits = 0

        num_pixels = int(np.sqrt(image.shape[0]))
        input_image = image.reshape(num_pixels, num_pixels)

        w_bits = int(ceil(log2(input_image.shape[1])))
        h_bits = int(ceil(log2(input_image.shape[0])))
        if not w_bits:
            w_bits = 1
        if not h_bits:
            h_bits = 1
    
        out_pixels = []

        for state, _ in counts.items():
            out_pixels.append((int(state[0:w_bits + h_bits], 2), int(state[w_bits + h_bits:], 2)))
            
        out_image = np.zeros(len(image))
        for pixel in out_pixels:
            out_image[pixel[0]] = pixel[1]
        
        out_img = out_image.reshape(out_image.shape[0],1)
        
        return out_img

    @staticmethod
    def OQIM_post(counts, image):
        """
        Perform postprocessing for the OQIM method.

        Args:
            counts (dict): Dictionary containing measurement outcomes and their counts.
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Processed output image.
        """
        classical_colors = []
        classical_coords = []


        for i in range(0, len(image)):
            color_list = []
            coord_list = []
            for item in counts.items():
                key = item[0]
                amount = item[1]
                bin_coord = key[2:]
                int_coord = int(bin_coord, 2)
                if int_coord == i:
                    if not int(key[1]):
                        color_list.append((key[0], amount))
                    else:
                        coord_list.append((key[0], amount))
            color_amount = 0
            for color, amount in color_list:
                if not int(color):
                    color_amount=color_amount+amount
            try:
                color = np.arccos((color_amount/sum(n for _, n in color_list))**(1/2))
                classical_colors.append(color)
            except ZeroDivisionError:
                classical_colors.append(0)
                
            coord_amount = 0
            for coord, amount in coord_list:
                if not int(coord):
                    coord_amount=coord_amount+amount
            try:
                coord = np.arccos((coord_amount/sum(n for _, n in coord_list))**(1/2))
                classical_coords.append(coord)
            except ZeroDivisionError:
                classical_coords.append(0)
                
        classical_colors = np.interp(classical_colors, (0, np.pi / 2), (0, 256)).astype(int)
        classical_coords = np.interp(classical_coords, (0, np.pi / 2), (0, len(image))).astype(int)

        img = []
        for i in range(len(classical_coords)):
            x = classical_coords[i]
            img.insert(x, classical_colors[i])
        
        img = np.array(img)

        out_img = img.reshape(img.shape[0],1)
        
        return out_img
    
    @staticmethod
    def QPIE_post(state_vect,image):
        np_statevec = np.real(state_vect)
        image = np.interp(np_statevec, (0, np.pi/2), (0, 256)).astype(int)

        return image
