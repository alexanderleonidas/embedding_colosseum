import numpy as np
import h5py
from PIL import Image
import tensorflow as tf

class MNISTDataLoader:

    def load_and_resize_images(self, pixels=28):

        # Define the desired size for the images
        desired_size = (pixels, pixels)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        resized_train_images = [self.resize_image(image,desired_size) for image in x_train]
        resized_test_images = [self.resize_image(image,desired_size) for image in x_test]

        return np.array(resized_train_images), y_train, np.array(resized_test_images), y_test
        
    def resize_image(self,image, size):
        img = Image.fromarray(image)
        img_resized = img.resize(size)
        img_resized = np.array(img_resized)
        input_image = img_resized.flatten()

        return input_image

    def get_image(self,image,label):
        
        input_image = image.reshape(1,-1)

        return input_image, label
    
    def select_classes(self, x_data, y_data, class1, class2):
        # Filter data based on specified classes
        selected_indices = np.where((y_data == class1) | (y_data == class2))
        selected_x_data = x_data[selected_indices]
        selected_y_data = y_data[selected_indices]

        return selected_x_data, selected_y_data


class CatsDataset:
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_and_resize_images(self, size=(64, 64)):
        train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes

        resized_train_images = [self.resize_image(image,size) for image in train_set_x_orig]
        resized_test_images = [self.resize_image(image,size) for image in test_set_x_orig]
        
        return np.array(resized_train_images), train_set_y_orig, np.array(resized_test_images), test_set_y_orig, classes
    
    def resize_image(self, image, size):
        img = Image.fromarray(image)
        img_resized = img.resize(size)
        img_gray = img_resized.convert('L')
        img_gray = np.array(img_gray)
        input_image = img_gray.flatten()
        return input_image
    
    def get_image(self,image,label):
        
        input_image = image.reshape(-1,1)

        return input_image,label