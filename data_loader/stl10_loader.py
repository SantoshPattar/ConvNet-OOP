#!/usr/bin/env python3
"""
Implements the Stl10Loader class by inheriting the DataLoader base class.

STL-10 dataset is loaded from an external path (i.e., in dataset folder) and preprocessed.

Created on Sun Apr 22 20:54:32 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""

from base.data_loader_base import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class Stl10Loader(DataLoader):

	def __init__(self, config):
		"""
		Constructor to initialize the training and testing datasets for STL-10 dataset.

		:param config: the JSON configuration namespace.
		:return none
		:raises none
		"""

		super().__init__(config)
		return

	def load_dataset(self):
		"""
		Loads the STL-10 image dataset from the disk location, and
		Updates the respective class members.

		:param none
		:return none
		:raises none
		"""

		# Paths to the training and testing data for STL-10 dataset.
		path = './dataset/stl-10'
		train_data_path = os.path.join(path, 'train_X.bin')
		train_label_path = os.path.join(path, 'train_y.bin')
		test_data_path = os.path.join(path, 'test_X.bin')
		test_label_path = os.path.join(path, 'test_y.bin')

		# Read the training data images and thier labels from the disk.
		self.train_data = self.read_images(train_data_path)
		self.train_labels = self.read_labels(train_label_path)

		# Read the test data images and thier labels from the disk.
		self.test_data = self.read_images(test_data_path)
		self.test_labels = self.read_labels(test_label_path)

		return

	def read_images(self, path_to_data):
		"""
		Reads a binary file from the disk that contains image data, and
		stores them in numpy array.

		:param none
		:return none
		:raises none
		"""

		# FIXME: change the reshape's arguments with the config namespace keys.
		with open(path_to_data, 'rb') as f:
			everything = np.fromfile(f, dtype=np.uint8)
			images = np.reshape(everything, (-1, 3, 96, 96))
			images = np.transpose(images, (0, 3, 2, 1))
			return images

	def read_labels(self, path_to_labels):
		"""
		Reads a binary file from the disk that contains image labels, and
		stores them in numpy array.

		:param none
		:return none
		:raises none
		"""

		# FIXME: change the reshape's arguments with the config namespace keys.
		with open(path_to_labels, 'rb') as f:
			labels = np.fromfile(f, dtype=np.uint8)
			return labels

	def display_data_element(self, which_data, index):
		"""
		Displays a data element from the STL-10 dataset (training/testing).

		:param which_data: Specifies the dataset to be used (i.e., training or testing).
		:param index: Specifies the index of the data element within a particular dataset.
		:returns none
		:raises none
		"""

		# Create a new figure.
		plt.figure()

		# FIXME: modify appropriately to include save to disk option. Refer FashionMNSIT's display function.
		if(which_data == "train_data"):
			plt.imshow( self.train_data[index, : , : ] )
			#plt.imshow( self.trainData[index, : , : ] )
			# plt.show()
			plt.savefig('./resources/images/sample_training.png', bbox_inches='tight')
			plt.close()

		elif(which_data == "test_data"):
			plt.imshow( self.test_data[index,:,:] )
			#plt.imshow( self.testData[index,:,:])
			# plt.show()
			plt.savefig('./resources/images/sample_testing.png', bbox_inches='tight')
		else:
			print("Error: display_data_element: whicData parameter is invalid !")

		# Close the figure.
		plt.close()
		return

	def preprocessDataset(self):
		"""
		Preprocess the STL-10 dataset.

		Performs normalization on data values of training and testing dataset, and
		Converts the categorical class labels to boolean one-hot encoded vector for training and testing datasets.

		:param none
		:returns none
		:raises none
		"""
		# Convert the integer pixel data to floating data to speed up keras execution.
		#self.trainData = self.trainData.astype('float32')
		#self.testData = self.testData.astype('float32')

		# Rescale the pixel values from orignal values to the values in range 0 10 1.
		#self.trainData = self.trainData / self.config.config_namespace.image_pixel_size
		#self.testData = self.testData / self.config.config_namespace.image_pixel_size

		# Convert from categorical to  boolean one hot encoded vector.
		self.trainLabelsOneHot = to_categorical( self.train_labels )
		self.testLabelsOneHot = to_categorical( self.test_labels )
		return
