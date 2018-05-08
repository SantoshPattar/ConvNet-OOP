#!/usr/bin/env python3
"""
Implements the FashionMnistLoader class by inheriting the DataLoader base class.

Keras provides the FashionMNIST dataset in its datasets package.
This class loads the dataset from the keras library and pre-process it.

Created on Sun Apr 22 20:39:33 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""

from base.data_loader_base import DataLoader
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os

class FashionMnistLoader(DataLoader):

	def __init__(self, config):
		"""
		Constructor to initialize the training and testing datasets for FashionMNIST.

		:param config: the json configuration namespace.
		:return none
		:raises none
		"""

		super().__init__(config)
		return
		
	def load_dataset(self):
		"""
		Loads the fashion_mnist image dataset, and
		Updates the respective class members.

		:param none
		:return none
		:raises none
		"""

		# Load the dataset from the Keras library.
		print("Loading the dataset from the Keras library ...")
		(self.train_data, self.train_labels), (self.test_data, self.test_labels) = fashion_mnist.load_data()

		print("Dataset loaded successfully from the Keras library for the experiment", self.config.config_namespace.exp_name, "\n")
		
		# Reshape the training and testing data arrays to include channel (since gray scale only 1 channel).
		self.train_data = self.train_data.reshape( -1, 
							self.config.config_namespace.image_width, 
							self.config.config_namespace.image_height, 
							self.config.config_namespace.image_channel
							)

		self.test_data = self.test_data.reshape( -1, 
							self.config.config_namespace.image_width, 
						        self.config.config_namespace.image_height, 
							self.config.config_namespace.image_channel
							)

		print("Training data and testing data are reshaped to size: " , 
							self.config.config_namespace.image_width,
							self.config.config_namespace.image_height, 
							self.config.config_namespace.image_channel,
			  "\n")

	def display_data_element(self, which_data, index):
		"""
		Displays a data element from the FashionMNIST dataset (training/testing).

		:param  which_data: Specifies the dataset to be used (i.e., training or testing).
		:param index: Specifies the index of the data element within a particular dataset.
		:returns none
		:raises none
		"""

		# Create a new figure.
		plt.figure()

		# Display a training data element.
		if(which_data == "train_data"):
			plt.imshow( self.train_data[index, : , : ].reshape( self.config.config_namespace.image_width,
																self.config.config_namespace.image_height 
															),
						self.config.config_namespace.cmap_val
					)
			
			plt.title("Class Label of the Training Image is: {}".format(self.train_labels[index]))
			train_image_path = os.path.join( self.config.config_namespace.image_dir, "sample_training_fashion_mnist_image.png")

			if(self.config.config_namespace.save_plots == 'true'):
				plt.savefig( train_image_path , bbox_inches='tight')
				print("The ", which_data, " data from the index ", index, " is saved at path: ", train_image_path)
			else:
				plt.show()
				print("The ", which_data, " data from the index ", index, " is displayed.")

		# Display a testing data element.
		elif(which_data == "test_data"):
			plt.imshow( self.test_data[index,:,:].reshape( self.config.config_namespace.image_width, 
														   self.config.config_namespace.image_height 
														),
						self.config.config_namespace.cmap_val
					)

			plt.title("Class Label of the Testing Image is: {}".format(self.test_labels[index]))
			test_image_path = os.path.join( self.config.config_namespace.image_dir, "sample_testing_fashion_mnist_image.png")
			
			if(self.config.config_namespace.save_plots == 'true'):
				plt.savefig( test_image_path , bbox_inches='tight')
				print("The ", which_data, " data from the index ", index, " is saved at path: ", test_image_path)
			else:
				plt.show()
				print("The ", which_data, " data from the index ", index, " is displayed.")
				
		else:
			print("Error: display_data_element: whicData parameter is invalid !")

		# Close the figure.
		plt.close()
		return

	def preprocess_dataset(self):
		"""
		Preprocess the FashionMNIST dataset.

		Performs data type conversion and normalization on data values of training and testing dataset, and
		Converts the categorical class labels to boolean one-hot encoded vector for training and testing datasets.

		:param none
		:returns none
		:raises none
		"""

		# Convert the integer pixel data to floating data.
		self.train_data = self.train_data.astype('float32')
		self.test_data = self.test_data.astype('float32')

		# Rescale the pixel values from orignal values to the values in range 0 10 1.
		# Since, fashion_mnist has 256 possible pixel values we divide it by 255 to normalize in range 0 to 1.
		self.train_data = self.train_data / self.config.config_namespace.image_pixel_size
		self.test_data = self.test_data / self.config.config_namespace.image_pixel_size

		# Convert the class labels from categorical to boolean one hot encoded vector.
		self.train_label_one_hot  = to_categorical( self.train_labels )
		self.test_label_one_hot  = to_categorical( self.test_labels )

		print("Training and testing datasets are normalized and their respective class labels are converted to one-hot encoded vector. \n")
		return
