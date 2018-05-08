#!/usr/bin/env python3
"""
Defines an abstract class DataLoader, that wraps the dataset loading process. 
Dataset can either be present in the library or on the disk.

It also supports dataset printing options.

Created on Sun Apr 22 20:17:31 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""

import numpy as np

class DataLoader(object):

	def __init__(self, config):
		"""
		Constructor to initialize the training, testing datasets and their properties.

		:param config: the JSON configuration namespace.
		:return none
		:raises none
		"""

		# Configuration parameters.
		self.config = config

		# Training Dataset's Data and Lables.
		self.train_data = np.array([])
		self.train_labels = np.array([])

		# Testing Dataset's Data and Lables.
		self.test_data = np.array([])
		self.test_labels = np.array([])

		# Total number of Class-Labels.
		self.no_of_classes = 0

		# Class Label List.
		self.list_of_classes = []

		# One-hot Encoded Label Vector.
		self.train_label_one_hot  = np.array([])
		self.test_label_one_hot  = np.array([])

		# Load the dataset from disk/library.
		self.load_dataset()

		# Calculate the number of class labels and list them.
		self.calculate_class_label_size()

		# Print the details of the dataset.
		self.print_dataset_details()

		# Preprocess the dataset (normalize, one-hot-shot encoding).
		self.preprocess_dataset()
		return

	def load_dataset(self):
		"""
		Loads the dataset.

		:param none
		:return none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to read the dataset from the disk.
		# Update respective data members of the DataLoader class.
		raise NotImplementedError

	def print_dataset_details(self):
		"""
		Prints the details of the dataset (training & testing size).

		:param none
		:return none
		:raises none
		"""

		# Number of samples in the dataset.
		print("Training dataset size (Data, Labels) is: ", self.train_data.shape, self.train_labels.shape)
		print("Testing dataset size (Data, Labels) is: ", self.test_data.shape, self.test_labels.shape)

		# Number of class labels and their list.
		print("Total number of Classes in the dataset: ", self.no_of_classes)
		print("The ", self.no_of_classes," Classes of the dataset are: ", self.list_of_classes)
		return

	def calculate_class_label_size(self):
		"""
		Calculates the total number of classes in the dataset.

		:param none
		:return none
		"""

		self.list_of_classes = np.unique(self.train_labels)
		self.no_of_classes = len(self.list_of_classes)
		print("Number of classes and its list from the loaded dataset calculated.")
		return

	def display_data_element(self, which_data, index):
		"""
		Displays a data element from a particular dataset (training/testing).

		:param which_data: Specifies the dataset to be used (i.e., training or testing).
		:param index: Specifies the index of the data element within a particular dataset.
		:returns none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to display a given data element.
		raise NotImplementedError

	def preprocess_dataset(self):
		"""
		Preprocess the dataset.

		:param none
		:returns none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to pre-process the dataset.
		# Data values in the training and testing dataset should be in floating point values, and
		# the class labels are to be in one-hot encoded vector.
		raise NotImplementedError