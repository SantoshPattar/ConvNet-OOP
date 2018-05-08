#!/usr/bin/env python3
"""
Defines an abstract class BaseModel, that wraps the ConvNet model construction process. 

It defines, configures, trains and evaluates a ConvNet model. 
Also supports prediction of unknown class-labels on testing dataset.

Saving and loading a predefined model are also supported.

Created on Mon Apr 23 09:04:04 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""

import numpy as np
import os
from keras.models import Sequential
from keras.callbacks import History, ModelCheckpoint

class BaseModel(object):
	def __init__(self, config, dataset):

		"""
		Constructor to initialize the ConvNet's architecture parameters.

		:param config: the JSON configuration namespace.
		:param dataset: the training and testing dataset.
		:return none
		:raises none
		"""

		# Configuaration parameters.
		self.config = config

		# Training and testing datasets.
		self.dataset = dataset

		# ConvNet model.
		self.cnn_model = Sequential()

		# History object, holds training history.
		self.history = History()

		# Saved model path.
		self.saved_model_path = os.path.join( self.config.config_namespace.saved_model_dir, "my_model.h5" )

		# Checkpoint for ConvNet model.
		self.checkpoint =  ModelCheckpoint( self.saved_model_path, 
											monitor = 'val_acc', 
											verbose = self.config.config_namespace.checkpoint_verbose, 
											save_best_only = True, 
											mode = 'max' 
										)

		# Callbacks list.
		self.callbacks_list = [self.checkpoint]

		# Evaluation scores.
		self.scores = []

		# Training time.
		self.train_time = 0

		# Predicted class labels.
		self.predictions = np.array([])

		# Validte the stride size.
		self.validate_stride()

		# Construct the ConvNet model.
		self.define_model()

		# Configure the ConvNet model.
		self.compile_model()

		# Train the ConvNet model using testing dataset.
		self.fit_model()

		# Evaluate the ConvNet model using testing dataset.
		self.evaluate_model()

		# Predict the class labels of testing dataset.
		self.predict()
		return

	def calculate_number_of_filters(self):
		"""
		Calaculates the filter size for a given layer.

		:param none
		:return none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to calculate the filter size.
		raise NotImplementedError

	def validate_stride(self):
		"""
		Validate the stride size based on the input data's size, filter's size and padding volume specified.

		:param none
		:return none
		:raises Exception: Invalid stride size.
		"""

		valid_stride_width = (
			self.config.config_namespace.image_width - self.config.config_namespace.kernel_row + 
			2 * self.config.config_namespace.padding_size 
			) / self.config.config_namespace.stride_size + 1

		valid_stride_height = (
			self.config.config_namespace.image_height - self.config.config_namespace.kernel_column + 
			2 * self.config.config_namespace.padding_size
			) / self.config.config_namespace.stride_size + 1

		if( not float(valid_stride_width).is_integer() 
			and 
			not float(valid_stride_height).is_integer() 
		):
			print("Invalid stride size specified, model does not fit to the specification. !")
			raise Exception
		else:
			return

	def define_model(self):
		"""
		Constructs the ConvNet model.

		:param none
		:return none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to add layers to the ConvNet.
		raise NotImplementedError

	def compile_model(self):
		"""
		Configures the ConvNet model.

		:param none
		:return none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to configure the ConvNet model.
		raise NotImplementedError

	def fit_model(self):
		"""
		Trains the ConvNet model.

		:param none
		:return none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to configure the ConvNet model.
		raise NotImplementedError

	def evaluate_model(self):
		"""
		Evaluates the ConvNet model.

		:param none
		:return none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to evaluate the constructed ConvNet model.
		raise NotImplementedError

	def predict(self):
		"""
		Predicts the class labels of unknown data.

		:param none
		:return none
		:raises NotImplementedError: Exception: Implement this method.
		"""

		# Implement this method in the inherited class to predict the class-labels of unknown data.
		raise NotImplementedError

	def save_model(self):
		"""
		Saves the ConvNet model to disk in h5 format.

		:param none
		:return none
		"""

		if( self.cnn_model is None ):
			raise Exception("ConvNet model not configured and trained !")
		
		self.cnn_model.save( self.saved_model_path )
		print("ConvNet model saved at path: ", self.saved_model_path, "\n")

		return

	def load_cnn_model(self):
		"""
		Loads the saved model from the disk.

		:param none
		:return none
		:raises NotImplementedError: Implement this method.
		"""

		if( self.cnn_model is None ):
			raise Exception("ConvNet model not configured and trained !")
		
		self.cnn_model.load_weights( self.saved_model_path )
		print("ConvNet model loaded from the path: ", self.saved_model_path, "\n")

		return
