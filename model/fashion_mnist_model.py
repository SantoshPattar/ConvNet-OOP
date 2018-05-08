#!/usr/bin/env python3
"""
ConvNet model for fashion_mnist dataset.

Created on Mon Apr 23 09:44:25 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""

from base.model_base import BaseModel
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import time

class FashionMnistModel(BaseModel):

	def __init__(self, config, dataset):
		"""
		Constructor to initialize the ConvNet for FashionMNIST dataset.

		:param config: the JSON configuration namespace.
		:param dataset: Training and testing datasets.
		:return none
		:raises none
		"""

		super().__init__(config, dataset)
		return

	def define_model(self):
		"""
		Construct the ConvNet model.

		:param none
		:return none
		:raises none
		"""

		# TODO: two ways of defining
		# 1) intialize an array with all layers(remember there is layer size parameter in JSON) and pass it to Sequential constructor.
		# 2) add the layers to the model through "add" method of Sequential class.

		if( self.config.config_namespace.model_type == 'Sequential' ):
			print("The Keras ConvNet model type used for this experiment is: ", self.config.config_namespace.model_type)
			self.cnn_model = self.define_sequential_model()

		else:
			# TODO: handle functional model here.
			# self.cnn_model = Model()
			self.define_functional_model()

		# Summary of the ConvNet model.
		self.cnn_model.summary()
		return

	def define_sequential_model(self):
		"""
		Design a sequential ConvNet model.

		:param none
		:return cnn_model: The ConvNet sequential model.
		:raises none
		"""

		self.cnn_model = Sequential()

		# 1st Layer.
		self.cnn_model.add( Conv2D( filters = self.config.config_namespace.no_of_filters_l1,
					kernel_size = ( self.config.config_namespace.kernel_row, self.config.config_namespace.kernel_column ),
					activation = self.config.config_namespace.conv_activation_l1,
					input_shape = ( self.config.config_namespace.image_width, self.config.config_namespace.image_height, 						self.config.config_namespace.image_channel ),
					    padding = self.config.config_namespace.padding,
				            strides = self.config.config_namespace.stride_size
				        )
				)

		# 2nd Layer.
		self.cnn_model.add( LeakyReLU(alpha = self.config.config_namespace.relu_alpha) )

		# 3rd Layer.
		self.cnn_model.add( MaxPooling2D( pool_size = (self.config.config_namespace.pool_size_row, 							self.config.config_namespace.pool_size_column),
						padding = self.config.config_namespace.padding
				                )
				)

		# Add dropout layer, if necessary.
		if( self.config.config_namespace.dropout == 'true' ):
			self.cnn_model.add( Dropout( self.config.config_namespace.dropout_probability_l1 ) )

		# 4th Layer.
		self.cnn_model.add( Conv2D( filters = self.config.config_namespace.no_of_filters_l2,
					kernel_size = ( self.config.config_namespace.kernel_row, self.config.config_namespace.kernel_column ),
					activation = self.config.config_namespace.conv_activation_l2,
					padding = self.config.config_namespace.padding,
					strides = self.config.config_namespace.stride_size
				        )
				)

		# 5th Layer.
		self.cnn_model.add( LeakyReLU(alpha = self.config.config_namespace.relu_alpha) )

		# 6th Layer.
		self.cnn_model.add( MaxPooling2D( pool_size = (self.config.config_namespace.pool_size_row, 							self.config.config_namespace.pool_size_column),
						padding = self.config.config_namespace.padding
				                )
				)

		# Add dropout layer, if necessary.
		if( self.config.config_namespace.dropout == 'true' ):
			self.cnn_model.add( Dropout( self.config.config_namespace.dropout_probability_l1 ) )

		# 7th Layer.
		self.cnn_model.add( Conv2D( filters = self.config.config_namespace.no_of_filters_l3,
					kernel_size = ( self.config.config_namespace.kernel_row, 
					self.config.config_namespace.kernel_column ),
					activation = self.config.config_namespace.conv_activation_l3,
					padding = self.config.config_namespace.padding,
					strides = self.config.config_namespace.stride_size
				        )
				)

		# 8th Layer.
		self.cnn_model.add( LeakyReLU(alpha = self.config.config_namespace.relu_alpha) )

		# 9th Layer.
		self.cnn_model.add( MaxPooling2D( pool_size = (self.config.config_namespace.pool_size_row, 						self.config.config_namespace.pool_size_column),
					padding = self.config.config_namespace.padding
						)
				)

		# Add dropout layer, if necessary.
		if( self.config.config_namespace.dropout == 'true' ):
			self.cnn_model.add( Dropout( self.config.config_namespace.dropout_probability_l2 ) )

		# 10th Layer.
		self.cnn_model.add( Flatten() )

		# 11th Layer.
		self.cnn_model.add( Dense( units = self.config.config_namespace.no_of_filters_l4,
					activation = self.config.config_namespace.dense_activation_l1
					)
				)

		# 12th Layer.
		self.cnn_model.add( LeakyReLU(alpha = self.config.config_namespace.relu_alpha) )

		# Add dropout layer, if necessary.
		if( self.config.config_namespace.dropout == 'true' ):
			self.cnn_model.add( Dropout( self.config.config_namespace.dropout_probability_l3 ) )

		# 13th Layer.
		self.cnn_model.add( Dense( self.dataset.no_of_classes,
					activation = self.config.config_namespace.dense_activation_l2
					)
				)

		return self.cnn_model

	def define_functional_model(self):
		"""
		Define (construct) a functional ConvNet model.

		:param none
		:return cnn_model: The ConvNet sequential model.
		:raises none
		"""

		print("yet to be implemente\n")
		return

	def compile_model(self):
		"""
		Configure the ConvNet model.

		:param none
		:return none
		:raises none
		"""

		self.cnn_model.compile( loss = self.config.config_namespace.compile_loss,
								optimizer = self.config.config_namespace.compile_optimizer,
								metrics = [self.config.config_namespace.compile_metrics1]
							)


    # FIXME; check if dataset is need in the whole class, if not needed just pass it to required functions. (see grid search)
	def fit_model(self):
		"""
		Train the ConvNet model.

		:param none
		:return none
		:raises none
		"""

		start_time = time.time()

		if( self.config.config_namespace.save_model == "true"):
			print("Training phase under progress, trained ConvNet model will be saved at path", self.saved_model_path, " ...\n")
			self.history = self.cnn_model.fit( x = self.dataset.train_data,
											   y = self.dataset.train_label_one_hot ,
											   batch_size = self.config.config_namespace.batch_size,
											   epochs = self.config.config_namespace.num_epochs,
											   callbacks = self.callbacks_list,
											   verbose = self.config.config_namespace.fit_verbose,
											   validation_data = ( self.dataset.test_data, self.dataset.test_label_one_hot  )
											)
		else:
			print("Training phase under progress ...\n")
			self.history = self.cnn_model.fit( x = self.dataset.train_data,
											   y = self.dataset.train_label_one_hot ,
											   batch_size = self.config.config_namespace.batch_size,
											   epochs = self.config.config_namespace.num_epochs,
											   verbose = self.config.config_namespace.fit_verbose,
											   validation_data = ( self.dataset.test_data, self.dataset.test_label_one_hot  )
											)

		end_time = time.time()

		self.train_time = end_time - start_time
		print( "The model took %0.3f seconds to train.\n"%self.train_time )
		
		return

	def evaluate_model(self):
		"""
		Evaluate the ConvNet model.

		:param none
		:return none
		:raises none
		"""
		
		self.scores = self.cnn_model.evaluate( x = self.dataset.test_data,
                                               y = self.dataset.test_label_one_hot ,
                                               verbose = self.config.config_namespace.evaluate_verbose
                                            )

		print("Test loss: ", self.scores[0])
		print("Test accuracy: ", self.scores[1])

		return

	def predict(self):
		"""
		Predict the class labels of testing dataset.

		:param none
		:return none
		:raises none
		"""

		self.predictions = self.cnn_model.predict( x = self.dataset.test_data,
                                                   verbose = self.config.config_namespace.predict_verbose
                                                )

		return
