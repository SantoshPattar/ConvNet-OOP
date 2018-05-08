#!/usr/bin/env python3
"""
Defines an abstract class GridSearchBase, that wraps the hyperparameter search process. 

It search for a optimal set of hyperparameters to train the ConvNet model.

Created on Thu Apr 26 01:28:33 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from base.model_base import BaseModel

class GridSearchBase(BaseModel):

	def __init__(self, config, dataset):
		"""
		Initializes the grid search parameters.

		:param config: The configuration file.
		:return none
		:raises none
		"""

		# Configuration parameters.
		self.config = config

		# Dataset.
		self.dataset = dataset

		# Scikit-learn wrapper.
		self.model_wrapper = 0

		# Dictionary of hyperparameters.
		self.param_grid = {}

		# Parallel processing option, -1 for true and 1 for false.
		self.n_jobs = 1

		# Grid search model.
		self.grid = GridSearchCV(self.model_wrapper, self.param_grid)

		# Grid search results.
		self.grid_result = 0

		super().__init__(config)
		return

	def create_model(self):
		"""
		creates and compiles the ConvNet model.

		:param none
		:return none
		:raises NotImplementedError: Implement this method.
		"""

		# Implement this method in the inherited class to define, configure and compile the ConvNet model.
		raise NotImplementedError