#!/usr/bin/env python3
"""
Implements grid search for a optimal set of hyperparameters to train the ConvNet model for fashion_mnist dataset.

Created on Thu Apr 26 01:28:17 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""

from base.grid_search_base import GridSearchBase
from model.fashion_mnist_model import FashionMnistModel
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from model.fashion_mnist_model import FashionMnistModel
from data_loader.fashion_mnist_loader import FashionMnistLoader
from utils.process_argument import get_args
from utils.process_configuration import ConfigurationParameters


def create_model(self):
	"""
	creates and compiles the ConvNet model.

	:param none
	:return cnn_model: The configured ConvNet model.
	:raises none
	"""
	
	model = FashionMnistModel(self.config, self.dataset)
	cnn_model = model.define_model()
	return cnn_model

def main():

	try:
		args = get_args()

		# Parse the configuration parameters for the CNN Model.
		config = ConfigurationParameters( args )

	except:
		print( 'Missing or invalid arguments !' )
		exit(0)
	

	dataset = FashionMnistLoader(config)
	
	g_search = GridSearchBase(config, dataset)

	# create a Scikit-learn wrapper.
	g_search.model_wrapper = KerasClassifier(build_fn = g_search.create_model, verbose=0)

	# define the grid search parameters
	batch_size = [1, 2]
	epochs = [300, 500]
	g_search.param_grid = dict(
								batch_size=batch_size, 
								epochs=epochs
							)

	g_search.grid = GridSearchCV(	estimator=g_search.model_wrapper, 
									param_grid=g_search.param_grid, 
									n_jobs=g_search.n_jobs
								)
	g_search.grid_result = g_search.grid.fit(
												dataset.train_data, 
												dataset.train_label_one_hot
											)

	# summarize results
	print("Best: %f using %s" % (g_search.grid_result.best_score_, g_search.grid_result.best_params_))
	means = g_search.grid_result.cv_results_['mean_test_score']
	stds = g_search.grid_result.cv_results_['std_test_score']
	params = g_search.grid_result.cv_results_['params']

	for mean, stdev, param in zip(means, stds, params):
			print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == '__main__':
	main()