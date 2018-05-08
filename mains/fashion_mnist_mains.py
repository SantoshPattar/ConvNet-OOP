#!/usr/bin/env python3
"""
Execution Flow for the FashionMNIST experiment.

Created on Sun Apr 22 21:08:16 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""

# Reproduce results by seed-ing the random number generator.
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from utils.process_configuration import ConfigurationParameters
from data_loader.fashion_mnist_loader import FashionMnistLoader
from model.fashion_mnist_model import FashionMnistModel
from utils.model_utils import Report
from utils.process_argument import get_args

def main():

	try:

		# Capture the command line arguments from the interface script.
		args = get_args()

		# Parse the configuration parameters for the ConvNet Model.
		config = ConfigurationParameters( args )

	except:
		print( 'Missing or invalid arguments !' )
		exit(0)

	# Load the dataset from the library, process and print its details.
	dataset = FashionMnistLoader(config)

	# Test the loaded dataset by displaying the first image in both training and testing dataset.
	dataset.display_data_element( 'train_data', 1 )
	dataset.display_data_element( 'test_data', 1 )

	# Construct, compile, train and evaluate the ConvNet Model.
	model = FashionMnistModel(config, dataset)

	# Save the ConvNet model to the disk.
	# model.save_model()

	# Generate graphs, classification report, confusion matrix.
	report = Report(config,  model)
	report.plot()
	report.model_classification_report()
	report.plot_confusion_matrix()

if __name__ == '__main__':
	main()
