#!/usr/bin/env python3
"""
Execution Flow for the STL-10 experiment.

Created on Sun Apr 22 22:53:00 2018

@author: pattar
"""
# Reproduce results by seed-ing the random number generator.
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from utils.process_configuration import ConfigurationParameters
from data_loader.stl10_loader import Stl10Loader
#import base.model_base import BaseModel

def main():

	# Parse the configuration parameters for the CNN Model.
	config = ConfigurationParameters("./configuration_files/fashion_config.json")

	# Load the dataset from the library and print its details.
	dataset = Stl10Loader(config)

	# Test the loaded dataset,
	# i.e., display the first images in both training and testing dataset.
	dataset.displayDataElement( "train", 1 )
	dataset.displayDataElement( "test", 1 )

    # Construct the CNN Model.
	# model = mb.ModelBase(config)

if __name__ == '__main__':
    main()