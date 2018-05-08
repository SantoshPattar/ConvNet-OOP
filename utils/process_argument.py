#!/usr/bin/env python3
"""
Process the json configuration file.

Configuration file holds the parameters to intialize the CNN model.
These files are located in configaration_files folder.

Created on Wed May 2 09:02:54 2018

@author: Santosh Pattar
@author: Veerabadrappa
@version: 1.0
"""
import argparse

def get_args():
	"""
	Get arguments from the command line.

	:param none
	:return none
	:raises none
	"""
    
	parser = argparse.ArgumentParser( description = __doc__ )

	# Configuration file path argument.
	parser.add_argument(
						'-c', '--config',
						metavar = 'C',
						help = 'The Configuration file',
						default = './configuration_files/fashion_config.json',
						required = False
					)

	# Epoch size argument.
	parser.add_argument(
						'-e', '--epoch',
						metavar = 'E',
						help = 'Number of epoches for traning the model',
						default = 1,
						required = False
					)

	# Convert to dictonary.
	args = vars(parser.parse_args())

	if( args['config'] == './configuration_files/fashion_config.json' ):
		print('Using default configuration file.')
	else:
		print('Using configurations from file:', args['config'])

	if( args['epoch'] == 1 ):
		print('Using default epoch size of 1.')
	else:
		print('Using epoch size:', args['epoch'])
		
	return args
