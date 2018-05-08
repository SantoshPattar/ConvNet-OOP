#!/bin/sh

# Setup the execution path.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Function to display the help message.
usage() {
 
	echo "Usage: $0 -x, --experiment <string> [-c, --config <path>] [-e, --epoch <number>] [-h,--help]"
	echo	
	echo "Runs the ConvNet experiment."
	echo	
	echo "Mandatory or optional arguments to long options are also mandatory or optional for any corresponding short options."
	echo
	echo "Experiment options:"
	echo "-x, --experiment	name of the experiment to be run."
	echo "				As of now, acceptable values are:"
	echo "				fashion-mnist for FashionMNIST dataset"
	echo "				stl-10 for STL-10 dataset"
	echo "-c, --config		use this configuration file." 
	echo "-e, --epoch		number of training epoches."
	echo
	echo "Other options:"
	echo "-h, --help		display this help and exit."
}

# Check for mandatory arguments.
if [ $# -eq 0 ]
then
    echo "No arguments supplied."
    echo "-x, --experiment is compulsory."
    echo
    usage
    exit 1
fi

# Argument variables.
EXP=
CONFIG=
EPOCH=

# Parse the command line arguments.
ARGS=`getopt -o hx:c:e: --long help,experiment:,config:,epoch: -n 'run_experiments.sh' -- "$@"`
eval set -- "$ARGS"

while true; do
  case "$1" in
    -x | --experiment ) EXP=$2; shift 2 ;;
    -c | --config) CONFIG=$2; shift 2;;
    -e | --epoch ) EPOCH=$2; shift 2;;
    -h | --help ) usage; exit 0 ;;
    -- ) shift; break ;;
    * ) usage; exit 1 ;;
  esac
done

# Check for -experiment argument.
if [ -z $EXP ]
then
	echo "-x, --experiment is compulsory."
	echo 
	usage
	exit 1
fi

# Run the experiment with required arguments.
if [ "$EXP" = "fashion-mnist" ] && [ ! -z $CONFIG ] && [ ! -z $EPOCH ]
then
	echo "Executing FashionMNIST experiment with config file and epoch arguments."
	python ./mains/fashion_mnist_mains.py -c $CONFIG -e $EPOCH
elif [ "$EXP" = "fashion-mnist" ] && [ ! -z $EPOCH ]
then
	echo "Executing FashionMNIST experiment with epoch argument."
	python ./mains/fashion_mnist_mains.py -e $EPOCH
elif [ "$EXP" = "fashion-mnist" ] && [ ! -z $CONFIG ]
then
	echo "Executing FashionMNIST experiment with config argument."
	python ./mains/fashion_mnist_mains.py -c $CONFIG
elif [ "$EXP" = "fashion-mnist" ] 
then
	echo "Executing FashionMNIST experiment."
	python ./mains/fashion_mnist_mains.py 
elif [ "$EXP" = "stl-10" ]
then
	echo "Executing STL-10 experiment."
    python ./mains/stl10_mains.py  
else  
	echo "Invalid experiment-name !"
	echo
	usage
	exit 1
fi
