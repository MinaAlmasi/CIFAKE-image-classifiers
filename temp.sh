#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run simple neural network 
echo -e "\n [INFO:] Running classification pipeline with LeNet CNN ..." # user msg 
python src/classify_LeNet.py -data REAL
python src/classify_LeNet.py 

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Classifications complete!"