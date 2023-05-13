#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run simple neural network 
echo -e "\n [INFO:] Running classification pipeline with simple neural network ..." # user msg 
python3.9 src/classify_NN.py -data FAKE
python3.9 src/classify_NN.py -data REAL

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Classifications using NN complete!"