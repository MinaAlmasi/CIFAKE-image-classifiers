#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run LeNet CNN
echo -e "\n [INFO:] Running classification pipeline with LeNet CNN ..." # user msg 
python3.9 src/classify_LeNet.py -data FAKE
python3.9 src/classify_LeNet.py -data REAL

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Classifications using LeNet complete!"