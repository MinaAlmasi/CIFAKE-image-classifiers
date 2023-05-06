#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run simple neural network 
echo -e "\n [INFO:] Running classification pipeline with VGG16 CNN ..." # user msg 
python src/classify_VGG16.py -data REAL 
python src/classify_VGG16.py -data FAKE 

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Classifications complete!"