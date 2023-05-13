#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# run VGG16
echo -e "\n [INFO:] Running classification pipeline with VGG16 CNN ..." # user msg 
python3 src/classify_VGG16.py -data FAKE 
python3 src/classify_VGG16.py -data REAL 

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Classifications using VGG16 complete!"