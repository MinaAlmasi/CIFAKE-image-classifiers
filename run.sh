#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# preprocess data 
echo -e "\n [INFO:] Create metadata ..." # user msg 
python3 src/create_metadata.py

# run simple neural network 
echo -e "\n [INFO:] Running classification pipeline with simple neural network ..." # user msg 
python3 src/classify_NN.py -data FAKE
python3 src/classify_NN.py -data REAL

# run LeNet CNN
echo -e "\n [INFO:] Running classification pipeline with LeNet CNN ..." # user msg 
python3 src/classify_LeNet.py -data FAKE
python3 src/classify_LeNet.py -data REAL

# run VGG16
echo -e "\n [INFO:] Running classification pipeline with VGG16 CNN ..." # user msg 
python3 src/classify_VGG16.py -data FAKE 
python3 src/classify_VGG16.py -data REAL 

# run visualisation
python3 src/visualise.py 

# eval best classifiers on REAL test data 
python3 src/final_evaluate.py 

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Classifications complete!"