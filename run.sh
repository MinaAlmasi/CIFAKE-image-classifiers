#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# preprocess data 
echo -e "\n [INFO:] Create metadata ..." # user msg 
python src/metadata.py

# run simple neural network 
echo -e "\n [INFO:] Running classification pipeline with simple neural network ..." # user msg 
python src/classify_NN.py -data FAKE
python src/classify_NN.py -data REAL

# run LeNet CNN
echo -e "\n [INFO:] Running classification pipeline with LeNet CNN ..." # user msg 
python src/classify_LeNet.py -data FAKE
python src/classify_LeNet.py -data REAL

echo -e "\n [INFO:] Running classification pipeline with VGG16 CNN ..." # user msg 
python src/classify_VGG16.py -data FAKE 
python src/classify_VGG16.py -data REAL 

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Classifications complete!"