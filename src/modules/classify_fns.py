'''
Script for self-assigned (Assignment 4), Visual Analytics, Cultural Data Science, F2023

This script comprises several functions which make up a pipeline for training models 

@MinaAlmasi
'''

# keras layers 
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# functions
def optimise_model(model): 
    '''
    Define a dynamic learning rate and compile the model with it (model optimisation).

    Args: 
        - model: intialised tensorflow model 
    
    Returns: 
        - model: model compiled with new learning rate ! 
    '''

    # define optimisation schedule 
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    
    # insert optimisation schedule in algorithm
    sgd = SGD(learning_rate=lr_schedule)

    # compile model with optimisation algorithm
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_data, validation_data, epochs:int):    
    '''
    Train initalised CNN for a specified amount of epochs. Evaluate model on validation data. 

    Args: 
        - model: intialised CNN
        - train_data: the data to be trained on 
        - validation_data: the data that the model evaluated on 
        - epochs: number of epochs that the model should train for

    Returns: 
        - history: history object containing information about the model training (e.g., loss and accuracy) 
            see documentation for more info: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History 
    '''
    # train model
    history = model.fit( # batch_size is not defined in model.fit as documentation specifies that is should not be done when using a data generator (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
        train_data, 
        validation_data = validation_data,
        epochs=epochs, 
        verbose=1, 
        use_multiprocessing=True
        )

    return history