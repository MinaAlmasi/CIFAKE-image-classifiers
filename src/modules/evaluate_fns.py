'''
Script for Assignment 3, Visual Analytics, Cultural Data Science, F2023

This script comprises several functions used for evaluating a CNN being trained with tensorflow. 
This entails saving the training and validation plot as well as the classification metrics.

The functions in this module are either adapted from a previous portfolio assignment or from class.
This will be stated in a comment beside the function.

@MinaAlmasi
'''

# util 
import numpy as np

# plotting
import matplotlib.pyplot as plt

# evaluation ! 
from sklearn.metrics import classification_report

# functions
def save_model_card(model, n_epochs:int, savepath:str, filename): # adapted from prev. assignment
    '''
    Save model card (summary of layers, trainable parameters) as txt file in desired directory (savepath).

    Args: 
        - model: model with defined layers
        - n_epochs: amount of epochs the model has been trained for
        - savepath: path where model card should be saved 
        - filename: what the .txt file should be called
    
    Outputs: 
        - .txt file of model summary in "savepath"
    '''

    # define full path
    filepath = savepath / filename
    
    # write model summary as txt
    with open(filepath,'w') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

def plot_model_history(model_hist, epochs, savepath, filename): # adapted from class notebook
    '''
    Plots two subplots, one for training and validation loss and the other for training and validation accuracy.
    Saves to .png file

    Args: 
        - model_hist: history of fitted model 
        - epochs: how many epochs the model has trained for 
        - savepath: where the .png should be saved
        - filename: name of plot (e.g., "history.png")

    Returns: 
        - .png file of plot in "savepath"
    '''

    # define theme 
    plt.style.use("seaborn-colorblind")

    # define figure size 
    plt.figure(figsize=(12,6))

    # create plot of train and validation loss, defined as two subplots on top of each other ! (but beside the accuracy plot)
    plt.subplot(1,2,1) #nrows, ncols, #index = position
    plt.plot(np.arange(1, epochs+1), model_hist.history["loss"], label="train_loss") # plot train loss 
    plt.plot(np.arange(1, epochs+1), model_hist.history["val_loss"], label="val_loss", linestyle=":") # plot val loss
    
    # text description on plot !!
    plt.title("Loss curve") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    # create a plot of train and validation accuracy, defined as two subplots on top of each other ! (but beside the loss plot)
    plt.subplot(1,2,2) #nrows, ncols, #index = position
    plt.plot(np.arange(0, epochs), model_hist.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), model_hist.history["val_accuracy"], label="val_acc", linestyle=":")

    # text description on plot !! 
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
   
    plt.savefig(savepath / filename, dpi=300)

def get_model_metrics(model, test_data):
    '''
    Evaluates fitted classifier model on test data, returns classification report.

    Args: 
        - model: fitted model to be evaluated
        - test_data: tensorflow test data

    Returns:
        - model_metrics: classification report containing information such as accuracy, F1, precision and recall 
    '''
    # make predictions
    predictions = model.predict(test_data)

    # extract true test vals
    y_test = test_data.classes

    # extract prediction with highest probability 
    y_pred = np.argmax(predictions, axis=1)

    # extract class names 
    class_names = test_data.class_indices.keys()

    # evaluate predictions
    model_metrics = classification_report(y_test, y_pred, target_names = class_names)

    return model_metrics

def save_model_metrics(model_metrics, savepath, filename): # adapted from prev. assignment
    '''
    Converts scikit-learn's classification report (metrics.classification_report) to a .txt file. 

    Args:
        - model_metrics: metrics report (sklearn.metrics.classification_report() or returned from clf_evaluate)
        - filename: filename for .txt report
        - savepath: directory where the text file should be stored. 
    
    Outputs: 
        - .txt file of metrics in "savepath"
    '''

    # define filename 
    filepath = savepath / filename

    # write model metrics to txt
    with open(filepath, "w") as file: 
        file.write(model_metrics)