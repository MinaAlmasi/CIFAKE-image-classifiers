'''
Utils script for Assignment 4, Visual Analytics, Cultural Data Science, F2023

The following script contains:
    - a function that loads model histories in a given directory
    - two functions to plot model histories in a 2x2 formation (add_headers() and plot_histories())
    - functions to take .txt files of metrics data and create a final table showcasing one metric for each .txt file (e.g., F1_score for all models!)

@MinaAlmasi
'''

# utils
import pickle
import pathlib
import re 

# plotting 
from matplotlib import pyplot as plt
import numpy as np

# table 
import pandas as pd
from tabulate import tabulate 


def load_model_histories(resultspath:pathlib.Path):
    '''
    Loads all history objects from a given path and returns them in a dictionary.

    Args:
        - resultspath: path to directory containing history objects
    
    Returns: 
        - history_objects: dictionary containing all history objects in path 
    '''
    # define empty dictionary where history objects will be saved 
    history_objects = {}

    for file in resultspath.iterdir():
        if "history" in file.name: # open all files which have "history" in their name
            with open(resultspath / file.name , 'rb') as f:
                # load history object 
                history = pickle.load(f)
                # define history object name (e.g., "REAL_LeNet_history.pickle" -> "REAL_LeNet")
                history_name = re.sub("_history.pickle", "", file.name)
                # add to history_objects dict ! 
                history_objects[history_name] = history
    
    return history_objects 

# add_headers() used in plot_histories() taken from https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots 
def add_headers(fig,*,row_headers=None,col_headers=None,row_pad=1,col_pad=5,**text_kwargs): # (adapted slightly, removed ability to rotate headers, added comments and docstring)
    '''                                                                                     
    Add row and column headers to a figure created with matplotlib.pyplot.subplots.

    Args:
        - fig: figure object to annotate
        - row_headers: list of strings to be used as row headers e.g., ["Row 1", "Row 2"]
        - col_headers: list of strings to be used as column headers e.g., ["Column 1", "Column 2"]
        - row_pad: padding between tick labels and row headers. Defaults to 1.
        - col_pad: padding between tick labels and column headers. Defaults to 5.
        - **text_kwargs: keyword arguments passed to the text objects for the row and column headers
    '''

    # get axes
    axes = fig.get_axes()

    # iterate over axes, get subplotspecs and add headers
    for ax in axes:
        sbs = ax.get_subplotspec()

        # put headers on cols
        if (col_headers is not None) and sbs.is_first_row(): # if col_headers is not None, on the first row of the subplotspec
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad), # pad between tick labels and column headers
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs, # pass kwargs to text object to style headers
            )

        # put headers on rows
        if (row_headers is not None) and sbs.is_first_col(): # if row_headers is not None, on the first column of the subplotspec
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0), # pad between tick labels and row headers
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                **text_kwargs, # pass kwargs to text object to style headers
            )

def plot_histories(history1:dict, history2:dict, row_headers:list, savepath:pathlib.Path, filename:str):
    '''
    Plots the loss and accuracy histories of two models side by side.

    Args: 
        - history1: history object of first model
        - history2: history object of second model
        - row_headers: list containing strings of row header 1 and 2 e.g., ["Row 1", "Row 2"]
        - savepath: path to save figure
        - filename: filename of figure

    Output:
        - .pngªª saved to savepath/filename
    '''

    # get number of epochs
    epochs_history1 = len(history1["loss"]) + 1
    epochs_history2 = len(history2["loss"]) + 1 

    # define theme 
    plt.style.use("seaborn-colorblind")

    # define figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(12,6))

    # loss history 1
    axes[0, 0].plot(np.arange(1, epochs_history1), history1["loss"], label="Train") 
    axes[0, 0].plot(np.arange(1, epochs_history1), history1["val_loss"], label="Val", linestyle=":") 

    # accuracy history 1
    axes[0, 1].plot(np.arange(1, epochs_history1), history1["accuracy"], label="Train") 
    axes[0, 1].plot(np.arange(1, epochs_history1), history1["val_accuracy"], label="Val", linestyle=":") 
    
    # loss history 2
    axes[1, 0].plot(np.arange(1, epochs_history2), history2["loss"], label="Train") 
    axes[1, 0].plot(np.arange(1, epochs_history2), history2["val_loss"], label="Val", linestyle=":") 
    
    # accuracy history 2
    axes[1, 1].plot(np.arange(1, epochs_history2), history2["accuracy"], label="Train") 
    axes[1, 1].plot(np.arange(1, epochs_history2), history2["val_accuracy"], label="Val", linestyle=":") 
    
    # set x labels
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 1].set_xlabel("Epochs")

    # add legend to the left upper corner
    axes[0, 1].legend(bbox_to_anchor=(0.9, 1.3), loc="upper left", prop = {"size":14, "weight":"bold"})

    # add column and row headers
    col_headers = ["Loss", "Accuracy"]
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

    # save fig 
    fig.savefig(savepath / filename, dpi=300)

def create_data_from_metrics_txt(filepath:pathlib.Path):
    '''
    Create a dataframe from a text file containing the classification report from sklearn.metrics.classification_report

    Args:
        - filepath: path to text file

    Returns: 
        - data: dataframe containing the classification report 
    '''

    data = pd.read_csv(filepath)

    # replace macro avg and weighted avg with macro_avg and weighted_avg
    data.iloc[:,0]= data.iloc[:,0].str.replace(r'(macro|weighted)\savg', r'\1_avg', regex=True)

    # split the columns by whitespace
    data = data.iloc[:,0].str.split(expand=True)

    # define new column names 
    new_cols = ['class', 'precision', 'recall', 'f1-score', 'support']
    data.columns = new_cols

    # identify the row with the accuracy score 
    is_accuracy = data['class'] == 'accuracy'

    # move the accuracy row values into the precision and recall columns (they are placed incorrectly when the columns are split)
    data.loc[is_accuracy, ['f1-score', 'support']] = data.loc[is_accuracy, ['precision', 'recall']].values

    # set precision and recall to None for the accuracy row
    data.loc[is_accuracy, ['precision', 'recall']] = None

    return data

def create_metrics_dataframes(resultspath:pathlib.Path):
    '''
    Loads all history objects from a given path and returns them in a dictionary.

    Args:
        - resultspath: path to directory containing txt files with metrics from scikit-learn's classification report
    
    Returns: 
        - metrics_dfs: dictionary containing all txt files in path 
    '''

    # empty dictionary where dataframes will be saved
    metrics_dfs = {}

    # empty list for number of epochs from filename
    epochs = []

    for file in resultspath.iterdir(): 
        if "metrics" in file.name: # only work on all files which have "metrics in their name"
            # create dataframe from txt file 
            metrics_data = create_data_from_metrics_txt(resultspath/file.name)
            # define metrics name with regex (e.g., "REAL_LeNet_18e.txt" -> "REAL_LeNet")
            metrics_name = re.sub("_metrics_\d+e.txt", "", file.name)
            # add to metrics_dfs dict ! 
            metrics_dfs[metrics_name] = metrics_data
        
            # get number of epochs from file name 
            for epoch in re.findall(r'\d+(?=e\.txt)', file.name):
                epochs.append(epoch)

    return metrics_dfs, epochs

def create_table(data:dict, epochs:list, header_labels:list, metric:str="f1-score"): 
    # Capitalize header_labels
    header_labels = [header.title() for header in header_labels]

    # define empty list for nicely formatted table data
    tabledata = []

    for key, value in data.items():
        # create name 
        modelname = re.sub("_", " ", key)

        # create table row with model name and chosen metric
        tablerow = [modelname] + [str(value) for value in value[metric]] 

        # append epoch value to tablerow for each model
        tablerow.append(str(epochs.pop(0)))

        # append tablerrow to tabledata
        tabledata.append(tablerow)

    # create table 
    table = tabulate(tabledata,
        headers = header_labels, 
        tablefmt="github"
    )

    return table