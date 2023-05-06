'''
Script for self-assigned Assignment 4, Visual Analytics, Cultural Data Science, F2023

Visualise the training histories of two models side by side, thereby creating three plots, one for each model (NN, LeNet, VGG16).

@MinaAlmasi
'''

# utils
import pickle
import pathlib
import re 

# plotting 
from matplotlib import pyplot as plt
import numpy as np

def load_model_histories(path):
    '''
    Loads all history objects from a given path and returns them in a dictionary.

    Args:
        - path: path to directory containing history objects
    
    Returns: 
        - history_objects: dictionary containing all history objects in path 
    '''

    history_objects = {}

    for file in path.iterdir():
        if "history" in file.name:
            with open(path / file.name , 'rb') as f:
                history = pickle.load(f)
                history_name = re.sub("_history.pickle", "", file.name) 
                history_objects[history_name] = history
    
    return history_objects 

# add_headers() taken from https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots 
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

def plot_histories(history1:dict, history2:dict, row_headers:list, savepath:str, filename:str):
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
    axes[0, 0].plot(np.arange(1, epochs_history1), history1["loss"], label="train_loss") # plot train loss 
    axes[0, 0].plot(np.arange(1, epochs_history1), history1["val_loss"], label="train_loss") # plot train loss 

    # accuracy history 1
    axes[0, 1].plot(np.arange(1, epochs_history1), history1["accuracy"], label="train_loss") # plot train loss 
    axes[0, 1].plot(np.arange(1, epochs_history1), history1["val_accuracy"], label="train_loss") # plot train loss 
    
    # loss history 2
    axes[1, 0].plot(np.arange(1, epochs_history2), history2["loss"], label="train_loss") # plot train loss
    axes[1, 0].plot(np.arange(1, epochs_history2), history2["val_loss"], label="train_loss") # plot train loss
    
    # accuracy history 2
    axes[1, 1].plot(np.arange(1, epochs_history2), history2["accuracy"], label="train_loss") # plot train loss 
    axes[1, 1].plot(np.arange(1, epochs_history2), history2["val_accuracy"], label="train_loss") # plot train loss 
    
    # set x labels
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 1].set_xlabel("Epochs")

    # add column and row headers
    col_headers = ["Loss", "Accuracy"]
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

    # save fig 
    fig.savefig(savepath / filename, dpi=300)

def main(): 
    path = pathlib.Path(__file__)

    resultspath = path.parents[1] / "results"
    savepath = path.parents[1] / "test"
    savepath.mkdir(exist_ok=True)

    # create history plots for real and fake data
    history_objects = load_model_histories(resultspath)
    for model in ["NN", "LeNet", "VGG16"]:
        plot_histories(history_objects[f"REAL_{model}"], history_objects[f"FAKE_{model}"], [f"REAL {model}", f"FAKE {model}"], savepath, f"{model}_histories.png")




if __name__ == "__main__":
    main()