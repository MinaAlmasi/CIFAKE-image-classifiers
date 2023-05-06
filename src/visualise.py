import pickle
import pathlib 
import pandas as pd 
import re 
from matplotlib import pyplot as plt
import numpy as np

def load_model_histories(path):
    history_objects = {}

    for file in path.iterdir():
        if "history" in file.name:
            with open(path / file.name , 'rb') as f:
                history = pickle.load(f)
                history_name = re.sub("_history.pickle", "", file.name) 
                history_objects[history_name] = history
    
    return history_objects 

def add_headers( # taken from https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots (adapted slightly)
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers,
                **text_kwargs,
            )

def plot_histories(history1, history2, row_headers, savepath, filename):
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

    history_objects = load_model_histories(resultspath)

    for model in ["NN", "LeNet", "VGG16"]:
        plot_histories(history_objects[f"REAL_{model}"], history_objects[f"FAKE_{model}"], [f"REAL {model}", f"FAKE {model}"], savepath, f"{model}_histories.png")


if __name__ == "__main__":
    main()