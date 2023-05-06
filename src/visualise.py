import pickle
import pathlib 
import pandas as pd 
import re 

def load_model_histories(path):
    history_objects = {}

    for file in path.iterdir():
        if "history" in file.name:
            with open(path / file.name , 'rb') as f:
                history = pickle.load(f)
                history_name = re.sub("_history.pickle", "", file.name) 
                history_objects[history_name] = history
    
    return history_objects 

def plot_histories(model_hist1, model_hist2, savepath, filename): # adapted from class notebook
    '''
    Plots four subplots of train/val accuracy and validation for comparing models 
    Saves to .png file

    Args: 
        - model_hist: history of fitted model 
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

def main(): 
    path = pathlib.Path(__file__)

    resultspath = path.parents[1] / "results"

    history_objects = load_model_histories(resultspath)
    print(history_objects.keys())
    
    #df = pd.read_csv(resultspath/"REAL_NN_metrics_20e.txt", header=1)
    #print(df.columns)
    #print(df)

if __name__ == "__main__":
    main()