'''
Script for self-assigned Assignment 4, Visual Analytics, Cultural Data Science, F2023

Script to create plots and table displayed in README. Concretely, the script creates:
1. Visualise the training histories of two models side by side, thereby creating three plots, one for each model (NN, LeNet, VGG16).
2. Create dataframes from metrics .txt files and from those dataframes creates an overview table displaying all F1 scores 

The script relies on functions defined in /modules/visualisation.py 

Run script in the terminal by typing: 
    python src/visualise.py

@MinaAlmasi
'''

# utils 
import pathlib 

# custom module for plotting and creating table
from modules.visualisation import plot_histories, create_metrics_dataframes, create_table

def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    resultspath = path.parents[1] / "E1_results" # get raw results 

    savepath = path.parents[1] / "E1_visualisations" # define path for visualisations going in the readme 
    savepath.mkdir(exist_ok=True)

    # create history plots for real and fake data
    history_objects = load_model_histories(resultspath)
    for model in ["NN", "LeNet", "VGG16"]:
        plot_histories(history_objects[f"REAL_{model}"], history_objects[f"FAKE_{model}"], [f"REAL {model}", f"FAKE {model}"], savepath, f"{model}_histories.png")

    # get dataframes
    metrics_data, epochs = create_metrics_dataframes(resultspath)

    # get header for table   
    header_labels = metrics_data["REAL_VGG16"]["class"].tolist() + ["epochs"]

    # turn dataframes into table 
    metrics_table = create_table(metrics_data, header_labels, epochs)

    # save metrics_table
    with open(savepath/"all_metrics_table.txt", 'w') as file:
        file.write(metrics_table)
    
if __name__ == "__main__":
    main()