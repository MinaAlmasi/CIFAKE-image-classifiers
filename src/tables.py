# utils 
import pathlib 
import re 

# data wrangling 
import pandas as pd 

# make table
from tabulate import tabulate

def create_data_from_metrics_txt(filepath):
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

def create_metrics_dataframes(resultspath):
    '''
    Loads all history objects from a given path and returns them in a dictionary.

    Args:
        - resultspath: path to directory containing txt files with metrics from scikit-learn's classification report
    
    Returns: 
        - metrics_dfs: dictionary containing all txt files in path 
    '''

    # define empty dictionary where dataframes will be saved
    metrics_dfs = {}

    for file in resultspath.iterdir(): 
        if "metrics" in file.name: # only work on all files which have "metrics in their name"
            # create dataframe from txt file 
            metrics_data = create_data_from_metrics_txt(resultspath/file.name)
            # define metrics name with regex (e.g., "REAL_LeNet_18e.txt" -> "REAL_LeNet")
            metrics_name = re.sub("_metrics_\d+e.txt", "", file.name)
            # add to metrics_dfs dict ! 
            metrics_dfs[metrics_name] = metrics_data

    return metrics_dfs

def create_table(data_dict): 
    # get header for table   
    header_labels = data_dict["REAL_VGG16"]["class"].tolist()

    print(header_labels)

    table = tabulate(
        [["REAL NN"] + data_dict["REAL_NN"]["f1-score"].tolist(),   
        ["FAKE NN"] + data_dict["FAKE_NN"]["f1-score"].tolist(),   
        ["REAL LeNet"] + data_dict["REAL_LeNet"]["f1-score"].tolist(),   
        ["Fake LeNet"] + data_dict["FAKE_LeNet"]["f1-score"].tolist(),   
        ["REAL VGG16"] + data_dict["REAL_VGG16"]["f1-score"].tolist()], 
        headers = header_labels, 
        tablefmt="github"
    )

    return table



def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    resultspath = path.parents[1] / "results" # get raw results 

    savepath = path.parents[1] / "visualisations" # define path for visualisations going in the readme 
    savepath.mkdir(exist_ok=True)

    # get dataframes
    metrics_data = create_metrics_dataframes(resultspath)

    print(create_table(metrics_data))

if __name__ == "__main__":
    main()