import pickle
import pathlib 
import pandas as pd 

def create_data_from_metric_txt(path):
    '''
    Create a dataframe from a text file containing the classification report from sklearn.metrics.classification_report

    Args:
        - path: path to text file
    '''

    data = pd.read_csv(path)

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

def main(): 
    # define paths
    path = pathlib.Path(__file__)
    testpath = path.parents[1] / "test"
    resultspath = path.parents[1] / "results"

    filepath = resultspath / "FAKE_LeNet_metrics_11e.txt"

    data = create_data_from_metric_txt(filepath)
    print(data["class"])
    

if __name__ == "__main__":
    main()