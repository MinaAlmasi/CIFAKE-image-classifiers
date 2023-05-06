import pickle
import pathlib 
import pandas as pd 

def load_model_histories(path):
    for file in path.iterdir():
        print(file.name)

def main(): 
    path = pathlib.Path(__file__)

    resultspath = path.parents[1] / "results"

    with open(resultspath / "REAL_NN_history.pickle" , 'rb') as f:
        history = pickle.load(f)

    with open(resultspath/"REAL_NN_metrics_20e.txt") as f:
        contents = f.readlines()

    print(contents)

    column_names=["Class","Precision","Recall", "F1-score", "Support"]
    data = create_dataframe_from_list(contents)
    print(data)
    
    #df = pd.read_csv(resultspath/"REAL_NN_metrics_20e.txt", header=1)
    #print(df.columns)
    #print(df)

if __name__ == "__main__":
    main()