'''Load and preprocess data'''

# data wrangling 
import pandas as pd
import re 

# import image daraser 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_metadata(metadatapath:str): # function adapted from previous assignment
    '''
    Loads metadata stored as CSV files in "metadatapath" as pandas dataframes in a dictionary. 

    Args: 
        - metadatapath: path where the JSON files are stored. 
            NB! The JSON files are required to have the following naming convention: "X_data.json" (e.g train_data.json) 

    Returns: 
        metadata: dictionary with each json file being a seperate pandas dataframe. 
            Follows the naming convention of the JSON files, but removing the _data.json extension (e.g., metadata["train"]).
    '''

    metadata = {}

    # iterate over paths in metadatapath 
    for file in metadatapath.iterdir(): 
        
        # from file.name (e.g., "test_data.json"), rm ".json"
        file_name = re.sub(".csv", "", file.name) 

        # from file (filepath) read as pandas dataframe, call it the file_name and append to dfs dict (e.g., metadata["train"])
        metadata[file_name] = pd.read_csv(file) 

    return metadata


def load_tf_data(datagen, metadata, imagepath_col:str, label_col:str, image_size:tuple, batch_size:int, shuffle:bool, subset:str=None):
    '''
    Loads and preprocess data using Tensorflow's ImageDataGenerator and its method .flow_from_dataframe(). ImageDataGenerator has to be instantiated already. 

    Args: 
        - datagen: initalised ImageDataGenerator 
        - metadata: dataframe containing image paths (imagepath_col) and class labels (label_col)

        - imagepath_col: column containing image paths in the metadata
        - label_col: column containing class labels in the metadata
        
        - image_size: size that images should be resized to e.g., (224, 224) for VGG16 
        - batch_size: size of batches that the data should be loaded in.

        - shuffle: whether the data should be shuffled. Ideal for training and validation data (especially necessary if data is sorted by class), but AVOID doing so for test data. 
        - subset: write "training" or "validation" to define which subset is load. Only works if validation_split is defined in ImageDataGenerator. Defaults to None. 

    Returns: 
        - data: data ready to be used with a tensorflow model 
    '''

    # load train
    data = datagen.flow_from_dataframe(
        dataframe = metadata, 
        subset = subset,
        x_col = imagepath_col, 
        y_col = label_col, 
        target_size = image_size, 
        batch_size = batch_size, 
        class_mode = "categorical", 
        shuffle = shuffle, 
        seed = 129
    )

    return data


import pathlib
def main(): 
    path = pathlib.Path(__file__)
    metadatapath = path.parents[1] / "images" / "metadata" / "FAKE"

    meta_dict = load_metadata(metadatapath)

    # intialise datagenerator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

    # training data 
    train = load_tf_data(datagen, meta_dict["train"], "filepath", "label", (32, 32), 64, shuffle=True, subset="training")

    # load val data 
    val = load_tf_data(datagen, meta_dict["train"], "filepath", "label", (32, 32), 64, shuffle=True, subset="validation")

    # load test data
    test = load_tf_data(datagen, meta_dict["test"], "filepath", "label", (32, 32), 64, shuffle=False)

    

if __name__ == "__main__":
    main()