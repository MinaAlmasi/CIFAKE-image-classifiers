'''
Script for self-assigned Assignment 4, Visual Analytics, Cultural Data Science, F2023

Evaluating the best FAKE classifiers on the REAL test data.

Run script in the terminal by typing: 
    python src/visualise.py

@MinaAlmasi
'''

from tensorflow import keras

# utils
import pathlib 

# image import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# custom modules for loading data, metrics, logging
from modules.load_data import load_metadata, load_tf_data
from modules.classify_pipeline import clf_get_metrics
from modules.utils import custom_logger
from modules.visualisation import create_metrics_dataframes, create_table

def evaluate(modelpath, modelname, test_data):
    # load model 
    model = keras.models.load_model(modelpath / modelname)

    # test 
    metrics = clf_get_metrics(model, test_data)

    return metrics 

def main(): 
    # intialise logger
    logging = custom_logger("evaluate_FAKE_models")

    # define paths 
    path = pathlib.Path(__file__)

    # loading the REAL data to be the test data for the fake models 
    metadatapath = path.parents[1] / "images" / "metadata" / "REAL" 

    # resultspath 
    resultspath = path.parents[1] / "E2_results"
    resultspath.mkdir(exist_ok=True, parents=True)

    # save path for visualisations
    resultspath = path.parents[1] / "E2_visualisations"
    resultspath.mkdir(exist_ok=True, parents=True)

    # model paths 
    LeNet_modelpath = path.parents[1] / "models" / "LeNet_model"
    VGG16_modelpath = path.parents[1] / "models" / "VGG16_model"

    # load metadata 
    logging.info(f"Loading metadata REAL ...")
    meta_dict = load_metadata(metadatapath)

    # intialise datagenerator
    logging.info(f"Loading data REAL ...")
    datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, dtype="float32") 

    # load test data
    test = load_tf_data(datagen, meta_dict["test"], "rgb", "filepath", "label", (32, 32), 64, shuffle=False)

    # load models
    FAKE_LeNet = keras.models.load_model(LeNet_modelpath / "FAKE_LeNet_model_11e.h5")
    FAKE_VGG16 = keras.models.load_model(VGG16_modelpath / "FAKE_VGG16_model_13e.h5")

    ## get metrics 
    metrics_LeNet = clf_get_metrics(FAKE_LeNet, test)
    metrics_VGG16 = clf_get_metrics(FAKE_VGG16, test)

    # save metrics
    for model, metrics in {"LeNet_11e": metrics_LeNet, "VGG16_18e": metrics_VGG16}.items(): 
        file_name = model.split("_")
        with open(resultspath / f"FAKE_{file_name[0]}_metrics_{file_name[1]}.txt", "w") as file: 
            file.write(metrics)
    
    # make metrics into dataframes
    metrics_data, epochs = create_metrics_dataframes(resultspath)

    # define epochs 
    header_labels = metrics_data["FAKE_VGG16"]["class"].tolist() + ["epochs"]

    # turn dataframes into table 
    metrics_table = create_table(metrics_data, epochs, header_labels)

    print(metrics_table)

    # save metrics_table
    with open(resultspath /"FAKE_metrics_table.txt", 'w') as file:
        file.write(metrics_table)

if __name__ == "__main__":
    main()