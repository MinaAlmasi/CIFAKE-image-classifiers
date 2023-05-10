'''
Script for self-assigned Assignment 4, Visual Analytics, Cultural Data Science, F2023

Evaluating the best FAKE classifiers on the REAL test data

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
    resultspath = path.parents[1] / "final_evaluate_results"

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
    FAKE_VGG16 = keras.models.load_model(LeNet_modelpath / "FAKE_LeNet_model_11e.h5")
    FAKE_LeNet = keras.models.load_model(LeNet_modelpath / "FAKE_VGG16_model_11e.h5")

    ## get metrics 
    metrics_VGG16 = clf_get_metrics(FAKE_VGG16, test)
    metrics_LeNet = clf_get_metrics(FAKE_LeNet, test)

    # save metrics
    for model, metrics in {VGG16: metrics_VGG16, LeNet: metrics_LeNet}.items(): 
        with open(resultspath / f"FAKE_{model}_metrics.txt", "w") as file: 
            file.write(metrics)
    
    # make metrics into dataframes
    


if __name__ == "__main__":
    main()