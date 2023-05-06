import pickle
import pathlib 
import pandas as pd 

# to import model 
import tensorflow as tf

import sys
sys.path.append(str(pathlib.Path(__file__).parents[1]))
# image import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from load_data import load_metadata, load_tf_data
from classify_pipeline import clf_get_metrics

def main2(): 
    pass

def main(): 
    path = pathlib.Path(__file__)
    testpath = path.parents[1] / "test"
    modelpath = path.parents[1] / "models" / "LeNet_model"
    metadatapath = path.parents[1] / "images" / "metadata" / "FAKE" 

    model = tf.keras.models.load_model(modelpath / "FAKE_LeNet_model_11e.h5")


     # load metadata 
    meta_dict = load_metadata(metadatapath)

    # intialise datagenerator
    datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, dtype="float32") 

    # load test data
    test = load_tf_data(datagen, meta_dict["test"], "rgb", "filepath", "label", (32, 32), 64, shuffle=False)

    metrics = clf_get_metrics(model, test)

    metrics_data = pd.DataFrame.from_records(metrics)

    print(metrics_data)
    metrics_data.to_csv("FAKE_LeNet_metrics_11e.csv")

    
    

if __name__ == "__main__":
    main()