'''
Script for self-assigned Assignment 4, Visual Analytics, Cultural Data Science, F2023

Classify CIFAKE dataset using a CNN with the LeNet architecture

Run in the terminal by typing: 
    python src/classify_LeNet.py -data {DATA_LABEL} -epochs {NUMBER_OF_EPOCHS}

Where:
-data refers to which dataset (FAKE or REAL) the model should train on
-epochs refers to the maximum amount of epochs that the model should train for 

@MinaAlmasi
'''

# utils
import pathlib

# image import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# VGG16
from tensorflow.keras.models import Sequential # model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #layers 

# custom modules
from modules.load_data import load_metadata, load_tf_data
from modules.classify_pipeline import clf_pipeline
from modules.utils import custom_logger, input_parse

def cnn_lenet(input_shape:tuple=(32, 32, 3), output_layer_size:int=10):  # adapted from class notebook
    '''
    Instantiate convolutional neural network with LeNet architecture 

    Args: 
        - input_shape: tuple with image size and number of channels. Defaults to (32, 32, 3) i.e., images with 32x32 pixels and 3 color channels. 
        - output_layer_size: size of output layer. Should correspond to the amount of classes to be predicted. 

    Return: 
        - model: convolutional neural network with LeNet architecture 
    '''

    # define model
    model = Sequential()

    # first set of layers CONV => RELU => MAXPOOL
    model.add(Conv2D(32, (3,3),
                    padding="same",
                    input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2),
                        strides = (2,2)))

    # second set of layers CONV => RELU => MAXPOOL
    model.add(Conv2D(50, (5,5),
                    padding="same", activation="relu"))

    # add max pooling 
    model.add(MaxPooling2D(pool_size = (2,2),
                        strides = (2,2))) # stride of 2 left-right, and stride of 2 down 

    # FC => RELU
    model.add(Flatten()) # add flatten layer 
    model.add(Dense(128, activation="relu"))

    # softmax classifier
    model.add(Dense(output_layer_size, activation="softmax"))

    return model

def main():
    # define args
    args = input_parse()

    # intialize logger
    logging = custom_logger("LeNet_classifier")

    # define paths 
    path = pathlib.Path(__file__)
    metadatapath = path.parents[1] / "images" / "metadata" / args.data_label # args.data_label is either FAKE or REAL (indicating which dataset to work on)
    resultspath = path.parents[1] / "E1_results"
    modelpath = path.parents[1] / "models" / "LeNet_model"

    # load metadata 
    logging.info(f"Loading metadata {args.data_label} ...")
    meta_dict = load_metadata(metadatapath)

    # intialise datagenerator
    logging.info(f"Loading data {args.data_label} ...")
    datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, dtype="float32") 

    # training data 
    train = load_tf_data(datagen, meta_dict["train"], "rgb", "filepath", "label", (32, 32), 64, shuffle=True, subset="training")

    # load val data 
    val = load_tf_data(datagen, meta_dict["train"], "rgb", "filepath", "label", (32, 32), 64, shuffle=True, subset="validation")

    # load test data
    test = load_tf_data(datagen, meta_dict["test"], "rgb", "filepath", "label", (32, 32), 64, shuffle=False)

    # intialize model
    logging.info("Intializing model ...")
    model = cnn_lenet()

    # train, evaluate pipeline 
    clf_pipeline(
        model = model, 
        train_data = train, 
        val_data = val, 
        test_data = test, 
        epochs = args.n_epochs,
        early_stop_epochs=2,
        model_name = f"{args.data_label}_LeNet",
        modelpath = modelpath,
        resultspath = resultspath
    )

if __name__ == "__main__":
    main()