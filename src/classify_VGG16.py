'''
Script for self-assigned Assignment 4, Visual Analytics, Cultural Data Science, F2023

Classify CIFAKE dataset using the pre-trained CNN VGG16. 

Run in the terminal by typing: 
    python src/classify_VGG16.py -data 'data_label'

Where the -d which dataset (FAKE or REAL) the model should train on.

@MinaAlmasi
'''

# utils
import pathlib

# image import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input # prepare image for VGG16

# VGG16
from tensorflow.keras.applications.vgg16 import VGG16 # vgg16
from tensorflow.keras.layers import Dense, Flatten # layers
from tensorflow.keras.models import Model # generic model object 

# custom modules
from modules.load_data import load_metadata, load_tf_data
from modules.classify_pipeline import clf_pipeline
from modules.utils import custom_logger, input_parse

def cnn_vgg16(input_shape:tuple=(32, 32, 3), output_layer_size:int=10): 
    '''
    Initialise the pretrained CNN VGG16 without its classifier layers in order to train a new simple neural network with VGG16's weights on a new classification task.

    Args: 
        - input_shape: tuple with image size and number of channels. Defaults to (32, 32, 3) i.e., images with 32x32 pixels and 3 color channels. 
        - output_layer_size: size of output layer. Should correspond to the amount of classes to be predicted. 

    Returns: 
        - VGG16 with newly defined classifier layers. 

    '''

    # intialise model without classifier layers
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=input_shape)

    # disable convolutional layers prior to model training
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(output_layer_size, activation='softmax')(class1) # output_layer_size should correspond to amount of unique classes to predict !  

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    return model

def main():
    # define args
    args = input_parse()

    # initialize logger
    logging = custom_logger("VGG16_classifier")

    # define paths 
    path = pathlib.Path(__file__)
    metadatapath = path.parents[1] / "images" / "metadata" / args.data_label # args.data_label is either FAKE or REAL (indicating which dataset to work on)
    resultspath = path.parents[1] / "results"
    modelpath = path.parents[1] / "models" / "VGG16_model"

    # load metadata 
    logging.info(f"Loading metadata {args.data_label} ...")
    meta_dict = load_metadata(metadatapath)

    # intialise datagenerator
    logging.info(f"Loading data {args.data_label} ...")
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2, dtype="float32") 

    # training data 
    train = load_tf_data(datagen, meta_dict["train"], "rgb", "filepath", "label", (32, 32), 64, shuffle=True, subset="training")

    # load val data 
    val = load_tf_data(datagen, meta_dict["train"], "rgb", "filepath", "label", (32, 32), 64, shuffle=True, subset="validation")

    # load test data
    test = load_tf_data(datagen, meta_dict["test"], "rgb", "filepath", "label", (32, 32), 64, shuffle=False)

    # intialize model
    logging.info("Intializing model")
    model = cnn_vgg16()

    # train pipeline 
    clf_pipeline(
        model = model, 
        train_data = train, 
        val_data = val, 
        test_data = test, 
        epochs = args.n_epochs,
        early_stop_epochs=2,
        model_name = f"{args.data_label}_VGG16",
        modelpath = modelpath,
        resultspath = resultspath
    )

if __name__ == "__main__":
    main()