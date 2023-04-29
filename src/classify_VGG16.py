
import pathlib
import argparse

# image import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input # prepare image for VGG16

# VGG16
from tensorflow.keras.applications.vgg16 import VGG16 # vgg16
from tensorflow.keras.layers import Dense, Flatten # layers
from tensorflow.keras.models import Model # generic model object 

# custom modules
from load_data import load_metadata, load_tf_data
from classify_pipeline import clf_pipeline

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-data", "--data_label", help = "'FAKE' or 'REAL' to indicate which dataset you want to run the model training on", type = str, default="FAKE")
    parser.add_argument("-epochs", "--n_epochs", help = "number of epochs the model is run for", type = int, default=3)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

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

    # define paths 
    path = pathlib.Path(__file__)
    metadatapath = path.parents[1] / "images" / "metadata" / args.data_label # args.data_label is either FAKE or REAL (indicating which dataset to work on)
    resultspath = path.parents[1] / "results"
    modelpath = path.parents[1] / "models" / "NN_model"

    # load metadata 
    meta_dict = load_metadata(metadatapath)

    # intialise datagenerator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2, dtype="float32") 

    # training data 
    train = load_tf_data(datagen, meta_dict["train"], "rgb", "filepath", "label", (32, 32), 64, shuffle=True, subset="training")

    # load val data 
    val = load_tf_data(datagen, meta_dict["train"], "rgb", "filepath", "label", (32, 32), 64, shuffle=True, subset="validation")

    # load test data
    test = load_tf_data(datagen, meta_dict["test"], "rgb", "filepath", "label", (32, 32), 64, shuffle=False)

    # intialize model
    print("[INFO]: Intializing model")
    model = vgg16()

    # train pipeline 
    clf_pipeline(
        model = model, 
        train_data = train, 
        val_data = val, 
        test_data = test, 
        epochs = args.n_epochs,
        model_name = "NN_model",
        modelpath = modelpath,
        resultspath = resultspath
    )

if __name__ == "__main__":
    main()