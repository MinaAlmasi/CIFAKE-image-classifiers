
import pathlib
import argparse

# image import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# simple neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# custom modules
from modules.data_fns import load_metadata, load_tf_data
from modules.classify_fns import optimise_model, train_model
from modules.evaluate_fns import save_model_card, plot_model_history, get_model_metrics, save_model_metrics

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-data", "--data_label", help = "'FAKE' or 'REAL' to indicate which dataset you want to run the model training on", type = int, default="FAKE")
    parser.add_argument("-epochs", "--n_epochs", help = "number of epochs the model is run for", type = int, default=3)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def simple_neural_network(input_shape:tuple=(32, 32, 3), output_layer_size:int=10):
    '''
    Instantiate simple neural network.

    Args: 
        - input_shape: tuple with image size and number of channels. Defaults to (32, 32, 3) i.e., images with 32x32 pixels and 3 color channels. 
        - output_layer_size: size of output layer. Should correspond to the amount of classes to be predicted. 

    Return: 
        - model: simple neural netowrk 

    '''

    # intialize model 
    model = Sequential()

    # add a flatten layer
    model.add(Flatten())

    # add hidden layer
    model.add(Dense(128, input_shape = input_shape, activation="relu"))

    # add output layer 
    model.add(Dense(output_layer_size, activation="softmax"))

    return model 

def main():
    # define args
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)
    metadatapath = path.parents[1] / "images" / "metadata" / args.data_label # args.data_label is either FAKE or REAL (indicating which dataset to work on)

    # load metadata 
    meta_dict = load_metadata(metadatapath)

    # intialise datagenerator
    datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, dtype="float32") 

    # training data 
    train = load_tf_data(datagen, meta_dict["train"], "grayscale", "filepath", "label", (32, 32), 64, shuffle=True, subset="training")

    # load val data 
    val = load_tf_data(datagen, meta_dict["train"], "grayscale", "filepath", "label", (32, 32), 64, shuffle=True, subset="validation")

    # load test data
    test = load_tf_data(datagen, meta_dict["test"], "grayscale", "filepath", "label", (32, 32), 64, shuffle=False)

    # intialize model
    print("[INFO]: Intializing model")
    model = simple_neural_network()
    save_model_card(model, n_epochs, outpath, "model_card.txt") # save model card 

    # optimise model
    model = optimise_model(model)

    # define epochs 
    n_epochs = args.n_epochs

    # train model
    print("[INFO]: Training model")
    model_history = train_model(model, train, val, n_epochs)

    # save model 
    modelpath = path.parents[1] / "models" / "NN_model" # define folder
    modelpath.mkdir(exist_ok=True, parents=True) # make if it does not exist

    # save model 
    model.save(modelpath / f"NN_model_{n_epochs}_epochs.h5")

    # save plot history (training and validation loss)
    plot_model_history(model_history, n_epochs, outpath, f"history_{n_epochs}_epochs.png")

    # evaluate model 
    print("[INFO]: Evaluating model")
    metrics = get_model_metrics(model, test_data)

    # save metrics
    print("[INFO]: Saving model")
    save_model_metrics(metrics, outpath, f"metrics_{n_epochs}_epochs.txt")

if __name__ == "__main__":
    main()