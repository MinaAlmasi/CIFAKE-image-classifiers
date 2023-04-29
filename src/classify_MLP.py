
import pathlib

# image import 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# custom modules
from modules.data_fns import load_metadata, load_tf_data
from modules.classify_fns import simple_neural_network, optimise_model, train_model
from modules.evaluate_fns import save_model_card, plot_model_history, get_model_metrics, save_model_metrics

def main():
    # define paths 
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

    # intialize model
    print("[INFO]: Intializing model")
    model = simple_neural_network()

    # optimise model
    model = optimise_model(model)

    # define epochs 
    n_epochs = 2

    # train model
    print("[INFO]: Training model")
    model_history = train_model(model, train, val, n_epochs)

    # save model 
    modelpath = path.parents[1] / "models"  # define folder
    modelpath.mkdir(exist_ok=True) # make if it does not exist

    # save model 
    model.save(modelpath / f"model_{n_epochs}_epochs.h5")

    # saving model information
    save_model_card(model, n_epochs, outpath, "model_card.txt")

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