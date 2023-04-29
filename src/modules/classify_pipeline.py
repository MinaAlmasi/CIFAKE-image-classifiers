'''
Script for self-assigned (Assignment 4), Visual Analytics, Cultural Data Science, F2023

This script comprises several functions which make up a pipeline for training and evaluating models using tensorflow and keras. 

@MinaAlmasi
'''

# keras layers 
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# functions
def optimise_model(model): 
    '''
    Define a dynamic learning rate and compile the model with it (model optimisation).

    Args: 
        - model: intialised tensorflow model 
    
    Returns: 
        - model: model compiled with new learning rate ! 
    '''

    # define optimisation schedule 
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    
    # insert optimisation schedule in algorithm
    sgd = SGD(learning_rate=lr_schedule)

    # compile model with optimisation algorithm
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_data, validation_data, epochs:int, early_stop_epochs:int=3):    
    '''
    Train initalised CNN for a specified amount of epochs. Evaluate model on validation data. 

    Args: 
        - model: intialised CNN
        - train_data: the data to be trained on 
        - validation_data: the data that the model evaluated on 
        - epochs: number of epochs that the model should train for
        - early_stop_epochs: number of epochs that the model should CONTINUE training for if accuracy does not improve (early stopping parameter).

    Returns: 
        - history: history object containing information about the model training (e.g., loss and accuracy) 
            see documentation for more info: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History 
    '''

    # early stopping
    callback = EarlyStopping(monitor='val_accuracy', patience=early_stop_epochs, restore_best_weights=True, verbose=1)

    # train model
    history = model.fit( # batch_size is not defined in model.fit as documentation specifies that is should not be done when using a data generator (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
        train_data, 
        validation_data = validation_data,
        epochs=epochs, 
        verbose=1, 
        use_multiprocessing=True
        )

    return history

def save_model_card(model, savepath:str, filename): # adapted from prev. assignment
    '''
    Save model card (summary of layers, trainable parameters) as txt file in desired directory (savepath).

    Args: 
        - model: model with defined layers
        - savepath: path where model card should be saved 
        - filename: what the .txt file should be called
    
    Outputs: 
        - .txt file of model summary in "savepath"
    '''

    # define full path
    filepath = savepath / filename
    
    # write model summary as txt
    with open(filepath,'w') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

def save_model_history(model_hist, save_path, filename):
    '''
    Save model history 
    '''
    
    # define 
    filepath = savepath / filename

    with open(filename, 'wb') as file:
        pickle.dump(model_hist.history, file)

def plot_model_history(model_hist, epochs, savepath, filename): # adapted from class notebook
    '''
    Plots two subplots, one for training and validation loss and the other for training and validation accuracy.
    Saves to .png file

    Args: 
        - model_hist: history of fitted model 
        - epochs: how many epochs the model has trained for 
        - savepath: where the .png should be saved
        - filename: name of plot (e.g., "history.png")

    Returns: 
        - .png file of plot in "savepath"
    '''

    # define theme 
    plt.style.use("seaborn-colorblind")

    # define figure size 
    plt.figure(figsize=(12,6))

    # create plot of train and validation loss, defined as two subplots on top of each other ! (but beside the accuracy plot)
    plt.subplot(1,2,1) #nrows, ncols, #index = position
    plt.plot(np.arange(1, epochs+1), model_hist.history["loss"], label="train_loss") # plot train loss 
    plt.plot(np.arange(1, epochs+1), model_hist.history["val_loss"], label="val_loss", linestyle=":") # plot val loss
    
    # text description on plot !!
    plt.title("Loss curve") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    # create a plot of train and validation accuracy, defined as two subplots on top of each other ! (but beside the loss plot)
    plt.subplot(1,2,2) #nrows, ncols, #index = position
    plt.plot(np.arange(0, epochs), model_hist.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), model_hist.history["val_accuracy"], label="val_acc", linestyle=":")

    # text description on plot !! 
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
   
    plt.savefig(savepath / filename, dpi=300)

def get_model_metrics(model, test_data):
    '''
    Evaluates fitted classifier model on test data, returns classification report.

    Args: 
        - model: fitted model to be evaluated
        - test_data: tensorflow test data

    Returns:
        - model_metrics: classification report containing information such as accuracy, F1, precision and recall 
    '''
    # make predictions
    predictions = model.predict(test_data)

    # extract true test vals
    y_test = test_data.classes

    # extract prediction with highest probability 
    y_pred = np.argmax(predictions, axis=1)

    # extract class names 
    class_names = test_data.class_indices.keys()

    # evaluate predictions
    model_metrics = classification_report(y_test, y_pred, target_names = class_names)

    return model_metrics

def save_model_metrics(model_metrics, savepath, filename): # adapted from prev. assignment
    '''
    Converts scikit-learn's classification report (metrics.classification_report) to a .txt file. 

    Args:
        - model_metrics: metrics report (sklearn.metrics.classification_report() or returned from clf_evaluate)
        - filename: filename for .txt report
        - savepath: directory where the text file should be stored. 
    
    Outputs: 
        - .txt file of metrics in "savepath"
    '''

    # define filename 
    filepath = savepath / filename

    # write model metrics to txt
    with open(filepath, "w") as file: 
        file.write(model_metrics)
    

def model_pipeline(train_data, val_data, test_data, epochs, model_name):
    '''
    Train and evaluate instantiated keras model with scikit-learn and tensorflow. 
    '''

    # optimise intialized model 
    model = optimise_model(model)

    # train model 
    print("[INFO]: Training model")
    model_history = train_model(model, train_data, val_data, epochs)

    # save model 
    model.save(model_path / f"model_{epochs}e.h5")

    # save plot history and history object 
    plot_model_history(model_history, n_epochs, outpath, f"{model_name}_history_{epochs}e.png")

    # evaluate model
    print("[INFO]: Evaluating model")
    metrics = get_model_metrics(model, test_data)

    # save metrics 
    print("[INFO]: Classification pipeline complete. Saving model")
    save_model_metrics(metrics, outpath, f"{model_name}_metrics_{n_epochs}e.txt")





