'''
Utils script for Assignment 4, Visual Analytics, Cultural Data Science, F2023

The following script contains: 
    - a function that customises a logging logger to display messages in the terminal
    - a function that to parse arguments to the command line 

@MinaAlmasi
'''

import logging, argparse

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-data", "--data_label", help = "'FAKE' or 'REAL' to indicate which dataset you want to run the model training on", type = str, default="FAKE")
    parser.add_argument("-epochs", "--n_epochs", help = "number of epochs the model is run for", type = int, default=10)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def custom_logger(name):
    '''
    Custom logger for displaying messages to console while running scripts.

    Args: 
        - name: name of logger
    Returns: 
        - logger object to be used in functions and scripts

    '''

    # define loggger
    logger = logging.getLogger(name)

    # set level of logging (level of detail of what should be logged)
    logger.setLevel(level=logging.INFO)

    # instantiate console logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler) # add handler to overall logger

    # define formatting of logging
    formatter = logging.Formatter('%(asctime)s | %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    console_handler.setFormatter(formatter) # add formatting to console handler

    return logger