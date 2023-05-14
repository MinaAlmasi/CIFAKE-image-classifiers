'''
Script for self-assigned Assignment 4, Visual Analytics, Cultural Data Science, F2023

Create metadata for CIFAKE data containing columns: 'filename', 'class' (int e.g., "1"), label (string e.g., "airplane"), 'filepath'

Run the script in the terminal by typing: 
    python src/create_metadata.py

@MinaAlmasi
'''

# utils 
import pathlib 

# data wrangling 
import pandas as pd

def get_filenames(subdirectory:pathlib.Path): 
    '''
    Get file names in subdirectory. 

    Args: 
        - subdirectory: subdirectory file names are located 

    Returns: 
        - filenames: list of file names in subdirectory 
    '''

    # if subdirectory is a directory and file is a regular file, get file name (file.name)
    filenames = [file.name for file in subdirectory.iterdir() if file.is_file()]

    filenames = sorted(filenames) # sort filenames alphanumerically

    return filenames

def filenames_from_subdirectories(path:pathlib.Path):
    '''
    Get file names from several subdirectories in specified path. 
    Divided into a dictionary.

    Args: 
        - path: directory 

    Returns: 
        - filenames: dictionary with subdirectory names (keys) and file names (values)
    '''

    filenames = {}

    for path in path.iterdir():
        if path.is_dir():
            # append filenames for each subdirectory with subdirectory name as key 
            filenames[path.name] = get_filenames(path)

    return filenames 

def create_dataframe(filenames:list, class_labels:dict, datapath:pathlib.Path, folderlabel:str):
    '''
    Create dataframe from filenames in CIFAKE dataset. 
    
    The function extracts the number in the parenthesis of the file names (e.g., '0 (2).jpg') to create a seperate class column. 
    File names without any number (e.g., '0.jpg' refer to class 1). 

    Args: 
        - filenames: list of files 
        - class_labels: dictionary with class names in the format {int:label} e.g., {1:"airplane", 2:"automobile", 3:"bird"}
        - datapath: path where the files are located (filenames) e.g., .. / "images" / "test"
        - folderlabel: name of the subdirectory (e.g., "FAKE")

    Returns: 
        - dataframe containing columns: 'filename', 'class' (int e.g., "1"), label (string e.g., "airplane"), 'filepath'
    '''

    # create dataframe with filenames from label name
    df = pd.DataFrame(filenames, columns=["filename"])

    # pattern to extract (e.g., '0 (2).jpg' -> '2') 
    pattern = "\((\d+)\)"

    # extract label from filename
    df["class"] = df["filename"].str.extract(pattern)
    df["class"] = df["class"].fillna(1)

    # add class label
    df["label"] = df["class"].astype(int).map(class_labels)

    # subset only last two elements of pathlb path
    dataparts = list(datapath.parts[-2:])
    df["filepath"] = pathlib.Path(*dataparts) / folderlabel / df["filename"] # unpack tuple of dataparts, add filename

    return df 

def save_metadata(filenames:dict, class_labels:dict, datapath:pathlib.Path, metadatapath:pathlib.Path):
    '''
    Save metadata for CIFAKE dataset as CSV files. 

    Args: 
        - filenames: list of files 
        - class_labels: dictionary with class names in the format {int:label} e.g., {1:"airplane", 2:"automobile", 3:"bird"}
        - datapath: path where the files are located (filenames) e.g., .. / "images" / "test"
        - metadatapath: where the metadata files should be saved. 

    Outputs: 
        - dataframes as .CSV containing columns: 'filename', 'class' (int e.g., "1"), label (string e.g., "airplane"), 'filepath'
    '''

    # iterate over filenames dictionary (e.g., test["FAKE"], test["REAL"]), create dataframes for each seperately!
    for folderlabel, folderfilenames in filenames.items():
        # create dataframe
        temp_df = create_dataframe(folderfilenames, class_labels, datapath, folderlabel)

        # define path 
        filepath = metadatapath / folderlabel
        filepath.mkdir(exist_ok=True, parents=True)

        # save dataframe to CSV 
        temp_df.to_csv(filepath/ f"{datapath.name}.csv", index=False)


def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    trainpath = path.parents[1] / "images" / "train"
    testpath = path.parents[1] / "images" / "test"

    metadatapath = path.parents[1] / "images" / "metadata"

    # define label names in reverse order explicitly
    labels = {1:"airplane", 2:"automobile", 3:"bird", 4:"cat", 5:"deer", 6:"dog", 7:"frog", 8:"horse", 9:"ship", 10:"truck"}

    # get filenames
    train_filenames = filenames_from_subdirectories(trainpath)
    test_filenames = filenames_from_subdirectories(testpath)

    # create metadata for both train and test 
    save_metadata(train_filenames, labels, trainpath, metadatapath)
    save_metadata(test_filenames, labels, testpath, metadatapath)

if __name__ == "__main__":
    main()