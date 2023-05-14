The src folder contains the following scripts that can be run by typing ```python src/XX.py``` with the ```env``` activated:

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```classify_XX.py``` | Scripts to preprocess data and train the ```FAKE```and ```REAL``` classifiers, one for each model architecture.|
| ```create_metadata.py```  | Script to create metadata for the dataset (e.g., to extract class labels from filenames)|
| ```visualise.py```  | Script to visualise the results from model training and evaluation done in ```classify_XX.py``` scripts (E1)|
| ```final_evaluate.py```  | Script to evaluate ```FAKE``` classifiers on ```REAL``` testdata (E2)|
 

The above mentioned scripts rely on functions defined in the ```modules``` folder.