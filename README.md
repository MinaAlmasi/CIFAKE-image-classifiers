# CIFAKE: Comparing classifiers on FAKE vs. REAL image data
This repository forms the self-assigned *assignment 4* in the subject Visual Analytics, Cultural Data Science, F2023. The code is written by Mina Almasi (202005465).

The repository aims to investigate the utility of artificially generated images as an alternative to data augmentation when training classifiers to predict real life images. For this purpose, the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset (Bird & Lofti, 2023) is used. 


## Data 
The ```CIFAKE``` dataset contains 60,000 images that are synthetically generated to be equivalent to the ```CIFAR-10 dataset``` along with 60,000 original CIFAR-10 images (Krizhevsky, 2009). The synthetically generated images were created using the text-to-image [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4). Examples of these artificial images are shown below.

<p align="center">
  <img src="https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/docs/CIFAKE-dataset.png">
</p>

*Figure by Bird & Lofti (2023)*

## Experimental Pipeline and Motivation
The first step when investigating any cultural project with image analysis is to acquire the data needed to answer our questions. However, in problems such as classification which require an abundance of data, this can become problematic if access to data is limited. This is usually approached by **data augmentation** which refers to the creation of new, slightly modified versions of existing data (e.g., by rotating, cropping, flipping the images). With the emergence of generative image models, it is relevant to explore the utility of artificially generated images as an alternative to data augmentation. 

Therefore, this project concretely aims to assess whether the ```CIFAKE``` artificial images can be used to train classifiers that would also perform well on the CIFAR-10 images. 

For this purpose, two experiments are conducted:

###  ```(E1)``` Training Classifiers on ```REAL``` vs ```FAKE``` data
In experiment 1, three classifiers will be trained for each dataset (```FAKE``` and ```REAL``` ) seperately using TensorFlow. These classifiers increase in complexity:

1. Simple Neural Network 
2. CNN with the LeNet Architecture (See also [Wiki/LeNet](https://en.wikipedia.org/wiki/LeNet))
3. Pre-trained VGG-16. 


### ```(E2)``` Testing ```FAKE``` classifiers on ```REAL``` Test Data
In experiment 2, the best performing ```FAKE``` classifier will be evaluated on the ```REAL``` test dataset to see whether its performance transfers across datasets. 


## Reproducibility 
To reproduce the results, follow the instructions in the [Pipeline](https://github.com/MinaAlmasi/CIFAKE-image-classifiers#pipeline) section. 

NB! Be aware that training the model is computationally heavy. Cloud computing (e.g., [UCloud]([UCloud](https://cloud.sdu.dk/))) with high amounts of ram (or a good GPU) is encouraged.

## Project Structure
The repository is structured as such: 
| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```E1_results``` | Results from experiment 1 (E1): model histories, individual loss/accuracy curves, evaluation metrics |
| ```E1_visualisations``` | Visualisations made on results from experiment 1 (E1).|
| ```E2_results``` | Results from experiment 2 (E2): evaluation metrics of the two fake classifiers on the ```REAL``` test data|
| ```E2_visualisations``` | Visualisations made on results from from experiment 2 (E2).|
| ```src```  | Scripts for creating metadata for dataset, running classifications, creating visualisations and doing the final evaluation (E2).|
| ```requirements.txt``` | Necessary packages to be pip installed|
| ```setup.sh``` | Run to install ```requirements.txt``` within newly created ```env``` |
| ```run.sh``` | Run to reproduce entire pipeline including creating metadata, running classifications, evaluating classifiers, making visualisations.|


## Pipeline 
### Setup
Prior to running the pipeline, please firstly install the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset from Kaggle. 

Secondly, create a virtual environment (```env```) and install necessary requirements by running: 
```
bash setup.sh
```
### Running Experimental Pipeline
To run the entire experimental pipeline, type the following in the terminal:
```
bash run.sh
```


## Results 
The results are shown below. Please note that the prefix ```FAKE``` or ```REAL``` of the model refers to whether the model has been trained on the ```FAKE``` or ```REAL``` dataset. 

### (```E1```) Loss and Accuracy Curves
For the loss and accuracy curves below, it is worth noting that the six models have not run for the same amount of epochs due to a strict early-stopping callback, making the model training stop if the validation accuracy does not improve for more than 2 epochs. 

#### Neural Network
<p align="left">
  <img src="https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/E1_visualisations/NN_histories.png">
</p>

#### LeNet
<p align="left">
  <img src="https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/E1_visualisations/LeNet_histories.png">
</p>

#### VGG16
<p align="left">
  <img src="https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/E1_visualisations/VGG16_histories.png">
</p>

In general, the ```LeNet``` and ```NN``` seem to fit well to the data in comparison to the ```VGG16```that shows signs of overfitting with the training loss continously dropping while validation loss is increasing slightly. Although the ```REAL LeNet``` also shows signs of this (with a spike upward in validation loss at the 8th epoch), it is less prominent.  

### (```E1```)  Evaluation Metrics: F1-score
|            |   Airplane |   Automobile |   Bird |   Cat |   Deer |   Dog |   Frog |   Horse |   Ship |   Truck |   Accuracy |   Macro_Avg |   Weighted_Avg |   Epochs |
|------------|------------|--------------|--------|-------|--------|-------|--------|---------|--------|---------|------------|-------------|----------------|----------|
| REAL VGG16 |       0.65 |         0.69 |   0.52 |  0.48 |   0.54 |  0.57 |   0.67 |    0.65 |   0.72 |    0.68 |       0.62 |        0.62 |           0.62 |       10 |
| FAKE VGG16 |       0.86 |         0.87 |   0.84 |  0.78 |   0.91 |  0.73 |   0.94 |    0.87 |   0.84 |    0.85 |       0.85 |        0.85 |           0.85 |       13 |
| FAKE LeNet |       0.86 |         0.89 |   0.8  |  0.77 |   0.89 |  0.7  |   0.95 |    0.84 |   0.82 |    0.87 |       0.84 |        0.84 |           0.84 |       11 |
| REAL LeNet |       0.68 |         0.75 |   0.47 |  0.48 |   0.58 |  0.48 |   0.72 |    0.71 |   0.74 |    0.69 |       0.63 |        0.63 |           0.63 |       18 |
| REAL NN    |       0.36 |         0.45 |   0.29 |  0.21 |   0.32 |  0.34 |   0.36 |    0.41 |   0.46 |    0.46 |       0.37 |        0.37 |           0.37 |       20 |
| FAKE NN    |       0.55 |         0.74 |   0.58 |  0.52 |   0.67 |  0.43 |   0.55 |    0.55 |   0.61 |    0.63 |       0.59 |        0.58 |           0.58 |       20 |

For all models, the F1-score for each class along with the overall accuracies are highlighted in the table above. For precision and recall metrics, please check the individual metrics.txt files in the ```E1_results``` folder. 

In general, accuracies are higher for the ```FAKE``` dataset. It may be that the dataset is less complex/noisy.

### (```E2```) Evaluating ```FAKE``` Classifiers on ```REAL``` Test Data
Since the ```FAKE LeNet (F1 = 0.84)``` and ```FAKE VGG16 (F1 = 0.85)``` performed similarly, both are evaluated on the ```REAL``` CIFAR-10 test dataset. The table below shows the F1-scores: 

|            |   Airplane |   Automobile |   Bird |   Cat |   Deer |   Dog |   Frog |   Horse |   Ship |   Truck |   Accuracy |   Macro_Avg |   Weighted_Avg |   Epochs |
|------------|------------|--------------|--------|-------|--------|-------|--------|---------|--------|---------|------------|-------------|----------------|----------|
| FAKE LeNet |       0.38 |         0.39 |   0.33 |  0.28 |   0.27 |  0.3  |   0.11 |    0.41 |   0.56 |    0.46 |       0.36 |        0.35 |           0.35 |       11 |
| FAKE VGG16 |       0.46 |         0.44 |   0.37 |  0.34 |   0.37 |  0.39 |   0.17 |    0.48 |   0.57 |    0.53 |       0.42 |        0.41 |           0.41 |       18 |

Interestingly, although the ```FAKE``` models do not outperform the ```REAL``` models on the ```REAL``` test data, the ```VGG16```performs surprisingly well with (```F1 = 0.42```). This is especially remarkable, considering the loss curves of  ```VGG16``` showing signs of overfitting. A possible explanation is to be found in the fact that  ```VGG16``` is pre-trained and likely contains image embeddings for the 10 classes, making it an easier task to fit a classifier with  ```VGG16```. 


## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk

## References
Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. 

Bird, J.J., Lotfi, A. (2023). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. arXiv preprint https://arxiv.org/abs/2303.14126 
