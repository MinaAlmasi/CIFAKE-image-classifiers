# CIFAKE-image-classifiers
This repository forms the self-assigned *assignment 4* in the subject Visual Analytics, Cultural Data Science, F2023. The assignment description can be found here. All code is written by Mina Almasi (202005465) although some code may be adapted from classwork (see also [Code Reuse]()).

The repository aims to investigate the usefulness of artificially generated images when training classifiers. For this purpose, the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset (Bird & Lofti, 2023) is used. 


## Data 
The ```CIFAKE``` dataset contains 60,000 images that are synthetically generated to be equivalent to the ```CIFAR-10 dataset``` (Krizhevsky, 2009) along with 60,000 original CIFAR-10 images: 

<p align="center">
  <img src="https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/docs/CIFAKE-dataset.png">
</p>

## Experimental pipeline and Motivation
The evaluation of artificial (```FAKE```) images is achieved with the following pipeline by training six classifiers. 

1. Train three classifiers in increasing complexity on the ```FAKE```


2. Testing the best ```Fake``` classifier 



## Reproducibility 
To reproduce the results, follow the instructions in the [Pipeline]() section. 

NB! Be aware that training the model is computationally heavy. Cloud computing (e.g., UCloud) with high amounts of ram (or a good GPU) is encouraged.


## Project Structure
The repository is structured as such: 
```

```

## Pipeline 

## Results 
### Loss Curves
#### Neural Network
<p align="left">
  <img src="https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/visualisations/NN_histories.png">
</p>

#### LeNet
<p align="left">
  <img src="https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/visualisations/LeNet_histories.png">
</p>

#### VGG16
<p align="left">
  <img src="https://github.com/MinaAlmasi/CIFAKE-image-classifiers/blob/main/visualisations/VGG16_histories.png">
</p>


### Evaluation Metrics 
|            |   Airplane |   Automobile |   Bird |   Cat |   Deer |   Dog |   Frog |   Horse |   Ship |   Truck |   Accuracy |   Macro_Avg |   Weighted_Avg |   Epochs |
|------------|------------|--------------|--------|-------|--------|-------|--------|---------|--------|---------|------------|-------------|----------------|----------|
| REAL VGG16 |       0.65 |         0.69 |   0.52 |  0.48 |   0.54 |  0.57 |   0.67 |    0.65 |   0.72 |    0.68 |       0.62 |        0.62 |           0.62 |       10 |
| FAKE VGG16 |       0.86 |         0.87 |   0.84 |  0.78 |   0.91 |  0.73 |   0.94 |    0.87 |   0.84 |    0.85 |       0.85 |        0.85 |           0.85 |       13 |
| FAKE LeNet |       0.86 |         0.89 |   0.8  |  0.77 |   0.89 |  0.7  |   0.95 |    0.84 |   0.82 |    0.87 |       0.84 |        0.84 |           0.84 |       11 |
| REAL LeNet |       0.68 |         0.75 |   0.47 |  0.48 |   0.58 |  0.48 |   0.72 |    0.71 |   0.74 |    0.69 |       0.63 |        0.63 |           0.63 |       18 |
| REAL NN    |       0.36 |         0.45 |   0.29 |  0.21 |   0.32 |  0.34 |   0.36 |    0.41 |   0.46 |    0.46 |       0.37 |        0.37 |           0.37 |       20 |
| FAKE NN    |       0.55 |         0.74 |   0.58 |  0.52 |   0.67 |  0.43 |   0.55 |    0.55 |   0.61 |    0.63 |       0.59 |        0.58 |           0.58 |       20 |

## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk

### Code Reuse 
Write text here 

## References
Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. 

Bird, J.J., Lotfi, A. (2023). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. arXiv preprint https://arxiv.org/abs/2303.14126 