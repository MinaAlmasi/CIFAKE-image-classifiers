Place your [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset in this folder ! 

The data folder should be structured with subdirectories as such: 
```
├── images
│   ├── metadata        <--- metadata is already there, created by running create_metadata.py
│   │   ├── FAKE
│   │   └── REAL
│   ├── test            <--- testdata split into FAKE and REAL
│   │   ├── FAKE
│   │   └── REAL
│   └── train           <--- traindata split into FAKE and REAL
│       ├── FAKE
│       └── REAL
```