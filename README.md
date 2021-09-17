# Neural Optical Beam Propagation
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![GitHub stars](https://img.shields.io/github/stars/mremilien/NeuralOpticalBeamPropagation.svg?style=social)](https://github.com/mremilien/NeuralOpticalBeamPropagation/stargazers)

This reporsitory contains the Pytorch implementation of the thesis **"Neural Optical Beam Propagation"** which is actually my master dissertation at University of College London (UCL) supervised by Dr.Kaan Aksit. The main goal of this thesis is to simulate the calculation of beam propagation on the tilted planes by a deep learning method. And more details could be found in my [dissertation](https://github.com/mremilien/NeuralOpticalBeamPropagation/blob/main/docs/neural_optical_beam_propagation.pdf).

## Dependencies
I implemented our code on python 3.6. And other used modules are:
* PyTorch==1.6.0
* CUDA==9.0
* Diffractsim==0.0.10


## File structure
If you run our code correctly, the file structure should be like this:
* `data/` the dataset for testing our algorithm.
* `docs/` the documents about this project including my dissertation and ppt.
* `output/` training results will be saved in this folder.
* `src/` includes all of the source code.
   * `beam_propagation/` the code for the fast rotate angular spectrum function.
   * `generate_dataset/` generate our datasets.
   * `neural_network1/` our basic goal which is simulating the beam propagation with a certain tited angle.
   * `neural_network2/` the extension content of our thesis that adds a an additional input, the tilted angle, for our model.


## Datasets
There are several datasets for this project, we [use Faces-LFW](http://vis-www.cs.umass.edu/lfw/lfw.tgz) here as an example to train our model.
You should copy and unzip this file at `./data/original/lfw`.
Then run the following commands.
``` bash
cd ./src/generate_dataset/

# the dataset used for our basic goal.
python generate_dataset_nn1.py
python separate_train_test.py ../../data/processed/lfw_nn1

# the dataset used for our extension content.
python generate_dataset_nn2.py
python separate_train_test.py ../../data/processed/lfw_nn2
```
Finally, the file sturction of the `data/` directory should be like this.
```
data
├── original
│   └── lfw
└── processed
    ├── lfw_nn1
    |   ├── data
    |   ├── configuration.json
    |   ├── testset.txt
    |   └── trainset.txt
    └── lfw_nn2
        ├── data
        ├── configuration.json
        ├── testset.txt
        └── trainset.txt
 ```
 
 
## Training
``` bash
# for the basic goal
cd ./src/neural_network1
python train_model.py --epochs 10  --batch-size 4  --learning-rate 0.001 --validation 20 --device 0

# for the extension content
cd ./src/neural_network2
python train_model.py --epochs 10  --batch-size 4  --learning-rate 0.001 --validation 20 --device 0
```




