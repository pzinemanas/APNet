# APNet

This repository includes the code for running the experiments reported in

Zinemanas, P.; Rocamora, M.; Miron, M.; Font, F.; Serra, X. [An Interpretable Deep Learning Model for Automatic Sound Classification](https://www.mdpi.com/2079-9292/10/7/850). Electronics 2021, 10, 850. https://doi.org/10.3390/electronics10070850 

## Installation instructions
APNet uses DCASE-models and therefore please follow the recomendations from this library: 

We recommend to install DCASE-models in a dedicated virtual environment. For instance, using [anaconda](https://www.anaconda.com/):
```
conda create -n apnet python=3.6
conda activate apnet
```
For GPU support:
```
conda install cudatoolkit cudnn
```
DCASE-models uses [SoX](http://sox.sourceforge.net/) for functions related to the datasets. You can install it in your conda environment by:
```
conda install -c conda-forge sox
```
Before installing the library, you must install only one of the Tensorflow variants: CPU-only or GPU.
``` 
pip install "tensorflow<1.14" # for CPU-only version
pip install "tensorflow-gpu<1.14" # for GPU version
```

Now please install DCASE-models:
``` 
pip install "DCASE-models==0.2.0-rc0"
```

Install other dependencies:
``` 
pip install "mirdata>=0.3.0"
``` 

Now you can clone and use APNet:
``` 
git clone https://github.com/pzinemanas/APNet.git
cd APNet
```

## Usage

### Download datasets
``` 
cd experiments
python download_datasets.py -d UrbanSound8k

```
### Train models
``` 
python train.py -m APNet -d UrbanSound8k -f MelSpectrogram -fold fold1
# Repeat for the other folds
``` 

### Evaluate models
``` 
python evaluate.py -m APNet -d UrbanSound8k -f MelSpectrogram
``` 
