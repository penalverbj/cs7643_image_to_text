# cs7643_image_to_text

## Downloading data
---
The following command will download the listed data sets:
```
bash data/download.sh
```
### DiffusionDB
- 1st 10k data from https://huggingface.co/datasets/poloclub/diffusiondb
- See https://github.com/poloclub/diffusiondb/blob/main/notebooks/example-loading.ipynb for how to load data

### COCO
- Training 2017 data set + annotations
- https://cocodataset.org/#download
- Unfortunately, their download tool doesn't work...
    - https://github.com/cocodataset/cocoapi/issues/368

## Setting up environment
---
### Conda Environment
To set up the conda environment you must have Conda installed on your machine. Please follow this website for installation steps: [Conda Install](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 

Once that is done please go to your terminal, navigate to the main project directory, and run the following command:
```
conda env create -f environment.yml
```
This will download all the dependencies you need to run this project, except for PyTorch.

### PyTorch Install
You will also need PyTorch. Before running the install commands, make sure you have your conda environment activated. To do this, go to your terminal and run the following command:
```
conda activate 7643final
```

Please follow the instructions on their website to download the version that is appropriate for your machine: [PyTorch Install](https://pytorch.org/)

Once it is done installing your environment is done setting up.  

## Importing pretrained models
### TinyViT
https://github.com/microsoft/Cream/tree/main/TinyViT
```
$ git clone https://github.com/microsoft/Cream.git
$ cd Cream/TinyViT
$ pip install -r requirements.txt
$ python
>>> import torch
>>> from models.tiny_vit import tiny_vit_21m_224
>>> model = tiny_vit_21m_224(pretrained=True)
>>> torch.save(model, '[GIT_REPO_DIR]/models/tinyvit_21M.pt')
```

test
