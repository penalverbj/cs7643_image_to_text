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
---
- TinyViT and DistillGPT2 were used as the encoder and decoder, respectively
- Hugging Face's pretrained "nlpconnect/vit-gpt2-image-captioning" VisionEncoderDecoderModel was used as the teacher for knowledge distilation
### TinyViT
- https://github.com/microsoft/Cream/tree/main/TinyViT
- We're including the entire git repo because the pre-trained pytorch model requires the same dir structure when imported
- We have also commented out line 578 of TinyViT/models/tiny_vit.py so that we may access individual hidden states of the TinyViT classifier
- CNN layers were initially frozen
```
$ git clone https://github.com/microsoft/Cream.git
$ cp -r Cream/TinyViT [GIT_REPO_LOC]/models/
$ cd [GIT_REPO_LOC]/models/TinyViT
$ pip install -r requirements.txt
$ python
>>> import torch
>>> from models.tiny_vit import tiny_vit_21m_224
>>> model = tiny_vit_21m_224(pretrained=True)
>>> torch.save(model, './tinyvit_21M.pt')
```
### DistillGPT2
- https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel
- Embedding layers were bypassed and layers were initially frozen

### Teacher Model
- https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder#randomly-initializing-visionencoderdecodermodel-from-model-configurations
- See "nlpconnect/vit-gpt2-image-captioning"

## Knowledge Distillation
---
### Preprocessing
- Inspired by efficient knowledge distilation process described by TinyViT, we 1st perform the teacher foward-pass on all COCO training samples prior to knowledge distilation.
    - https://arxiv.org/pdf/2207.10666.pdf
- The hidden states outputed from the teacher model are saved in `data/teacher_out`
- `data/teacher_out/teacherHidden.csv*` contains teacher hidden state values split into multiple csv files
    - Each line corresponds to a sample in `data/coco/train2017` and each column a hidden state dim
- `data/teacher_out/teacherResults.csv.gz` contains teacher text output generated from the hidden states
```
$ cd src/
$ python

# NOTE: This will take ~4 hours on Nvidia 2080Ti
>>> from teacher import Teacher
>>> teacher_class = Teacher()
>>> teacher_class.process_batch()

$ cd ../data/teacher_out
$ split teacherHidden.csv teacherHidden.csv.part -b 50m
$ gzip *

# To unsplit, simply do
zcat teacherHidden.csv.part*.gz > teacherHidden.csv
```
### Running
```
$ python src/distill.py
```