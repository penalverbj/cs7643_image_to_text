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