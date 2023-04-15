#!/usr/bin/bash

ROOT_DIR=$(git rev-parse --show-toplevel)
cd ${ROOT_DIR}/data

# Download diffusiondb (first 10k data)
# curl command didn't actually download data
# Following Section 2 of https://github.com/poloclub/diffusiondb/blob/main/notebooks/example-loading.ipynb
# TLDR: the data is split up into 2000 zip files with each containing 1000 data
# Zip file contaings PNG files and 1 JSON that maps PNGs to prompts 
if [ ! -d "./diffusiondb_huggingface" ]; then
    mkdir ./diffusiondb_huggingface
    cd ./diffusiondb_huggingface

    for i in {000001..000010}; do
        wget https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-${i}.zip
        mkdir ./part-${i}
        unzip -q part-${i}.zip -d ./part-${i}
        rm part-${i}.zip
    done
    
    cd ..
fi

# https://cocodataset.org/#download
# https://github.com/cocodataset/cocoapi/issues/368
if [ ! -d "./coco" ]; then
    mkdir coco
    cd coco

    curl -O http://images.cocodataset.org/zips/train2017.zip
    unzip -q train2017.zip
    rm train2017.zip
    curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip

    cd ..
fi
