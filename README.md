# Portuguese Meals Classification

A classificator for ['Portuguese Meals Dataset'](https://www.kaggle.com/datasets/catarinaantelo/portuguese-meals)  

- [Portuguese Meals Classification](#portuguese-meals-classification)
  - [Sneak peek at results](#sneak-peek-at-results)
  - [Usage](#usage)
    - [Run locally](#run-locally)
    - [Docker](#docker)
  - [Dataset](#dataset)
  - [Data processing](#data-processing)
  - [Model](#model)
  - [Pipelines and tracking](#pipelines-and-tracking)

## Sneak peek at results

F1 score of `99.54%` on validation split (15% of dataset).  
Baseline model has a size of `153mb`, TFlite converted model has a size of `50mb` and it's possible to bring the size down to aroun `14mb` with `pruning` and by using other `quantization` settings, confusion matrix for that run:  
![confusion matrix](images/cofusion_matrix.jpg)

## Usage

### Run locally

- git clone
- pip install -r requirements.txt
- python3 app.py  
  
`Gradio` will start a server which you can use for inference, usually it starts at `http://127.0.0.1:7861/`. It also starts a publicly availible server at `*.gradio.app`. Actual addresses will be shown in the console.  

![Inference preview](images/inferece.jpg)  

~~Alternatively it's also availible on [Huggingface spaces](https://PLSINPUT)~~  
~~And on my [website](https://empyempt.github.io/Portfolio/)~~

### Docker

Docker container is also availible on [Docker Hub](https://hub.docker.com/repository/docker/empyempt/portugesemeals)

~~~bash
docker pull empyempt/portugesemeals:latest
~~~

## Dataset

['Portuguese Meals Dataset'](https://www.kaggle.com/datasets/catarinaantelo/portuguese-meals) is the only dataset used to train the model  

Raw data is stored at `./data/raw/`  
Augmented images were compressed into `.npz` files and saved at `./data/augmented_compressed/`  
Data was split into `Train-Test-Val` chunks in `70-15-15` proportions. All of label-path pairs are saved in `./data/csv/full.csv` as:  

|           Label           |        File path        |
| ------------------------- | ----------------------- |
| Class name as string      | Relative file path      |  

Used data (raw images, csv, .npz) can be retrieved with [DVC](https://dvc.org/):  

~~~bash
dvc pull
~~~

## Data processing

Since initial dataset is imbalanced a combination of `undersampling`, `oversampling`, `augmentation` and other techniques were used  
Augmentation was done with [Albumentations](https://albumentations.ai/)  
`Label smoothing` was used during training  
`Test time augmentation` is used at inference for higher accuracy

## Model

Built using Tensorflow with `EfficientNetV2B3` as backbone  
`TFLite` converted model is used for inferece  
Since `EfficientNetV2B3` is a pretrained `keras` model I wasn't able to *easily* `prune` the model or do `pruning aware training` to achive `faster prediction` time or `lower model size` (thought it IS possible)  
Also, because of the reason stated above I wasnt able to *properly* `quantize` the model and used `default TFlite optimization`  

## Pipelines and tracking

Data is tracked with [DVC](https://dvc.org/), code is tracked by [GIT](https://git-scm.com/) and uploaded to [this repo](https://github.com/EmpyEmpt/Portuguese-Meals-Classification) on GitHub (well yeah you're already here anyways), experiments are tracked with [WANDB](https://wandb.ai/) in [this](https://wandb.ai/empyempt/Portugese%20Meals%20Classification) project
