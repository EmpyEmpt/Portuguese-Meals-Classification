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

`96.43%` accuracy on val split (10% of dataset) and a `95.78%` on a whole dataset.  
Baseline model has a size of `129mb`, TFlite converted model has a size of `42mb` and it's possible to bring the size down to aroun `12mb` with `pruning` and by using other `quantization` settings, confusion matrix for that run:  
![confusion matrix](images\cofusion_matrix.jpg)

## Usage

### Run locally

- git clone
- pip install -r requirements.txt
- python3 inference.py  
  
`Gradio` will start a server which you can use for inference, usually it starts at `http://127.0.0.1:7861/`. It also starts a publicly availible server at `*.gradio.app`. Actual addresses will be shown in the console.  

![Inference preview](images\inferece.jpg)

### Docker

Docker container is also availible on [Docker Hub](https://hub.docker.com/repository/docker/empyempt/portugesemeals)

~~~bash
docker pull empyempt/portugesemeals:latest
~~~

## Dataset

['Portuguese Meals Dataset'](https://www.kaggle.com/datasets/catarinaantelo/portuguese-meals) is the only dataset used to train the model  

Raw data is stored at `./data/raw/`  
Augmented images were compressed into `.npz` files and saved at `./data/augmented_compressed/`  
Data was split into `Train-Test-Val` chunks in `80-10-10` proportions. Splits were saved as `.csv` files at `./data/csv/` as:  

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
`Test time augmentation` and `label smoothing` were used as well to achieve higher accuracy  

## Model

Built using Tensorflow with `EfficientNetB3` as backbone  
`TFLite` converted model is used for inferece  
Since `EfficientNetB3` has some 'custom' layers I wasn't able to *easily* `prune` the model or do `pruning aware training` to achive `faster prediction` time or `lower model size` (thought it IS possible with layer-by-layer pruning)  
Also, because of the reason stated above I wasnt able to *properly* `quantize` the model and used `default TFlite optimization`  
It whould be nice to convert the model to `ONNX` format or use `TensorRT`, and I stated it at the end of the [notebook](<https://github.com/EmpyEmpt/Portuguese-Meals-Classification/blob/912a5bbda7bb03f2061055a5266e0f6cef6408b4/notebooks/all_together.ipynb>), but I wasn't able to  

## Pipelines and tracking

Data is tracked with [DVC](https://dvc.org/), code is tracked by [GIT](https://git-scm.com/) and uploaded to [this repo](https://github.com/EmpyEmpt/Portuguese-Meals-Classification) on GitHub (well yeah you're already here anyways), experiments are tracked with [WANDB](wandb.ai) [here](https://wandb.ai/empyempt/Portugese%20Meals%20Classification?workspace=user-empyempt)
