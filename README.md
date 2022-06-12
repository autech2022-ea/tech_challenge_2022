# ☑️ Technical Assigment: EA ☑️

Hello! Welcome to this technical assigment. 
**Thank you for taking the time to review it!!**

## 🚪 Introduction

The solution for this technical assigment is divided into three main deliverables

1. Script to make inference on a single image or all the images inside a directory
1. Script to run the training
1. Google colab notebook with a deeper explanation on the model design. 
The training process can also be executed from there. 
[Notebook URL](https://github.com/autech2022-ea/tech_challenge_2022/blob/main/BriefExplanationAndDriver.ipynb)

## Summary of the design 

- PyTorch Lightning was used to define the model as `pl.LightningModule`.
- The LightningModule has a backbone that can be selected from a list of possibilites.
- The different backbones are pretrained models 
  [from PyTorch vision](https://pytorch.org/vision/stable/models.html) (ResNet, ViT, etc).
- The backbone can be selected when running `train.py` or `infer.py`
- Some data augmentations are standard from PyTorch vision, and other are hand-crafted for more flexibility
- A google colab notebook is also included, explaining some of the steps taken, and including some
  visualizations. [Notebook URL](https://github.com/autech2022-ea/tech_challenge_2022/blob/main/BriefExplanationAndDriver.ipynb)


## Installation

First of all, clone the repository

```bash
git clone https://github.com/autech2022-ea/tech_challenge_2022.git
```

And then, install the dependencies. Ideally this is run inside a virtual environment or similar:

``` 
# optional
# virtualenv env --python=python3.8
# source env/bin/activate
```

```bash
pip install -r requirements.txt
```

## 1. Inference Script

To make an inference (prediction) on a single image, or all the images of a directory, we need 
to run `infer.py`. There are several sample images in `./sample_set/`, so we can use them 
directly.

### Predict a single image


```bash
python infer.py -i ./sample_set/checked24.png
```

And the result will look like this:

```
$ python infer.py -i ./sample_set/checked24.png
WARNING: Checkpoint path not provided, going to use the pretrained
MESSAGE: Checkpoint used:
        ./trained_models/resnet34/val_acc=0.974-val_loss=0.799-epoch=31.ckpt
The image at path:  './sample_set/checked24.png'
Was predicted as:   0:'checked'
Original output is: [-0.0400328 -3.2875924 -6.266724 ]
```

### Predict all the images inside a directory

As we mentioned, there are several images in `./sample_set`. We can run inference
on all of them with a single command.

```
$ python infer.py -d ./sample_set/
WARNING: Checkpoint path not provided, going to use the pretrained
MESSAGE: Checkpoint used:
        ./trained_models/resnet34/val_acc=0.974-val_loss=0.799-epoch=31.ckpt
The image at path:  './sample_set/checked24.png'
Was predicted as:   0:'checked'
Original output is: [-0.0400328 -3.2875924 -6.266724 ]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
The image at path:  './sample_set/checked3.png'
Was predicted as:   0:'checked'
Original output is: [-0.08739685 -2.6208677  -4.514701  ]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
The image at path:  './sample_set/other18.png'
Was predicted as:   1:'other'
Original output is: [-2.7917514  -0.06646255 -5.813048  ]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
The image at path:  './sample_set/other20.png'
Was predicted as:   0:'checked'
Original output is: [-0.20782778 -2.0601203  -2.8098426 ]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
The image at path:  './sample_set/other22.png'
Was predicted as:   1:'other'
Original output is: [-8.825800e+00 -3.411188e-04 -8.546707e+00]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
The image at path:  './sample_set/unchecked10.png'
Was predicted as:   2:'unchecked'
Original output is: [-3.633664   -3.0485744  -0.07671446]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
The image at path:  './sample_set/unchecked2.png'
Was predicted as:   2:'unchecked'
Original output is: [-6.5624790e+00 -7.4709644e+00 -1.9837003e-03]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
The image at path:  './sample_set/unchecked29.png'
Was predicted as:   2:'unchecked'
Original output is: [-5.7688880e+00 -8.9417763e+00 -3.2593482e-03]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
The image at path:  './sample_set/unchecked3.png'
Was predicted as:   2:'unchecked'
Original output is: [-6.4407706e+00 -8.5843458e+00 -1.7838056e-03]
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
```

### Other options for infer.py

There are other arguments that we can pass to `infer.py`. To see all of them,
we can run `-h`:

```
$ python infer.py -h
usage: infer.py [-h] [-b BACKBONE_NAME] [-c CHECKPOINT_PATH] [-i IMAGE_LOCATION] [-d DIRECTORY_LOCATION] [-w WORKERS] [-g GPUS] [-s BATCH_SIZE] [-r RESULT_CSV_FILE]

Run inference for a single image, or for all the images inside a directory. It can use one of the trained included models or use a checkpoint trained by the user. Either
--image_location or --directory_location must be provided. By default, results are stored at 'last_results.csv', but this path can be selected with --result_csv_file.

optional arguments:
  -h, --help            show this help message and exit
  -b BACKBONE_NAME, --backbone_name BACKBONE_NAME
                        The name of the backbone for the classifier. One of the following:
                        ['resnet18'|'resnet34'|'resnet50'|'vit_b_16'|'convnext_tiny'|'efficientnet_b3'|'efficientnet_b5']. Default and recommended is 'resnet34'.
  -c CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        Location of the checkpoint to use. If not provided, it willuse the best weights included for the particular backbone
  -i IMAGE_LOCATION, --image_location IMAGE_LOCATION
                        Location of a single image to run the prediction
  -d DIRECTORY_LOCATION, --directory_location DIRECTORY_LOCATION
                        Location of a directory to run inference in all the PNG images.
  -w WORKERS, --workers WORKERS
                        The number of workers for the DataLoaders. Default is CPU_COUNT/2.
  -g GPUS, --gpus GPUS  The number of GPUS to use. By default it will use what PyTorch finds
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -r RESULT_CSV_FILE, --result_csv_file RESULT_CSV_FILE
                        Path of a csv file to save the results
```

For instance, by default the last results are stored as .csv file in `./last_results.csv`.
This path can be overwritten with `-r my_results.csv`:

```
$ python infer.py -d ./sample_set/ -r my_results.csv; cat my_results.csv
...(omitted output)...
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
./sample_set/checked24.png,0:checked
./sample_set/checked3.png,0:checked
./sample_set/other18.png,1:other
./sample_set/other20.png,0:checked
./sample_set/other22.png,1:other
./sample_set/unchecked10.png,2:unchecked
./sample_set/unchecked2.png,2:unchecked
./sample_set/unchecked29.png,2:unchecked
./sample_set/unchecked3.png,2:unchecked

```

## 2. Training Script

The training script also has many options that we can configure. However, there is 
default for all of them, so the most basic way to start the training is:

```
python train.py 
```

### About the backbone

The classification task can be done using one of different backbone networks. This backbone
can be selected with `-b/--backbone_name`. These models are taken directly from
[https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html),
however, natually not all the models are available in this system. These are the ones I have 
tested to have a good performance and speed:

- "resnet18"
- "resnte34" **(default)**
- "resnet50"
- "vit_b_16"
- "convnext_tiny"
- "efficientnet_b3"
- "efficientnet_b5"

All training is started with the pretrained models, hence we will be only fine-tunning.

To start a train process with a non-default backbone (different than 'resnet34'), we
can use `-b/--backbone_name`:

```bash
python train.py --backbone_name convnext_tiny
```

### About the checkpoints

A callback is added, so the best 3 checkpoints are always saved and stored
at the location `./checkpoints/{backbone_name}`. When the training end, it will
also print the exact location of the best checkpoint for the run, for instance:

```
FINISHED TRAINING
This is the path of the best model (per val_acc):
./checkpoints/resnet34/val_acc=0.936-val_loss=0.193-epoch=27.ckpt
```

This best checkpoint obtained with `train.py` can naturally be used to run inference
with `infer.py`:

```
python infer.py -b resnet34 -c "./checkpoints/resnet34/val_acc=0.936-val_loss=0.193-epoch=27.ckpt" -i sample_set/other22.png  
MESSAGE: Checkpoint used:
        ./checkpoints/resnet34/val_acc=0.936-val_loss=0.193-epoch=27.ckpt
The image at path:  'sample_set/other22.png'
Was predicted as:   1:'other'
Original output is: [-8.1053362e+00 -6.2434253e-04 -8.0400047e+00]
```

### Other ways to configure train.py

It is also possible to adjust the initial learning rate with '-l', the batch size with '-s', and the 
maximum number of epochs with '-e'. For the complete list of options, we can run `python train.py  --help`:

```
$ python train.py  --help
usage: train.py [-h] [-b BACKBONE_NAME] [-e MAX_EPOCHS] [-g GPUS] [-l LEARNING_RATE] [-s BATCH_SIZE] [-c CHECKPOINTS_LOCATION] [-d DATASET_LOCATION] [-w WORKERS]

Run the training process for the model. There are reasonable default for all the flags. At the end of the training phase, the path of the best trained model will be
printed.

optional arguments:
  -h, --help            show this help message and exit
  -b BACKBONE_NAME, --backbone_name BACKBONE_NAME
                        The name of the backbone for the classifier. One of the following:
                        ['resnet18'|'resnet34'|'resnet50'|'vit_b_16'|'convnext_tiny'|'efficientnet_b3'|'efficientnet_b5']. Default and recommended is 'resnet34'.
  -e MAX_EPOCHS, --max_epochs MAX_EPOCHS
                        Maximum number of epochs. It may not reach the number because of the EarlyStopping callback.
  -g GPUS, --gpus GPUS  The number of GPUS to use. By default it will use what PyTorch finds
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate. Default is 0.001.
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -c CHECKPOINTS_LOCATION, --checkpoints_location CHECKPOINTS_LOCATION
                        Location to store the checkpoints. Default is './checkpoints'
  -d DATASET_LOCATION, --dataset_location DATASET_LOCATION
                        Location of the dataset. By default it will use the dataset included in the challenge, copied at './data/
  -w WORKERS, --workers WORKERS
                        The number of workers for the DataLoaders. Default is CPU_COUNT/2.

```

### About tensorboard

By default the train process is constantly loggin to tensorboard.
While the process is running, we can run in another terminal:

```
tensorboard --logdir=./lightning_logs/
```

And then we can see the progress of several metrics in the default URL [http://localhost:6006/#scalars](http://localhost:6006/#scalars)


### About the test set

I created another set of labeled images. After the training is finished, the model
will predict the result for this set. The set is included by default at `./test_dataset`.

These metrics are just reported, but not considered for model selection, etc:

```
TEST_SET Metrics = {'test_loss': 1.180214524269104, 'test_acc': 0.4457831382751465}
```

## Colab notebook

WIP

# Final notes, ideas to improve

- With enough time, it would be possible to create synthetic examples
    - Randomly create CSS and HTML code 
    - Randomly change colors, margins, shadows, background, texts, etc
    - Programatically take a screenshot of the results
- It may be interesting to break the problem in two steps: 1. checkbox/not a checkbox. 2. checked/uncheked.
- With more time, it would be interesting to explore a custom model.
- It may be useful to add a preprocess step to remove all the surrounding text, I wonder if 
it is making noise for the classifier.
