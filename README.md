# anfr_neurips
Code for Submission #17084

## ANFR Model
The weights for our pre-trained model are available as a github release on this page. The full list of arguments used to train the model on Imagenet are in the `pretrained_model_args.yaml` file.

## Datasets
To prepare the Fed-ISIC2019 dataset, please follow the instructions of the original repository found [here](https://github.com/owkin/FLamby).
To prepare the FedChest dataset, it is necessary to first download the original datasets as described in the folder `data`. Each dataset's folder contains the necessary pre-processing scripts. For FedChest, after pre-processing please move all the images and generated csv files to the FedChest folder and run the relevant script to generate datalists.

## Training

Our code is meant to be installed as a library using `pip install -e .

We provide sample bash scripts for training on each dataset. In short, the bash scripts call a dataset-specific `setup.py` script to easily modify the hyper-parameters before calling nvflare to train.
