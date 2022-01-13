# Camvid project

## Introduction

This is a semantic segmentation project aimed at the implementation of a UNet network for an arbitrary prespecified encoder (e.g. ResNet34 or ResNet50). I use the classical dataset *Camvid* gathered at the University of Cambridge, see http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid, and easily accessible via Kaggle dataset https://www.kaggle.com/carlolepelaars/camvid.

## Some details

The project is divide into multiple modules:

* `datasets/` contains some basic preprocessing of the data and an implementation of  *pytorch* datasets.
* `metrics/` contains relevant accuracy metrics (accounting for class imbalance caused by the presence of *void* class).
* `models/` contains the implementation of the UNet architecture using hooks and ResNet blocks (available in *torchvision*),
* `visualization/` contains some visualisation routines useful for the presentation of model quality.
* `training.py` contains the *pytorch_lightning* module describing the full structure of the model and its training and validation routines.

For the metric visualisation I currently use `TensorBoard` (however I also did some experiments with `wandb`).

In order not to clutter the training notebook all the dependencies are available in `requirements.txt`.

## Results

Using the implementation we were able to obtain a 0.90 test set and 0.92 validation set accuracy model which is on par with a model trained in the `fast.ai` course. 

## Little future improvements

I would like to compute the mean intersection over union metric to compare the model with SOTA techniques described on https://paperswithcode.com/sota/semantic-segmentation-on-camvid.



