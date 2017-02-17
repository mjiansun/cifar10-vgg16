# cifar10-vgg16

## Description
CNN to classify the cifar-10 database by using a vgg16 trained on Imagenet as base.
The approach is to transfer learn using the first three blocks (top layers) of vgg16 network and adding FC layers on top of them and train it on CIFAR-10. 

## Training
Trained using two approaches for 50 epochs:
1. Keeping the base model's layer fixed, and
2. By training end-to-end

First approach reached a validation accuracy of 95.06%. 
Second approach reached a validation accuracy of 97.41%. 

#### Files
Source Files:
* vgg_transfer.py - The main file with training
* vgg.py - Modified version of Keras VGG implementation to change the minimum input shape limit for cifar-10 (32x32x3)

Outputs:
* [outputs/output_1.txt](outputs/output_1.txt "Outputs for Approach 1")
* [outputs/output_2.txt](outputs/output_2.txt "Outputs for Approach 2")

Trained Models:
* [Approach 1](trained_model/cifar10-vgg16_model.h5 "Trained Model for Approach 1")
* [Approach 2](trained_model/cifar10-vgg16_model_alllayers.h5 "Trained Model for Approach 2")

Tensorboard graphs (Appoach 2):
Validation Accuracy:
![Validation Accuracy](outputs/val_acc.png?raw=true "Validation Accuracy")

Training Accuracy:
![Training Accuracy](outputs/acc.png?raw=true "Training Accuracy")
