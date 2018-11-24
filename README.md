# cifar10-vgg16

## Description
CNN to classify the cifar-10 database by using a vgg16 trained on Imagenet as base.
The approach is to transfer learn using the first three blocks (top layers) of vgg16 network and adding FC layers on top of them and train it on CIFAR-10. 
You can use previously trained network which is saved with ".h5" extension

## Training
Trained using two approaches
1. Keeping the base model's layer fixed,
2. By training end-to-end

Use this command to train the model, open the console (cmd in windows or terminal in linux)
```console
#cd "Your project folder which contains vgg_transfer.py"
python vgg_transfer.py
```


## Displaying Filters
After training successfully, the ".h5" model will be saved into trained_model directory.
To visualize some of the filters in the 'block3_conv1' layer, use this command
```console
#cd "Your project folder which contains shownetworkFilters.py"
python shownetworkFilters.py
```


## Displaying Graphics
To display the graphics of training results, simply run this on console
```console
#cd "Your project folder"
tensorboard --logdir="logs"
```
Monitor the Graphs using your webbrowser. http://localhost:6006


#### Files
Source Files:
* vgg_transfer.py - The main file with training
* vgg.py - Modified version of Keras VGG implementation to change the minimum input shape limit for cifar-10 (32x32x3)
* shownetworkFilters.py - Visualizes some of the filters in the 'block3_conv1' layer 