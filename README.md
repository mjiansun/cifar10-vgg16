# vgg16 cifar10-Dataset Training, Prediction Using Trained Model


## Description
CNN to classify the cifar-10 database by using a vgg16 trained on Imagenet as base.
The approach is to transfer learn using the first three blocks (top layers) of vgg16 network and adding FC layers on top of them and train it on CIFAR-10. 
You can use previously trained network which is saved with ".h5" extension. Also it is possible to predict the class of any unknown image(s).


## Training The Vgg16 Model
Trained using two approaches
1. Keeping the base model's layer fixed,
2. By training end-to-end

The model is modified to decrease the RAM and CPU requirements. Therefore, after block3_conv1 layer, the model is cut.
[See The Modified Vgg16 Model Visualization](outputs/model_plot.png)

Use this command to train the model, open the console (cmd in windows or terminal in linux)
```console
#cd "Your project folder which contains vgg_transfer.py"
python vgg_transfer.py
```




## Prediction
Any cifar-10 dataset image or custom 32x32x3 RGB image can be predicted using "vgg_predict.py" but you should modify the source and load the image as an input. 
Trained [Model](trained_model/cifar10-vgg16_model.h5 "Trained Model") is used to predict the input(s).
```console
#cd "Your project folder which contains vgg_predict.py"
python vgg_predict.py
```
[See the prediction output](outputs/output_predict.txt "Prediction output")


## Displaying Filters
After training successfully, the ".h5" model will be saved into trained_model directory.
To visualize some of the filters in the 'block3_conv1' layer, use this command:
```console
#cd "Your project folder which contains shownetworkFilters.py"
python shownetworkFilters.py
```
[See the filter visualization](outputs/stitched_filters_block3_conv1_8x8.png)


## Displaying Graphics
To display the graphics of training results, simply run this on console:
```console
#cd "Your project folder"
tensorboard --logdir="logs"
```
Monitor the Graphs using your webbrowser. http://localhost:6006


#### Files
Source Files:
* vgg_transfer.py - The main file for training the modified VGG16 model with cifar10-Dataset.
* vgg.py - Modified version of Keras VGG implementation to change the minimum input shape limit for cifar-10 (32x32x3)
* shownetworkFilters.py - Visualizes some of the filters in the 'block3_conv1' layer
* vgg_predict.py - Takes 2 images from the cifar10-Dataset and predicts the classes. 
