# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 01:09:08 2018

@author: kusiwu
@git: https://github.com/kusiwu
"""
# DEPENDENCIES
from keras import optimizers
from keras.datasets import cifar10
from tensorflow.python.client import device_lib
from keras.models import load_model
import numpy as np


img_width, img_height = 32, 32
dimensionx=3

nb_classes =10


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

xTestPictures=X_test[0:2,:,:,:]
yTestExpectedLabels=y_test[0:2];

xTestPictures=xTestPictures * 1. / 255; #very important! https://github.com/keras-team/keras/issues/6987

model = load_model('./trained_models/cifar10-vgg16_model_alllayers.h5')

model.summary()
print(device_lib.list_local_devices())

yFit = model.predict(xTestPictures, batch_size=10, verbose=1)
y_classes = yFit.argmax(axis=-1)
print("Found classes from Prediction:");
print(y_classes.flatten())

print("\n\nTrue classes:");
print(yTestExpectedLabels.flatten())


diffpredictionvstruth=yTestExpectedLabels.flatten()-y_classes.flatten()
print("\n\nDiff classes:");
print(diffpredictionvstruth)
#np.savetxt('./logs/DiffClasses.out', diffpredictionvstruth, delimiter=',')   # you can save the results like this


nb_validation_samples = xTestPictures.shape[0]
print("\n\n Wrong class number:");
print(str(len(np.nonzero(diffpredictionvstruth)[0]))+ '/' + str(nb_validation_samples) + ' is wrongly classified')

#print()
#print(yFit)
del model