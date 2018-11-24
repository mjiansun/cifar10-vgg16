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



img_width, img_height = 32, 32
dimensionx=3

nb_classes =10


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

xTestPictures=X_test[0:2,:,:,:]
yTestExpectedLabels=y_test[0:2];

model = load_model('./trained_models/cifar10-vgg16_model_alllayers.h5')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
              metrics=['accuracy'])

model.summary()
print(device_lib.list_local_devices())

yFit = model.predict(xTestPictures, batch_size=10, verbose=1)
y_classes = yFit.argmax(axis=-1)
print("Found classes:");
print(y_classes.flatten())

print("\n\nTrue classes:");
print(yTestExpectedLabels.flatten())

#print()
#print(yFit)
del model