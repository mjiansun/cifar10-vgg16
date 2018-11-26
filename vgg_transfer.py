import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.datasets import cifar10
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.engine import Model
from keras.models import load_model
from tensorflow.python.client import device_lib
from keras.utils.vis_utils import plot_model #for graphical demonstration of Network model #requires graphwiz. Not active for now...
from keras.callbacks import ModelCheckpoint
import os
import vgg



####To enable GPU computation, comment this two lines. (if you have CUDA installed, it will work with GPU)
#sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) #disables gpu
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #disables gpu

#IF you have old computer and want a working test without waiting too much, Set this option to 1
WANNAFASTTRAINING=1

img_width, img_height = 32, 32
#img_width, img_height = 224, 224
dimensionx=3

nb_epoch = 2 #needs maybe hours if you increase it. If increases also training will be good.
nb_classes =10 #Number of classes that exists in cifar dataset.

#SGD: Gradient Descent with Momentum and Adaptive Learning Rate
#for more, see here: https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
learningrate=1e-5 #be careful about this parameter. 1e-3 to 1e-8 will train better while learningrate decreases.
momentum=0.90


#according to this: https://datascience.stackexchange.com/questions/16807/why-mini-batch-size-is-better-than-one-single-batch-with-all-training-data
# you should select this batch sizes between [2-32] for better results.
# If your program not works, your memory is not enough!!!. You can decrease the amount of batch size to run this training successfully.
batch_trainsize=32
batch_testsize=32


#switch this parameter to 1 if you want to continue with previous trained model.
#	-> This will load the previous trained "cifar10-vgg16_model_alllayersv2.h5"
USEPREVIOUSTRAININGWEIGHTS=1



#Load Training and Test data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

if WANNAFASTTRAINING == 1 :
    X_train= X_train[0:1000,:,:,:]
    y_train= y_train[0:1000]
    X_test= X_train[0:200,:,:,:]
    y_test= y_train[0:200]

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

nb_train_samples = X_train.shape[0]
nb_validation_samples = X_test.shape[0]



if USEPREVIOUSTRAININGWEIGHTS == 0:
	base_model = vgg.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, dimensionx))
	# Extract the last layer from third block of vgg16 model
	last = base_model.get_layer('block3_pool').output
	# Add classification layers on top of it
	x = Flatten()(last)
	x = Dense(256, activation='relu')(x)
	x = Dropout(0.5)(x)
	pred = Dense(nb_classes, activation='sigmoid')(x)

	model = Model(base_model.input, pred)
else :
	model = load_model('./trained_models\cifar10-vgg16_model_alllayers.h5')


# set the base model's layers to non-trainable
# uncomment next two lines if you don't want to
# train the base model
# for layer in base_model.layers:
#     layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=learningrate, momentum=momentum),
              metrics=['accuracy'])

##See your model in terminal output
model.summary()

##See your CPU or GPU properties.
print(device_lib.list_local_devices())



###### please install pydot with pip install pydot and download graphwiz from website :https://graphviz.gitlab.io/_pages/Download/Download_windows.html
####add graphwiz path to visualize model graph. No need for now.
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#plot_model(model, to_file='outputs/model_plot.png', show_shapes=True, show_layer_names=True)


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_trainsize)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(X_test, Y_test, batch_size=batch_testsize)

# callback for tensorboard integration
tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

#Callback for checkpoint. if the mode is improved, at the end of the epoch, model is saved.
filepath="./trained_models/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

print(  " --nb_train_samples: " + str(nb_train_samples) + "\n" +
		" --nb_validation_samples: " + str(nb_validation_samples) + "\n" +
		" --nb_epoch: " + str(nb_epoch) +  "\n" +
        " --nb_classes: " + str(nb_classes) +  "\n" +
        " --learning rate: " + str(learningrate) +  "\n" +
        " --momentum: " + str(momentum) + "\n" +
        " --batchtrainsize: " + str(batch_trainsize) + "\n" +
        " --batchvalidationsize: " + str(batch_testsize) + "\n" +
        " --optimizer: SGD\n" +
        " --metrics: accuracy\n" +
        " --model: VGG16 (until block3_pool layer)\n"
        );

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps =nb_validation_samples,
    callbacks=[tb,checkpoint])


# serialize model to JSON
model_json = model.to_json()
with open("./outputs/model.json", "w") as json_file:
    json_file.write(model_json)

# save the model !!!Be careful, if you want to load it again, delete "_v2" tag in name
model.save('./trained_models\cifar10-vgg16_model_alllayers_v2.h5')

del model #prevent memory leak