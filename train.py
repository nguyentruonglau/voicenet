import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from imutils.paths import list_images
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn import metrics
from keras.callbacks import ModelCheckpoint
from voicenet import VoiceNet
import matplotlib.pyplot as plt
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-batch", "--batch_size", required=False, default=64,
    help="Batch size")
ap.add_argument("-class", "--num_class", required=False, default=80,
    help="Number of class")
ap.add_argument("-epochs", "--epochs", required=False, default=100,
    help="Number of epochs")
args = vars(ap.parse_args())

#defines some parameters
img_height = 40
img_width = 160
batch_size = int(args['batch_size'])
num_class = int(args['num_class'])

#train
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    './img_data/',
    labels = 'inferred',
    label_mode = 'int',
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 42,
    validation_split = 0.2,
    subset = 'training'
)

#validation
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    './img_data/',
    labels = 'inferred',
    label_mode = 'int',
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 42,
    validation_split = 0.2,
    subset = 'validation'
)

#normalize data
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


#prefetch train
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

#prefetch validation
ds_validation = ds_validation.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_validation = ds_validation.cache()
ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)


#model
model = VoiceNet(input_shape = (img_height, img_width, 1), classes = num_class)

#path to save model
if not os.path.exists('model'):
    os.mkdir('model')

path_save_model = './model/voicenet.hdf5'

#model checkpoint
checkpoint = ModelCheckpoint(path_save_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)


# Initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.001
EPOCHS = int(args['epochs'])

#Compiling
opt = Adam(learning_rate=INIT_LR) #decay=INIT_LR/EPOCHS
model.compile(optimizer = opt, loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

# Fitting the CNN
model.fit(ds_train, validation_data=ds_validation, epochs=EPOCHS, callbacks=[checkpoint])
























model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[
            keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ],
    metrics=['accuracy']
)


model.fit(ds_train, validation_data=ds_validation, epochs=2, verbose=1)










