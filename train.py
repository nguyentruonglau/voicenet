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
img_width = 80
batch_size = int(args['batch_size'])
num_class = int(args['num_class'])

#train
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'mfcc_feature',
    labels = 'inferred',
    label_mode = 'categorical',
    color_mode = 'grayscale',
    batch_size = 256,
    image_size = (40, 80),
    shuffle = True,
    seed = 42,
    validation_split = 0.1,
    subset = 'training'
)

#validation
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'mfcc_feature/',
    labels = 'inferred',
    label_mode = 'categorical',
    color_mode = 'grayscale',
    batch_size = 256,
    image_size = (40, 80),
    shuffle = True,
    seed = 42,
    validation_split = 0.1,
    subset = 'validation'
)

#normalize data
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

#display
def displays(H, EPOCHS):
  # plot the training loss and accuracy
  N = np.arange(0, EPOCHS)
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(N, H.history["accuracy"], label="train_acc")
  plt.plot(N, H.history["val_accuracy"], label="val_acc")
  plt.title("Training Accuracy (VoiceNet)")
  plt.xlabel("Epoch")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.savefig("VoiceNet.jpg")

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
with tf.device("/gpu:0"):
    H = model.fit(ds_train, validation_data=ds_validation, epochs=EPOCHS, callbacks=[checkpoint])
    
#display
displays(H, EPOCHS)
