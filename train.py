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
from utils import read_config_train

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-batch_size", "--batch_size", required=False, default=32,
    help="Batch size")
ap.add_argument("-num_class", "--num_class", required=False, default=40,
    help="Number of class")
ap.add_argument("-epochs", "--epochs", required=False, default=100,
    help="Number of epochs")
args = vars(ap.parse_args())

#path to config file
cfg_file = './cfg/config.cfg'

#read config file
options = read_config_train(cfg_file)

#defines some parameters
img_height = options.img_height
img_width = options.img_width
batch_size = int(args['batch_size'])
num_class = int(args['num_class'])

#train
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'mfcc',
    labels = 'inferred',
    label_mode = 'categorical',
    color_mode = 'grayscale',
    batch_size = options.batch_size,
    image_size = options.image_size,
    shuffle = True,
    seed = options.seed,
    validation_split = options.validation_split,
    subset = 'training'
)

#validation
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'mfcc',
    labels = 'inferred',
    label_mode = 'categorical',
    color_mode = 'grayscale',
    batch_size = options.batch_size,
    image_size = options.image_size,
    shuffle = True,
    seed = options.seed,
    validation_split = options.validation_split,
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
  plt.savefig("voicenet.jpg")

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

path_save_model = options.path_save_model

#model checkpoint
checkpoint = ModelCheckpoint(path_save_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)


# Initialize our initial learning rate and # of epochs to train for
INIT_LR = options.init_lr
EPOCHS = int(args['epochs'])

#Compiling
opt = Adam(learning_rate=INIT_LR) #decay=INIT_LR/EPOCHS
model.compile(optimizer = opt, loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

# Fitting the CNN
with tf.device("/gpu:0"):
    H = model.fit(ds_train, validation_data=ds_validation, epochs=EPOCHS, callbacks=[checkpoint])
    
#display
displays(H, EPOCHS)
