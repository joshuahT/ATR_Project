import numpy as np
import os
import nibabel as nib

from scipy import ndimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def read(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize(img, desired_height,  desired_width, desired_depth):
    """Resize across z-axis"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_img(path, desired_height = 128,  desired_width = 128, desired_depth = 64):
    volume = read(path)
    volume = normalize(volume)
    volume = resize(volume, desired_height=desired_height, desired_width=desired_width, desired_depth=desired_depth)
    return volume 







resize_h = 128
resize_w = 128
resize_d = 64
data_folder = f'{os.getcwd()}/labeled_data/data/'

length = 0
for label in os.listdir(data_folder):
    length += len(os.listdir(f'{data_folder}/{label}/'))

X = np.empty(shape=(length,resize_h, resize_w, resize_d))
Y = np.empty(shape=(length))

i = 0
for label in os.listdir(data_folder):
    img_directory = f'{data_folder}/{label}/'
    for img in os.listdir(img_directory):
        X[i] = process_img(img_directory + '/' + img, resize_h, resize_w, resize_d)
        Y[i] = label
        i += 1
        break
    break

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
with tf.device("CPU"):
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))













# @tf.function
# def rotate(volume):
#     """Rotate the volume by a few degrees"""

#     def scipy_rotate(volume):
#         # define some rotation angles
#         angles = [-20, -10, -5, 5, 10, 20]
#         # pick angles at random
#         angle = random.choice(angles)
#         # rotate volume
#         volume = ndimage.rotate(volume, angle, reshape=False)
#         volume[volume < 0] = 0
#         volume[volume > 1] = 1
#         return volume

#     augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
#     return augmented_volume

# def train_preprocessing(volume, label):
#     """Process training data by rotating and adding a channel."""
#     # Rotate volume
#     volume = rotate(volume)
#     volume = tf.expand_dims(volume, axis=3)
#     return volume, label

# def validation_preprocessing(volume, label):
#     """Process validation data by only adding a channel."""
#     volume = tf.expand_dims(volume, axis=3)
#     return volume, label




# train_dataset = (
#     train_loader.shuffle(len(x_train))
#     .map(train_preprocessing)
#     .batch(batch_size)
#     .prefetch(2)
# )
# # Only rescale.
# validation_dataset = (
#     validation_loader.shuffle(len(x_test))
#     .map(validation_preprocessing)
#     .batch(batch_size)
#     .prefetch(2)
# )






# initial_learning_rate = 0.0001
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
# )
# model.compile(
#     loss="SparseCategoricalCrossentropy ",
#     optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
#     metrics=["acc"],
# )

# checkpoint_cb = keras.callbacks.ModelCheckpoint(
#     "3d_image_classification.h5", save_best_only=True
# )
# early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# epochs = 100
# model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=epochs,
#     shuffle=True,
#     verbose=2,
#     callbacks=[checkpoint_cb, early_stopping_cb],
# )

