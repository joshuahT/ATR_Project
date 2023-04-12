#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import nibabel as nib

from scipy import ndimage


def read_nifti_file(filepath):
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


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
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


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


# In[3]:


get_ipython().run_line_magic("cd", "ATR_Project")


# In[2]:


# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
non_target_paths = [
    os.path.join(os.getcwd(), "labeled_data/data/0", x)
    for x in os.listdir("labeled_data/data/0")
]


saline_paths = [
    os.path.join(os.getcwd(), "labeled_data/data/1", x)
    for x in os.listdir("labeled_data/data/1")
]

rubber_paths = [
    os.path.join(os.getcwd(), "labeled_data/data/2", x)
    for x in os.listdir("labeled_data/data/2")
]

clay_paths = [
    os.path.join(os.getcwd(), "labeled_data/data/3", x)
    for x in os.listdir("labeled_data/data/3")
]

print("scans with non target: " + str(len(non_target_paths)))
print("scans with saline: " + str(len(saline_paths)))
print("scans with rubber: " + str(len(rubber_paths)))
print("scans with clay: " + str(len(clay_paths)))


# In[3]:


# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
non_target_scans = np.array([process_scan(path) for path in non_target_paths])
saline_scans = np.array([process_scan(path) for path in saline_paths])
rubber_scans = np.array([process_scan(path) for path in rubber_paths])
clay_scans = np.array([process_scan(path) for path in clay_paths])

# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
non_target_labels = np.array([0 for _ in range(len(non_target_scans))])
saline_labels = np.array([1 for _ in range(len(saline_scans))])
rubber_labels = np.array([2 for _ in range(len(rubber_scans))])
clay_labels = np.array([3 for _ in range(len(clay_scans))])

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate(
    (non_target_scans[70:], saline_scans[70:], rubber_scans[70:], clay_scans[70:]),
    axis=0,
)
y_train = np.concatenate(
    (non_target_labels[70:], saline_labels[70:], rubber_labels[70:], clay_labels[70:]),
    axis=0,
)
x_val = np.concatenate(
    (non_target_scans[:70], saline_scans[:70], rubber_scans[:70], clay_scans[:70]),
    axis=0,
)
y_val = np.concatenate(
    (non_target_labels[:70], saline_labels[:70], rubber_labels[:70], clay_labels[:70]),
    axis=0,
)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

y_train = tf.keras.utils.to_categorical(y_train, 4)
y_val = tf.keras.utils.to_categorical(y_val, 4)


# In[24]:


import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    volume = tf.repeat(volume, 3, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    volume = tf.repeat(volume, 3, axis=3)
    return volume, label


# In[25]:


# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 5
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train)).map(train_preprocessing).batch(batch_size)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)


# In[27]:


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 3))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=4, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()


# In[28]:


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 2
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


# In[26]:


import matplotlib.pyplot as plt

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)

print(train_dataset)
# plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")


# In[ ]:
