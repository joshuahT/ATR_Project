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


def serialize(x, y):
    x = tf.expand_dims(x, axis=-1)
    x = tf.repeat(x, 3, axis=-1)
    x_shape = x.shape
    x_bytes = x.numpy().tobytes()
    
    y_bytes = tf.io.serialize_tensor(y).numpy()
    return tf.train.Example(features=tf.train.Features(feature={
        'x_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x_shape)),
        'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_bytes])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_bytes]))
    }))



def process_img(path, desired_height = 128,  desired_width = 128, desired_depth = 64):
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
Y = np.empty(shape=(length), dtype=np.float64)

j = 0
print('Loading Images')
for label in os.listdir(data_folder):
    img_directory = f'{data_folder}/{label}/'
    for img in os.listdir(img_directory): 
        if j % 100 == 0:
            print('Loading Image #', j)
        X[j] = process_img(img_directory + '/' + img, resize_h, resize_w, resize_d)
        Y[j] = int(label)
        j += 1
        
    

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


def write_to_record(x_data, y_data, filename):
    print(f'Generating {filename}')
    y_data = tf.one_hot(y_data, depth=4, axis=1)
    with tf.io.TFRecordWriter(f'{filename}') as writer:
        for i in range(len(y_data)):
            if i % 100 == 0:
                print(f'{i} / {len(y_data)}')
            entry = serialize(x_data[i], y_data[i])
            writer.write(entry.SerializeToString())


write_to_record(x_train, y_train, 'training_dataset.tfrecord')
print('done')
write_to_record(x_test, y_test, 'testing_dataset.tfrecord')





