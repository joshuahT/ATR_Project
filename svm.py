# %%
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize

from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV,cross_val_score,HalvingGridSearchCV 
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.experimental import enable_halving_search_cv

import os


import numpy as np
import nibabel as nib

import pandas as pd



# %%
def process_img(filepath, resize_l=250, resize_w=250, resize_d=64):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    
    # scan = resize(scan, output_shape=(resize_l, resize_w, resize_d))
    # scan = hog(scan, orientations=16, pixels_per_cell=(20,20), cells_per_block=(1,1), channel_axis=-1)
    return scan

def create_padding(filepath, x=378, y=335, z=297):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    final_shape = [x,y,z]
    padding = []
    for i, dim in enumerate(final_shape):
        padding += [(0, dim - scan.shape[i])]

    return np.pad(scan, padding, mode='constant')
def process_with_pad(filepath):
    img = create_padding(filepath)
    img = hog(img, orientations=16, pixels_per_cell=(20,20), cells_per_block=(1,1), channel_axis=-1)
    return img

def process_with_aug(filepath, resize_l=250, resize_w=250, resize_d=64):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    
    scan = resize(scan, output_shape=(resize_l, resize_w, resize_d))
    aug1 = [ele.T for ele in scan]
    aug2 = [ele.T for ele in aug1]
    scan = hog(scan, orientations=16, pixels_per_cell=(20,20), cells_per_block=(1,1), channel_axis=-1)
    aug1 = hog(aug1, orientations=16, pixels_per_cell=(20,20), cells_per_block=(1,1), channel_axis=-1)
    aug2 = hog(aug2, orientations=16, pixels_per_cell=(20,20), cells_per_block=(1,1), channel_axis=-1)
    return [scan, aug1, aug2]


# %%
X = np.empty(shape=(1005,4608))
Y = []

j = 0
for label in os.listdir(data_folder):
    img_directory = f'{data_folder}/{label}/'
    for img in os.listdir(img_directory): 
        img_path = '{}{}'.format(img_directory, img)
        img = process_with_pad(img_path)
        
        X[j] = img
        Y += [label]        
        j += 1

        

data_folder = f'{os.getcwd()}/labeled_data/data/'

        

# %%
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=777)
X, Y = shuffle(X, Y, random_state=0)

# %%
# classifier = SVC(kernel='poly', degree=3, gamma='scale', probability=True, class_weight='balanced')
params = {
    'C': np.logspace(-4, 4, 10),
    # 'degree': [2,3,4],
    'gamma': np.logspace(0,1, 5),
    'kernel': ['rbf'],
    'coef0': np.logspace(-4, 4, 10),
    # 'class_weight': [{0: 1, 1: w, 2: w, 3: w} for w in [2,3,4,5,6]]
}

classifier = SVC()
classifier = GridSearchCV(classifier, params,  cv=5, n_jobs=-1)
scores = cross_val_score(classifier, X, Y, cv=5)
print(scores)
print(classifier.best_estimator_)