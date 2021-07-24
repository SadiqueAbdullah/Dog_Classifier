#Loading the required libraries
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
import cv2
from sklearn.datasets import load_files
from keras.utils import np_utils
import re
import os
import pickle
import gc
from glob import glob


from keras import models
import numpy as np
from tensorflow.python.keras.applications.resnet import ResNet50

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import io

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint

from keras.applications.resnet import preprocess_input, decode_predictions

def load_dataset(path):
    """
       This function is to load the data from a given location(file-path) and
       data: The data frame containing the images
       dof_files: It contains a dog images from various breed. It is a predictor variable
       dog_targets: Target Variable. Here it's Breed of dog to be classified out of 133 categories.
       """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets



from pathlib import Path
dataset = Path(os.getcwd())
image_data = Path(os.path.join(dataset,'data/dog_images'))
train_data = Path(os.path.join(dataset,'data/dog_images/train'))
valid_data = Path(os.path.join(dataset,'data/dog_images/valid'))
test_data = Path(os.path.join(dataset,'data/dog_images/test'))
model_test_data = Path(os.path.join(dataset,'test_images'))

# load train, test, and validation datasets
train_files, train_targets = load_dataset(train_data)
valid_files, valid_targets = load_dataset(valid_data)
test_files, test_targets = load_dataset(test_data)

# load list of dog breed names
dog_names = [item[20:-1] for item in sorted(glob("../data/dog_images/train/*/"))]
#load human dataset
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("../../../data/lfw/*/*"))
random.shuffle(human_files)


# Pre-Process the data

from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

### Obtain bottleneck features from another pre-trained CNN. (Resnet50 Network)

bottleneck_features1 = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features1['train']
valid_Resnet50 = bottleneck_features1['valid']
test_Resnet50 = bottleneck_features1['test']

def build_model():
    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

# Build the Model
    def paths_to_tensor(img_paths):
        list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
    Resnet50_model.add(Dropout(0.4))
    Resnet50_model.add(Dense(133, activation='softmax'))

    # Compile the Model with Resnet50 network

    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return Resnet50_model

# Train the Resnet50 Model
def train_model(Resnet50_model):
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5',
                                   verbose=1, save_best_only=True)

    Resnet50_model.fit(train_Resnet50, train_targets,
                       validation_data=(valid_Resnet50, valid_targets),
                       epochs=25, batch_size=30, callbacks=[checkpointer], verbose=1)
    return Resnet50_model

def save_model(Resnet50_model):
    Resnet50_model.save("trained_saved_model.h5")


def main():
    myModel= None
    tf.keras.backend.clear_session()
    gc.collect()
    myModel = build_model()
    myModel = train_model(myModel)
    save_model(myModel)

main()

