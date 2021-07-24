#!/usr/bin/env python
# coding: utf-8

# # Data Scientist Nanodegree
# ## Convolutional Neural Networks
# ## Project: Write an Algorithm for a Dog Identification App
# This notebook walks you through one of the most popular Udacity projects across machine learning and artificial intellegence nanodegree programs.  The goal is to classify images of dogs according to their breed.

# In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!).

# We break the notebook into separate steps.
# 
# * [Step 0](#step0): Import Datasets
# * [Step 1](#step1): Detect Humans
# * [Step 2](#step2): Detect Dogs
# * [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
# * [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
# * [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
# * [Step 6](#step6): Write your Algorithm
# * [Step 7](#step7): Test Your Algorithm

# ---

# Step 0: Import Datasets

# Import Dog Dataset
# 
# In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
# - `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
# - `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
# - `dog_names` - list of string-valued dog breed names for translating labels

# In[52]:
from cv2.cv2 import CascadeClassifier
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames']) 
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/dog_images/train')
valid_files, valid_targets = load_dataset('data/dog_images/valid')
test_files, test_targets = load_dataset('data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


# Import Human Dataset
# In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.

import random
random.seed(8675309)

# load file names in shuffled human dataset
human_files = np.array(glob("data/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

# Step 1: Detect Humans
# We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

import cv2
import matplotlib.pyplot as plt                        

# extract pre-trained face detector
face_cascade= cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[18])
print(img.shape)
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in image
faces = face_cascade.detectMultiScale(gray)
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

# Human face detector returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# Download 100 sample images for human & Dogs
human_files_short = human_files[:100]
dog_files_short = train_files[:100]

## Test the performance of the face_detector algorithm on the images in human_files_short and dog_files_short.

print(str(np.sum([face_detector(file) for file in human_files_short])) + '% of human images with a detected face')
print(str(np.sum([face_detector(file) for file in dog_files_short])) + '% of dog images with a detected face')

# Step 2: Detect Dogs

# In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.

from keras.applications.resnet50 import ResNet50
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# Pre-process the Data
# When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

# where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  
# 
# The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape
# Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!

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


#  Making Predictions with ResNet-50
# Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).
# Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.
# By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# Write a Dog Detector
# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

# Test the performance of the dog_detector function on the images in human_files_short and dog_files_short.
print(str(np.sum([dog_detector(file) for file in human_files_short])) + '% of detected dogs in human_files_short file')
print(str(np.sum([dog_detector(file) for file in dog_files_short])) + '% of detected dogs in dog_files_short file')

# Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
# We rescale the images by dividing every pixel in every image by 255.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

# Checking the shpe of train_tensors
train_tensors.shape

#  Model Architecture

# For the below executed CNN , the base architecture contains 3 convolutional and 3 max pooling layers paired together in series and arrange in alternate sequence. The Convolutional layer serves the purpose of deriving high level features through filters. Max pooling layer reduces the dimensionality and hence the execution time. Relu activation function was used in all the Convolutional layers.
# Total 5 convolutional layers were used with filters ranging from 16 to 256 in creasing order with multipe of 2 in consecutive layers.
# 1 Golbal average pooling layer was used before final layer to reduce the dimensionality of the CNN.
# 5 Max pooling layers were used with default setting and with a pool size of 2,2.
# For the last and fully connected layer the nodes were set to 133 along with a softmax activation function to obtain probabilities of the prediction of Dog breed.

# ##### Did some experiments below to test and improve Test accuracy. 
# CL : Count of Convolutional layer
# MP : count of Max Pooling layer
# FL : Count of Flatten layer
# DOR : Dropout rate

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

model = Sequential()

model.add(Conv2D(filters=16,kernel_size=3,padding='same',activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size= 2))


model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size= 2))


model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size= 2))

model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size= 2))

model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size= 2))

model.add(GlobalAveragePooling2D())  

          
model.add(Dropout(0.4))
model.add(Dense(133,activation='softmax'))
          
model.summary()


# ### Compile the Model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# ### Train the Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

### specify the number of epochs that you would like to use to train the model.
epochs = 30
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

# ### Load the Model with the Best Validation Loss
model.load_weights('saved_models/weights.best.from_scratch.hdf5')

# ### Test the Model
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# To reduce training time without sacrificing accuracy, we will train a CNN using transfer learning.
# Obtain Bottleneck Features

bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


# Model Architecture
# The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()

# Compile the Model
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Train the Model
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


# Load the Model with the Best Validation Loss
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
# Test the Model

# Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

#Predict Dog Breed with the Model

from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

# Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

# Obtain bottleneck features from another pre-trained CNN.( VGG19 Network)

bottleneck_features1 = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']

### Obtain bottleneck features from another pre-trained CNN. (Resnet50 Network)

bottleneck_features1 = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features1['train']
valid_Resnet50 = bottleneck_features1['valid']
test_Resnet50 = bottleneck_features1['test']

# Define architecture with VGG19 network.
VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
VGG19_model.add(Dense(133, activation='softmax'))
VGG19_model.summary()

# Define architecture with Resnet50 network.

Resnet50_model_new = Sequential()
Resnet50_model_new.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model_new.add(Dropout(0.4))
Resnet50_model_new.add(Dense(133, activation='softmax'))

Resnet50_model_new.summary()

# Compile the model with VGG19 network
VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Compile the model with Resnet50
Resnet50_model_new.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model (VGG19)
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.DogVGG19.hdf5', 
                               verbose=1, save_best_only=True)

VGG19_model.fit(train_VGG19, train_targets, 
          validation_data=(valid_VGG19, valid_targets),
          epochs=20, batch_size=30, callbacks=[checkpointer], verbose=1)

# Train the model (Resnet 50)
checkpointer1 = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', 
                               verbose=1, save_best_only=True)

Resnet50_model_new.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=20, batch_size=100, callbacks=[checkpointer1], verbose=1)

# Load the model weights with the best validation loss (VGG19 model)
VGG19_model.load_weights('saved_models/weights.best.DogVGG19.hdf5')

# Load the model weights with the best validation loss (Resnet50 model)
Resnet50_model_new.load_weights('saved_models/weights.best.Resnet50.hdf5')

# Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set--(VGG19 model)
VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

# Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set
Resnet50_predictions_new = [np.argmax(Resnet50_model_new.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Resnet50_predictions_new)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions_new)
print('Test accuracy: %.4f%%' % test_accuracy)

# Write a function that takes a path to an image as input
# and returns the dog breed that is predicted by the model.
def Resnet50_predict_breed(img_path):
    bottleneck_feature2 = extract_Resnet50(path_to_tensor(img_path))     # extract bottleneck featurescorresponding to the chosen CNN model.
    predicted_vector = Resnet50_model_new.predict(bottleneck_feature2)       # gets the prediction vector which gives the index of the predicted dog breed.
    return dog_names[np.argmax(predicted_vector)]                       # the o/p is, dog breed that is predicted by the model


# ---

# ## Step 6: Write your Algorithm
# 
# Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
# - if a __dog__ is detected in the image, return the predicted breed.
# - if a __human__ is detected in the image, return the resembling dog breed.
# - if __neither__ is detected in the image, provide output that indicates an error.

def display_image(img_path):
    image = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageplot = plt.imshow(cv_rgb)
    return imageplot

def predict_breed(img_path):
    display_image(img_path)
    if dog_detector(img_path):
        print("It's a dog!")
        return print("and it could be from {} Breed of the dog ".format(Resnet50_predict_breed(img_path)))

    if face_detector(img_path):
        print("It's a human!")
        return print("On a funny note ! It resembles to {} Breed of Dog !!".format(Resnet50_predict_breed(img_path)))

    else:
        return print("This seems to be neither dog nor human..It must be something else .")

# ---

# Step 7: Test Your Algorithm
# Execute the algorithm from Step 6 on
sample_files = np.array(glob("static/test_images/*"))
print(sample_files)
predict_breed('static/test_images/Dog Breed Border Colie.JPG')

predict_breed('static/test_images/American Curl Cat.JPG')

predict_breed('static/test_images/human2.JPG')

predict_breed('static/test_images/Dog toy.JPG')

predict_breed('static/test_images/human5.JPG')

predict_breed('static/test_images/human4.JPG')

predict_breed('static/test_images/Dog Breed Chow Chow.JPG')

predict_breed('static/test_images/Dog Breed Poodle.JPG')

predict_breed('static/test_images/human3.JPG')

predict_breed('static/test_images/human1.JPG')
