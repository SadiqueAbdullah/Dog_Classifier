# Dog_Classifier
This Project is related to learning and implementing image classification using Deep neural network.
This project uses Convolutional Neural Networks (CNNs)! In this project, I built a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed out of roughly 133 known breeds. If supplied an image of a human, the code will identify the resembling dog breed.

#### The Process flow steps followed in the python Note book "dog_app.ipynb" are mentioned below;

Step 0: Import Datasets

Step 1: Detect Humans

Step 2: Detect Dogs

Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)

Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

Step 6: Write your Algorithm

Step 7: Test Your Algorithm

#### Some of the key libaries used are listed below along with the paths for the imported libraries;

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import random
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile

##### After training the pretrained Imagenet model by adding pooling layers, the accuracy of 82.16% was achieved. 

#### Key elements Web app for Dog breed classification:

1) Train.classifer.py: CNN model was trained and saved as "trained_saved_model.h5".
2) website.py : Flask app to predict the uploaded image as a dog breed classifier. VGG16 pretrained classifier was used for imgae classification.
3) index.html : This file is linked with flask app and coded in html which enables user to upload the image and then predict the dog breed with probablity numbers.

### Instructions for running/testing the Web app:

 ##### Step-1: run website.py file in the environment. It will show as "Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)"
 
 ![image](https://user-images.githubusercontent.com/77229486/126876946-df988830-9695-4124-b16a-746e3b64e15e.png)
 
 ##### Step-2: Click on "http://127.0.0.1:5000/" link or manually type the address in the browser. Following page will show up.
 
 ![image](https://user-images.githubusercontent.com/77229486/126877026-89a9d16f-4ccc-4df7-aaee-3e57de5677a4.png)

 ##### Step-3: Use "Choose File" button for uploading the image and "Predict" button" for prediction.
 
 ![image](https://user-images.githubusercontent.com/77229486/126877228-ef14e315-8060-482a-b716-edb71cfa16ae.png)
 
 ![image](https://user-images.githubusercontent.com/77229486/126877163-f7e17c5f-c71a-40f7-9327-36c1f34af903.png)

 
 

 


