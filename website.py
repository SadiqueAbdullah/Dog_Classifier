# Importing required libraries_Udacity
# Libraries for Web app
import flask
import numpy as np
from flask import request, render_template, Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# Libraries for ML

# Acceptable files
FILE_TYPE = {'jpg', 'jpeg', 'png', 'gif'}

# initialize our Flask application and the Keras model
app = Flask(__name__)
# model = load_model('trained_saved_model.h5')
model = VGG16()


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    pred = model.predict(image)
    label = decode_predictions(pred)
    label = label[0][0]
    classification = '%s (%.1f%%)' % (label[1], label[2] * 100)

    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
