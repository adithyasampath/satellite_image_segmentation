#keras utils
import keras
from keras.models import load_model
from keras.optimizers import Adam
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.applications import imagenet_utils
from keras.layers import Dense, Input, Dropout
from keras.models import Model
#image utils
from skimage.transform import resize
from skimage.io import imread
import numpy as np
#FLASK REST utils
from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
#misc
import os
import sys

#init flask
app = Flask(
    __name__,
    static_url_path="/images",
    static_folder="images",
    template_folder='templates')
CORS(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
valid_mimetypes = ['image/jpeg', 'image/png']


#helper functions
def load_image(filename):
    x = imread(filename)
    print(x.shape)
    x = resize(
        x, (96, 96), mode='constant', anti_aliasing=False)
    x = np.stack((x, ) * 3, axis=-1)
    x =  x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x

#init model
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

keras.backend.clear_session()
model = load_model(os.path.join(APP_ROOT, "weights", "resnet50.h5"))
model._make_predict_function()

def predict_function(image,model):
    np_image = load_image(image)
    feature = model.predict(np_image)
    prediction = labels[np.argmax(feature)]
    return prediction

#flask functions
@app.route('/')
def index():
    return render_template("upload.html")


@app.route('/getImage', methods=["POST"])
def upload():
    global model
    image_folder = os.path.join(APP_ROOT, "images")
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    if request.method == 'POST':
        if not 'file' in request.files:
            return jsonify({'error': 'no file'}), 400
        img_file = request.files['file']
        img_name = img_file.filename
        print(img_name)
        img_path = os.path.join("images", img_name)
        mim_type = img_file.content_type
        if not mim_type in valid_mimetypes:
            return jsonify({'error': 'bad file type'}), 400
        img_file.save(img_path)
        prediction = predict_function(img_path,model)
        return render_template(
            "complete.html", image_name=img_path, class_name=prediction)


if __name__ == "__main__":
    PORT = 5000
    if len(sys.argv) > 1:
        PORT = sys.argv[1]
    app.run(host='0.0.0.0', port=PORT)
