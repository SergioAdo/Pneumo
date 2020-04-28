
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import keras as ks
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from keras.models import load_model
from keras.preprocessing import image

from keras.backend import set_session


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import json

app = Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model = tf.keras.models.load_model('cnn_10ep_87.h5', compile=False)

model.summary()


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        f = request.files['file']
        img = tf.keras.preprocessing.image.load_img(f, target_size=(128, 128))
        img = np.expand_dims(img, axis=0)
        # On predit
        result = (model.predict(img)*100).tolist()

        print("Sain:", result[0][0], " Affecte:", result[0][1])

        return json.dumps(result)



if __name__ == '__main__':
    app.run(debug=True)
