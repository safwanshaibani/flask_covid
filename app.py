# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from keras_preprocessing.image.utils import load_img
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
import cv2
import numpy as np
import pandas as pd
import sys
import os
import glob
import re
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/Inception_model15_08.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()  
labels = ['Abnormal' , 'Normal']
#print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    
    img = image.load_img(img_path, target_size=(200, 200))
    
    # Preprocessing the image
    img = image.img_to_array(img)
    img = img.reshape(1, 200, 200, 3)
    img = img.astype('float32')/255.0
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)
    #x = np.vstack([x])

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(img) # , mode='caffe')

    preds = model.predict(img) #, batch_size=10)
    
    if preds[0][0] > 0.9:
        final = 'COVID-19 Negative'
        
    else:
        final = 'COVID-19 Positive'
        

    return final

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        ##pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = preds #str(pred_class[0][0][1])               # Convert to string
        return result
    return None


	#pred = labels[pred]

	#return render_template("prediction.html", data=pred)


if __name__ == "__main__":
	app.run(debug=True)