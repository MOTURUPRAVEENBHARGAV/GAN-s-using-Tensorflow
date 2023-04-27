from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from PIL import Image
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from rough_gen import noise_generator


app = Flask(__name__)

model = tf.saved_model.load("/home/ubuntu/Generated_1000")
rough_model = noise_generator()
  
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/submit', methods=['POST'])
def upload_image():

    if request.method == 'POST':
        
        upload = os.path.join('static', 'uploads')
        img = os.path.join('static','image')
        global size
        size= int(request.form["noise_size"])

        #GENERATING NOISE IMAGES
        plt.figure(figsize=(8,8))

        for i in range(0,size):
            noise = tf.random.normal([1, 100]) #latent space
            generated_image = rough_model(noise, training=False)

            plt.subplot(4, 4, i+1)
            plt.imshow(generated_image[0, :, :, 0])
            plt.axis('off')
            plt.tight_layout()
        file = os.path.join(upload,'noise.jpg')
        plt.savefig(file)
        
        

        ## PREDICTING IMAGES FROM NOISE

        noise = tf.random.normal([size, 100])
        generated_images = model(noise, training=False)
        generated_images = (generated_images * 127.5) + 127.5
        for i in range(generated_images.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_images[i].numpy().astype('int'))
            plt.axis('off')
            plt.tight_layout()

        pred_path = os.path.join(img,'predicted.jpg')
        plt.savefig(pred_path)
   


      
        return render_template('index.html', image=file)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
# @app.route('/display/<filename>')
# def display_image(filename):
#     #print('display_image filename: ' + filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/predict' ,methods=['POST'])
def result():
     if request.method == 'POST':
        img = os.path.join('static','image')
        pred_path = os.path.join(img,'predicted.jpg')
        upload = os.path.join('static', 'uploads')
        file = os.path.join(upload, 'noise.jpg')

        return render_template('index.html',predicted=pred_path, image=file, Negative= f"Uploaded and Predicted Successfully!! 😃")
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)