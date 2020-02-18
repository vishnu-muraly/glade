# Glass defect

from load import *
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
#from scipy.misc import imsave, imread, imresize


# for matrix math
import numpy as np
import argparse
import sys
import time

from werkzeug import secure_filename
# for importing our keras model
#import keras.models
import tensorflow as tf
# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os
# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model, graph, label
# initialize these variables
#model,graph = init()

# returns the graph model of the file

model_file = "data/retrained_graph.pb"
label_file = "data/retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"
input_name = "import/" + input_layer
output_name = "import/" + output_layer


graph, label = init(model_file, label_file)

input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)


photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)


@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST' and 'photo' in request.files:

        model_file = "data/retrained_graph.pb"
        label_file = "data/retrained_labels.txt"
        input_height = 224
        input_width = 224
        input_mean = 128
        input_std = 128
        input_layer = "input"
        output_layer = "final_result"
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer

        graph, label = init(model_file, label_file)

        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        filename = photos.save(request.files['photo'])
        os.rename('./'+filename, './'+'output.png')

        # print('debug')

        # read the image into memory

        t = read_tensor_from_image_file('./output.png', input_height=input_height,
                                        input_width=input_width, input_mean=input_mean, input_std=input_std)

        with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            end = time.time()

        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]

        #labels = load_labels(label_file)

        #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
        #template = "{} (score={:0.5f})"
        name = []
        text = []
        for i in top_k:
            #print(template.format(labels[i], results[i]))
            name.append(label[i])  
            text.append(results[i])  

        return render_template("index2.html", s1=name[0], s2=text[0], s3=name[1], s4=text[1], s5=name[2], s6=text[2])


if __name__ == "__main__":

    model_file = "data/retrained_graph.pb"
    label_file = "data/retrained_labels.txt"

    # decide what port to run the app in
    port = int(os.environ.get('PORT', 8000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
    # optional if we want to run in debugging mode
    app.run(debug=True)
