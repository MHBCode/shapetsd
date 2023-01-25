import time
from itsdangerous import base64_decode
import requests
#import pymongo
import requests_cache
import numpy as np
from PIL import Image
from csv import DictWriter
import pandas as pd
import base64
from geopy.geocoders import GoogleV3

from datetime import datetime

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.utils import load_img
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Key.json"
"""
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
dblist = myclient.list_database_names()

mydb = myclient["mydatabase"]
mycol = mydb["Vehicles"]
"""
model = load_model('final_model.h5')
import re
from io import BytesIO, StringIO
from flask import Flask, render_template, request, jsonify

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

requests_cache.install_cache('github_cache', backend='sqlite', expire_after=180)


def load_image(filename):
 # load the image
 img = load_img(filename, target_size=(32, 32))
 # convert to array
 img = img_to_array(img)
 # reshape into a single sample with 3 channels
 img = img.reshape(1, 32, 32, 3)
 # prepare pixel data
 img = img.astype('float32')
 img = img / 255.0
 return img




@app.route('/shape', methods=['GET', 'POST'])
def shape():
    return render_template('shape.html')

@app.route('/cubes', methods=['GET', 'POST'])
def cubes():
    return render_template('cubes.html')

@app.route('/', methods=['GET', 'POST'])
def admin():
    return render_template('Home.html')

@app.route('/home', methods=['GET', 'POST'])
def admins():
    
    return render_template('Home.html')



@app.route('/hook2', methods=['POST'])
def get_image2():
    #print(request.form)
    try:
        shapefind=request.form['shape']
        image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
        im=Image.open(BytesIO(base64.b64decode(image_data)))
        date_string = str(time.strftime("%Y-%m-%d-%H:%M"))
        name="./static/Pictures/"+date_string.replace(":","")+".png"
        name2="Pictures/"+date_string.replace(":","")+".png"
        im.save(name, format="png")
        img = cv2.imread(name)

        img = load_image(name)
        # load model
        
        # predict the class
        result = list(model.predict(img)[0])
        print()
        labels={0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"}
        #print()

        pred=labels[result.index(max(result))]
        os.remove(name)
        if shapefind==pred:
            results = {'output': True}
            return jsonify(results)
        else:
            results = {'output': False}
            return jsonify(results)
        
        #os.remove(name2)

        #return jsonify(results)
    except Exception as e:
        print(e)



@app.route('/hook', methods=['POST'])
def get_image():
    #print(request.form)
    try:
        shapefind=request.form['shape']
        image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
        im=Image.open(BytesIO(base64.b64decode(image_data)))
        date_string = str(time.strftime("%Y-%m-%d-%H:%M"))
        name="./static/Pictures/"+date_string.replace(":","")+".png"
        name2="Pictures/"+date_string.replace(":","")+".png"
        im.save(name, format="png")
        img = cv2.imread(name)

        # converting image into grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # setting threshold of gray image
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # using a findContours() function
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 0

        # list for storing names of shapes
        shape="None"

        for contour in contours:


            if i == 0:
                i = 1
                continue


            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
            
            M = cv2.moments(contour)
            x,y,w,h = cv2.boundingRect(approx)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
            print("===================>",len(approx))
            if len(approx) == 3:
                shape="triangle"
                cv2.putText(img, 'Triangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                print("Triangle detected")
                
            elif len(approx)==4:
                        aspectRatio = float(w)/float(h)
                        if aspectRatio > 0.95 and aspectRatio < 1.05:
                            shape = 'square'
                            cv2.putText(img, 'Square', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            print("Square")
                        else:
                            shape = "rectangle"
                            cv2.putText(img, 'Rectangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            print("Rectangle")

            elif len(approx)>7:
                shape="circle"
                cv2.putText(img, 'circle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                print("circle detected")
            else:
                print("Nothing Detected")
            if shape==shapefind:
                print(shape,shapefind)
                # os.remove(name)
                results = {'output': True}
                return jsonify(results)
        # displaying the image after drawing contours
        #cv2.imshow('shapes', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()






        
        # os.remove(name)
        #os.remove(name2)
        results = {'output': False}

        return jsonify(results)
    except Exception as e:
        print(e)
    
if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=8080)
    app.run(debug=True)
