import sys
import os
import pickle
import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, render_template, flash, request, redirect, url_for
import time
import h5py
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.debug = True

# Load models
log_pickle_model = pickle.load(open("log_model.sav", 'rb'))
sift = cv2.SIFT_create()
size = 128
intensity_model = load_model('squeezenet.h5')

# Upload folder config
UPLOAD_FOLDER = './static/inputimages'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- ROUTES ---------------- #

@app.route("/")
def hello():
    return render_template('home.html')

@app.route("/about/")
def about():
    return render_template('about.html')

@app.route("/main/")
def main():
    return render_template('main.html')

@app.route('/upload_file/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return render_template('main.html', user_image=filename)

@app.route('/prediction/<file>', methods=['GET', 'POST'])
def prediction(file):
    result1 = ""
    result2 = ""

    image_path = "./static/inputimages/" + file
    image = cv2.imread(image_path, 0)

    image_test = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    glcm_test = []
    images_sift_test = []

    img_arr_test = np.array(image_test)

    # GLCM Features
    gCoMat = graycomatrix(img_arr_test, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(gCoMat, prop='contrast')[0][0]
    dissimilarity = graycoprops(gCoMat, prop='dissimilarity')[0][0]
    homogeneity = graycoprops(gCoMat, prop='homogeneity')[0][0]
    energy = graycoprops(gCoMat, prop='energy')[0][0]
    correlation = graycoprops(gCoMat, prop='correlation')[0][0]

    # SIFT Features
    keypoints, descriptors = sift.detectAndCompute(image_test, None)

    if descriptors is None:
        descriptors = np.zeros((1, 2304))

    descriptors = descriptors.flatten()

    glcm_test.append([contrast, dissimilarity, homogeneity, energy, correlation])
    glcm_test = np.array(glcm_test)

    images_sift_test.append(descriptors[:2304])
    images_sift_test = np.array(images_sift_test)

    # Combine features
    features = np.concatenate((images_sift_test, glcm_test), axis=1)

    # Prediction
    if log_pickle_model.predict(features) == 1:
        result1 = "Cataract detected"

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        sensitivity = 156
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.5, 100000,
                                   param1=80, param2=40, minRadius=0, maxRadius=0)

        x, y, r = 0, 0, 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]

        mask = np.zeros((224, 224), np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), (255, 255, 255), -1)

        masked_data = cv2.bitwise_and(img, img, mask=mask)

        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        cnt = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        x, y, w, h = cv2.boundingRect(cnt[0])

        crop = masked_data[y:y + h, x:x + w]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        my_image = img_to_array(crop)
        my_image = my_image.reshape((1, 224, 224, 3))

        ans = intensity_model.predict(my_image)
        ans_class = np.argmax(ans)

        classes = ["Mild Cataract", "Normal Cataract", "Severe Cataract"]
        result2 = classes[ans_class]

    else:
        result1 = "No Cataract"
        result2 = "Normal Eye"

    return redirect(url_for('results', result1=result1, result2=result2))

@app.route('/result/<result1>/<result2>', methods=['GET', 'POST'])
def results(result1, result2):
    return render_template('result.html', result1=result1, result2=result2)

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)