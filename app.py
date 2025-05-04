import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from io import BytesIO
from PIL import Image
from recommendation import get_tracks_for_demographic
from dotenv import load_dotenv
load_dotenv()

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

app = Flask(__name__)

# Load pre-trained models
base_dir = os.path.dirname(__file__)

# Face-detection model files
faceProto  = os.path.join(base_dir, "biometrics", "opencv_face_detector.pbtxt")
faceModel  = os.path.join(base_dir, "biometrics", "opencv_face_detector_uint8.pb")

# Gender-detection model files (can be replaced with your Adience/WikiSet models)
genderProto = os.path.join(base_dir, "biometrics", "gender_deploy.prototxt")
genderModel = os.path.join(base_dir, "biometrics", "gender_net.caffemodel")

# Age-detection model files (can be replaced with your Adience/WikiSet models)
ageProto  = os.path.join(base_dir, "biometrics", "age_deploy.prototxt")
ageModel  = os.path.join(base_dir, "biometrics", "age_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# === Custom model loading functions ===
from detect import predict  # Custom model inference logic

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    
    return frameOpencvDnn, faceBoxes

def process_image(image_data, model_key='adience'):
    img_data = base64.b64decode(image_data.split(',')[1])  # Remove "data:image/jpeg;base64,"
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    resultImg, faceBoxes = highlightFace(faceNet, img)

    gender = "Unknown"
    age = "Unknown"

    if faceBoxes:
        face = resultImg[max(0, faceBoxes[0][1]):min(faceBoxes[0][3], resultImg.shape[0] - 1),
                         max(0, faceBoxes[0][0]):min(faceBoxes[0][2], resultImg.shape[1] - 1)]

        # Call to the custom prediction function based on selected model (Adience or WikiSet)
        result = predict(image_data, model_key=model_key)
        gender = result['gender']
        age = result['age']

    return resultImg, gender, age, bool(faceBoxes)

@app.route('/')
def root():
    return redirect(url_for('home'))

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    image_data = data['image']
    model_key = data.get('model', 'adience').lower()  # Default to 'adience'

    # Validate model selection
    if model_key not in ['adience', 'wikiset']:
        return jsonify({'error': 'Invalid model selected'}), 400

    # Process image
    resultImg, gender, age, has_face = process_image(image_data, model_key=model_key)

    # Get tracks if face was detected
    tracks = get_tracks_for_demographic(age, gender, n=3) if has_face else []

    # Encode the processed image
    _, buffer = cv2.imencode('.jpg', resultImg)
    result_image = base64.b64encode(buffer).decode('utf-8')

    response = {
        'gender': gender,
        'age': age,
        'image': result_image,
        'tracks': tracks
    }

    if not has_face:
        response['error'] = 'No face detected'

    return jsonify(response)

@app.route('/recognition', methods=['GET'])
def recognition():
    return render_template('recognition.html')

@app.route('/home')
def home():
    return render_template('index.html')  # Change this to index.html instead of recognition.html

@app.route('/recommend_tracks')
def recommend_tracks():
    age = request.args.get('age')
    gender = request.args.get('gender')
    n = int(request.args.get('n', 3))
    tracks = get_tracks_for_demographic(age, gender, n=n)
    return jsonify({'tracks': tracks})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
