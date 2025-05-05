import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from recommendation import get_tracks_for_demographic
from biometrics.detect import predict  # Your model selection logic lives here!

load_dotenv()

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

app = Flask(__name__)

# === Load Face Detection Model ===
base_dir = os.path.dirname(__file__)
faceProto = os.path.join(base_dir, "biometrics", "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "biometrics", "opencv_face_detector_uint8.pb")
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# === Helper: Detect Faces ===
def highlightFace(net, frame, conf_threshold=0.7):
    frame_copy = frame.copy()
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(height / 150)), 8)

    return frame_copy, faceBoxes

# === Process Image + Predict ===
def process_image(image_data, model_key='adience'):
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(BytesIO(img_data)).convert('RGB')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    resultImg, faceBoxes = highlightFace(faceNet, img)

    if faceBoxes:
        print(f"[INFO] Face detected. Using model: {model_key}")
        result = predict(image_data, model_key=model_key)
        gender, age = result['gender'], result['age']
    else:
        print("[WARNING] No face detected in the image.")
        gender, age = "Unknown", "Unknown"

    return resultImg, gender, age, bool(faceBoxes)

# === ROUTES ===
@app.route('/')
def root():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image_data = data.get('image')
        model_key = data.get('model', 'adience').lower()

        print(f"[DEBUG] Incoming request to /recognize with model: {model_key}")

        if not image_data:
            print("[ERROR] No image data received.")
            return jsonify({'error': 'No image data provided'}), 400
        if model_key not in ['adience', 'wikiset']:
            print(f"[ERROR] Invalid model key received: {model_key}")
            return jsonify({'error': 'Invalid model selected'}), 400

        resultImg, gender, age, has_face = process_image(image_data, model_key)
        _, buffer = cv2.imencode('.jpg', resultImg)
        result_image = base64.b64encode(buffer).decode('utf-8')

        response = {
            'gender': gender,
            'age': age,
            'image': result_image,
            'tracks': get_tracks_for_demographic(age, gender, n=3) if has_face else [],
        }

        if not has_face:
            response['error'] = 'No face detected'

        print(f"[SUCCESS] Responding with gender: {gender}, age: {age}")
        return jsonify(response)

    except Exception as e:
        print(f"[EXCEPTION] Error in /recognize: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/recommend_tracks')
def recommend_tracks():
    age = request.args.get('age')
    gender = request.args.get('gender')
    n = int(request.args.get('n', 3))
    return jsonify({'tracks': get_tracks_for_demographic(age, gender, n=n)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
