import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import os
from io import BytesIO
from PIL import Image
import traceback

from recommendation import get_tracks_for_demographic
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="178a1cf820ed475b90627b08a7ee3fcb",
    client_secret="9f85798b2f4b44ff911a7c56fa8319f9"
))


import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API Auth - make sure these environment variables are set!
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

# Flask App Init
app = Flask(__name__)

# Paths
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "biometrics", "age_gender_resnet18.pth")
faceProto = os.path.join(base_dir, "biometrics", "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "biometrics", "opencv_face_detector_uint8.pb")

# Load OpenCV Face Detector
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# PyTorch Age-Gender Model Definition
class AgeGenderResNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        age_pred = self.age_head(features).squeeze(1)
        gender_pred = self.gender_head(features)
        return age_pred, gender_pred

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model Weights
print(f"Loading model from: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = AgeGenderResNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image Preprocessing
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# Face Detection
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    h, w = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), max(1, int(round(h / 150))), 8)
    return frameOpencvDnn, faceBoxes

# Image Processing
def process_image(image_data):
    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        resultImg, faceBoxes = highlightFace(faceNet, img)

        gender = "Unknown"
        age = "Unknown"

        if faceBoxes:
            x1, y1, x2, y2 = faceBoxes[0]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(resultImg.shape[1] - 1, x2), min(resultImg.shape[0] - 1, y2)

            if x2 > x1 and y2 > y1:
                face = resultImg[y1:y2, x1:x2]
                print("Detected face shape:", face.shape)

                if face.shape[0] < 20 or face.shape[1] < 20:
                    raise ValueError("Detected face is too small for model.")

                face_tensor = transform(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

                with torch.no_grad():
                    age_pred, gender_pred = model(face_tensor)
                    age_val = int(round(age_pred.item()))
                    gender_idx = torch.argmax(gender_pred, dim=1).item()

                gender = "Male" if gender_idx == 0 else "Female"
                age = str(age_val)

                cv2.rectangle(resultImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resultImg, f"{gender}, {age}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return resultImg, gender, age
    except Exception as e:
        print("Error in process_image:", e)
        traceback.print_exc()
        return np.zeros((480, 640, 3), dtype=np.uint8), "Unknown", "Unknown"

# Routes
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    image_data = data['image']
    resultImg, gender, age_str = process_image(image_data)

    # Convert age string to int, fallback to 25 if invalid
    try:
        age_int = int(age_str)
    except ValueError:
        age_int = 25

    # Convert int age to age label format used by your recommendation
    if age_int < 11:
        age_label = "0-10"
    elif age_int < 21:
        age_label = "11-20"
    elif age_int < 31:
        age_label = "21-30"
    elif age_int < 41:
        age_label = "31-40"
    elif age_int < 51:
        age_label = "41-50"
    elif age_int < 61:
        age_label = "51-60"
    else:
        age_label = "61-100"

    # Get recommended tracks (e.g., 6 tracks, shuffled)
    tracks = get_tracks_for_demographic(age_label, gender, n=6, shuffle_result=True)

    # Encode result image to base64 string for sending back to client
    _, buffer = cv2.imencode('.jpg', resultImg)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'gender': gender,
        'age': age_int,
        'image': result_image,
        'tracks': tracks
    })

@app.route('/recommend_tracks')
def recommend_tracks():
    age = request.args.get('age', '25')
    gender = request.args.get('gender', 'Unknown')
    n = request.args.get('n', 6)

    try:
        age_int = int(age)
        n = int(n)
    except ValueError:
        age_int = 25
        n = 6

    # Convert int age to age label format
    if age_int < 11:
        age_label = "0-10"
    elif age_int < 21:
        age_label = "11-20"
    elif age_int < 31:
        age_label = "21-30"
    elif age_int < 41:
        age_label = "31-40"
    elif age_int < 51:
        age_label = "41-50"
    elif age_int < 61:
        age_label = "51-60"
    else:
        age_label = "61-100"

    tracks = get_tracks_for_demographic(age_label, gender, n=n)
    return jsonify({'tracks': tracks})
# Run App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
