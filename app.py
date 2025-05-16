import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import os
from io import BytesIO
from PIL import Image
from recommendation import get_tracks_for_demographic
from dotenv import load_dotenv
load_dotenv()

import torch
import torchvision.transforms as transforms

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

print("=== DEBUG - importing app.py ===")
print("faceModel _before_ assignment =", globals().get("faceModel"))

app = Flask(__name__)

# Load face detection model (OpenCV DNN) as before
base_dir = os.path.dirname(__file__)
faceProto = os.path.join(base_dir, "biometrics", "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "biometrics", "opencv_face_detector_uint8.pb")
print("DEBUG - faceModel =", faceModel)
print("DEBUG - faceProto =", faceProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load your PyTorch age/gender model
class AgeGenderResNet(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(AgeGenderResNet, self).__init__()
        # Define your model architecture here matching your trained model
        # Example: Using pretrained ResNet18 backbone and two heads (age regression + gender classification)
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = torch.nn.Identity()  # Remove final fc layer

        # Age regression head (1 output)
        self.age_head = torch.nn.Linear(512, 1)
        # Gender classification head (2 outputs: male/female)
        self.gender_head = torch.nn.Linear(512, 2)

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features).squeeze(1)  # regression output
        gender = self.gender_head(features)       # classification logits
        return age, gender

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(base_dir, "biometrics", "age_gender_resnet18.pth")
model = AgeGenderResNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transforms (must match training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
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

def process_image(image_data):
    # Decode base64 image
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert PIL RGB to OpenCV BGR

    resultImg, faceBoxes = highlightFace(faceNet, img)

    gender = "Unknown"
    age = "Unknown"

    if faceBoxes:
        x1, y1, x2, y2 = faceBoxes[0]
        # Clamp coordinates inside image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(resultImg.shape[1] - 1, x2), min(resultImg.shape[0] - 1, y2)
        face = resultImg[y1:y2, x1:x2]

        # Preprocess face for model
        face_tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            age_pred, gender_pred = model(face_tensor)
            age_val = age_pred.item()
            gender_idx = torch.argmax(gender_pred, dim=1).item()

        gender = "Male" if gender_idx == 0 else "Female"
        age = f"{age_val:.1f}"

        # Draw box and label
        cv2.rectangle(resultImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resultImg, f"{gender}, {age}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return resultImg, gender, age

@app.route('/')
def index():
    return render_template('index.html')

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
    resultImg, gender, age = process_image(image_data)

    _, buffer = cv2.imencode('.jpg', resultImg)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'gender': gender, 'age': age, 'image': result_image})

@app.route('/recommend_tracks')
def recommend_tracks():
    age = request.args.get('age')
    gender = request.args.get('gender')
    n = int(request.args.get('n', 6))
    tracks = get_tracks_for_demographic(age, gender, n=n)
    return jsonify({'tracks': tracks})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
