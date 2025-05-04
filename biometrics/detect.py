import cv2
import torch
import os
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from io import BytesIO
import base64

# === Load face detector ===
base_dir = os.path.dirname(__file__)
faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# === Preprocessing function ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Imagenet stats
                         [0.229, 0.224, 0.225])
])

# === Load models (Adience and WikiSet variants) ===
def load_model(path, num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

models_dict = {
    'adience': {
        'gender': load_model('models/adience_gender_model.pth', 2),
        'age': load_model('models/adience_age_model.pth', 8)
    },
    'wikiset': {
        'gender': load_model('models/wiki_gender_model.pth', 2),
        'age': load_model('models/wiki_age_model.pth', 8)
    }
}

age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
gender_classes = ['Male', 'Female']

# === Face detection ===
def get_face(frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            faces.append((x1, y1, x2, y2))
    return faces

# === Inference ===
def predict(image_b64, model_key='adience'):
    # Decode image
    image_data = image_b64.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image_np = np.array(Image.open(BytesIO(image_bytes)).convert('RGB'))
    frame = image_np.copy()

    # Face detection
    faces = get_face(frame)
    if not faces:
        return {'gender': 'Unknown', 'age': 'Unknown'}

    x1, y1, x2, y2 = faces[0]  # Only use first face
    face_img = Image.fromarray(frame[y1:y2, x1:x2])

    # Preprocess
    input_tensor = preprocess(face_img).unsqueeze(0)

    # Load correct models
    gender_model = models_dict[model_key]['gender']
    age_model = models_dict[model_key]['age']

    # Predict
    with torch.no_grad():
        gender_idx = torch.argmax(gender_model(input_tensor)).item()
        age_idx = torch.argmax(age_model(input_tensor)).item()

    return {
        'gender': gender_classes[gender_idx],
        'age': age_classes[age_idx]
    }
