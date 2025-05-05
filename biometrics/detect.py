import cv2
import os
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# === Directory Setup ===
base_dir = os.path.dirname(__file__)
faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# === PyTorch Model Classes ===
# === PyTorch Model Classes ===
class GenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

class AgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        return self.model(x)


# === Load Model Functions ===
def load_caffe_model(prototxt_path, caffemodel_path):
    print(f"Loading Caffe model from: {prototxt_path}, {caffemodel_path}")
    model = cv2.dnn.readNet(caffemodel_path, prototxt_path)
    print(f"Model loaded successfully.")
    return model

def load_pytorch_model(model_class, path):
    model = model_class()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and not any(k.startswith("model.") for k in checkpoint.keys()):
        model.model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# === Load Models ===
models_dict = {
    'adience': {
        'gender': load_caffe_model(os.path.join(base_dir, 'gender_deploy.prototxt'),
                                   os.path.join(base_dir, 'gender_net.caffemodel')),
        'age': load_caffe_model(os.path.join(base_dir, 'age_deploy.prototxt'),
                                os.path.join(base_dir, 'age_net.caffemodel')),
    },
    'wikiset': {
        'gender': load_pytorch_model(GenderModel, os.path.join(base_dir, 'gender_model_final.pth')),
        'age': load_pytorch_model(AgeModel, os.path.join(base_dir, 'age_model_final.pth')),
    }
}

# === Class Labels ===
gender_classes = ['Male', 'Female']
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']

# === Face Detection ===
def get_face(frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []

    print(f"Total detections: {detections.shape[2]}")
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            faces.append((x1, y1, x2, y2))
            print(f"Face found: {x1}, {y1}, {x2}, {y2}, Confidence: {confidence}")
    return faces

# === Prediction Logic ===
def predict(image_b64, model_key='adience'):
    try:
        image_bytes = base64.b64decode(image_b64.split(',')[1])
        frame = np.array(Image.open(BytesIO(image_bytes)).convert('RGB'))
        print("Image decoded and converted to numpy array.")
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        return {'gender': 'Error decoding image', 'age': str(e)}

    faces = get_face(frame)
    if not faces:
        print("No faces detected in the image.")
        return {'gender': 'No face detected', 'age': 'No face detected'}

    x1, y1, x2, y2 = faces[0]
    face_crop = Image.fromarray(frame[y1:y2, x1:x2]).resize((224, 224))
    print(f"Face crop size: {face_crop.size}")

    # Debugging: Log the selected model
    print(f"Using {model_key} model for prediction.")  # Debugging line

    if model_key == 'adience':
        print("Using Adience model for prediction.")
        gender_model = models_dict['adience']['gender']
        age_model = models_dict['adience']['age']

        face_img_resized = cv2.resize(frame[y1:y2, x1:x2], (227, 227))
        input_blob = cv2.dnn.blobFromImage(
            face_img_resized, 1.0, (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746), swapRB=False
        )
        gender_model.setInput(input_blob)
        gender_preds = gender_model.forward()
        print(f"Gender predictions: {gender_preds}")
        gender_idx = np.argmax(gender_preds)

        age_model.setInput(input_blob)
        age_preds = age_model.forward()
        print(f"Age predictions: {age_preds}")
        age_idx = np.argmax(age_preds)

    elif model_key == 'wikiset':
        print("Using Wikiset model for prediction.")  # Debugging line
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(face_crop).unsqueeze(0)

        with torch.no_grad():
            gender_model = models_dict['wikiset']['gender']
            age_model = models_dict['wikiset']['age']

            gender_out = gender_model(tensor)
            age_out = age_model(tensor)
            gender_idx = torch.argmax(gender_out, dim=1).item()
            age_idx = torch.argmax(age_out, dim=1).item()
            print(f"Gender output: {gender_out}, Age output: {age_out}")
    else:
        print("Invalid model key provided.")
        return {'gender': 'Invalid model selected', 'age': 'Invalid model selected'}

    print(f"Predicted gender: {gender_classes[gender_idx]}, Predicted age: {age_classes[age_idx]}")
    
    return {
        'gender': gender_classes[gender_idx],
        'age': age_classes[age_idx]
    }
