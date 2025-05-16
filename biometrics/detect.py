import cv2
import argparse
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np

# Model definition
class AgeGenderResNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features).squeeze(1)
        gender = self.gender_head(features)
        return age, gender

# Face detection
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    h, w = frame.shape[:2]
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

# Face processing
def predict_face(model, device, transform, frame, box, padding=20):
    x1, y1, x2, y2 = box
    y1 = max(0, y1 - padding)
    y2 = min(frame.shape[0] - 1, y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(frame.shape[1] - 1, x2 + padding)
    face = frame[y1:y2, x1:x2]

    if face.shape[0] < 20 or face.shape[1] < 20:
        print("Face too small, skipping...")
        return None, None

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_pil = face_pil.resize((224, 224))  # Resize to ResNet18 input
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            pred_age, pred_gender = model(face_tensor)
            age = int(pred_age.item())
            gender_idx = torch.argmax(pred_gender, dim=1).item()
            gender_map = {0: "Male", 1: "Female"}
            gender = gender_map[gender_idx]
            return age, gender
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None, None

# CLI argument
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file')
args = parser.parse_args()

# Paths
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "age_gender_resnet18.pth")
faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")

# Load face detector
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeGenderResNet().to(device)
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit(1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully.")

# Transform
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# Static image mode
if args.image:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not load image: {args.image}")
        exit(1)

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes:
        print("No face detected.")
    else:
        for box in faceBoxes:
            age, gender = predict_face(model, device, transform, frame, box)
            if age is not None and gender is not None:
                print(f"Gender: {gender}")
                print(f"Age: {age} years")
                cv2.putText(resultImg, f"{gender}, {age}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Detection", resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Webcam mode (replacing face detection logic)
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        exit(1)

    print("Press SPACE to capture image, ESC to quit.")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    captured = False
    result_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        display_frame = frame.copy()
        if captured:
            cv2.putText(display_frame, "Image captured! Press R to reset.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Press SPACE to Capture", display_frame)

        if captured and result_frame is not None:
            cv2.imshow("Prediction", result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            captured = True
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((224, 224))  # Resize input to 224x224
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_age, pred_gender = model(img_tensor)
                age = int(pred_age.item())
                gender_idx = torch.argmax(pred_gender, dim=1).item()
                gender_map = {0: "Male", 1: "Female"}
                gender = gender_map[gender_idx]

            print(f"Predicted Age: {age}")
            print(f"Predicted Gender: {gender}")

            result_frame = frame.copy()
            cv2.putText(result_frame, f"Age: {age}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(result_frame, f"Gender: {gender}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        elif key == ord('r') and captured:
            captured = False
            result_frame = None

    cap.release()
    cv2.destroyAllWindows()

