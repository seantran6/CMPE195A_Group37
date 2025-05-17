import cv2
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import argparse

# -------------------------------
# Model Definition
# -------------------------------
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

# -------------------------------
# Helper Functions
# -------------------------------
def load_model(model_path: str, device) -> nn.Module:
    model = AgeGenderResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

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

def predict_face(model, device, transform, frame, box, padding=20):
    x1, y1, x2, y2 = box
    y1 = max(0, y1 - padding)
    y2 = min(frame.shape[0] - 1, y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(frame.shape[1] - 1, x2 + padding)
    face = frame[y1:y2, x1:x2]

    if face.shape[0] < 20 or face.shape[1] < 20:
        return None, None

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize((224, 224))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_age, pred_gender = model(face_tensor)
        age = int(pred_age.item())
        gender_idx = torch.argmax(pred_gender, dim=1).item()
        gender_map = {0: "Male", 1: "Female"}
        return age, gender_map[gender_idx]

def map_age_to_range(age: int) -> str:
    if age < 10:
        return "0-9"
    elif age < 18:
        return "10-17"
    elif age < 25:
        return "18-24"
    elif age < 33:
        return "25-32"
    elif age < 41:
        return "33-40"
    elif age < 51:
        return "41-50"
    elif age < 61:
        return "51-60"
    else:
        return "60+"

# -------------------------------
# API Function
# -------------------------------
def detect_traits(image_path: str) -> Tuple[str, str]:
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "age_gender_resnet18.pth")
    faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    transform = ResNet18_Weights.DEFAULT.transforms()

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image: {image_path}")

    _, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        raise ValueError("No face detected.")

    age, gender = predict_face(model, device, transform, frame, faceBoxes[0])
    if age is None or gender is None:
        raise RuntimeError("Model inference failed.")

    return map_age_to_range(age), gender.lower()

# -------------------------------
# Main (CLI & Webcam)
# -------------------------------
def run_cli_or_webcam():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image file')
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "age_gender_resnet18.pth")
    faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    transform = ResNet18_Weights.DEFAULT.transforms()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not load image: {args.image}")
            return

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected.")
        else:
            for box in faceBoxes:
                age, gender = predict_face(model, device, transform, frame, box)
                if age is not None and gender is not None:
                    print(f"Gender: {gender}, Age: {age}")
                    cv2.putText(resultImg, f"{gender}, {age}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Result", resultImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
            return

        print("Press SPACE to capture image, ESC to quit.")
        captured = False
        result_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            if captured:
                cv2.putText(display_frame, "Image captured! Press R to reset.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Live", display_frame)
            if captured and result_frame is not None:
                cv2.imshow("Prediction", result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                captured = True
                faceBoxes = highlightFace(faceNet, frame)[1]
                if faceBoxes:
                    age, gender = predict_face(model, device, transform, frame, faceBoxes[0])
                    print(f"Predicted Age: {age}, Gender: {gender}")
                    result_frame = frame.copy()
                    cv2.putText(result_frame, f"Age: {age}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(result_frame, f"Gender: {gender}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    print("No face detected.")
            elif key == ord('r') and captured:
                captured = False
                result_frame = None

        cap.release()
        cv2.destroyAllWindows()

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    run_cli_or_webcam()
