import cv2
import math
import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

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

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file')
args = parser.parse_args()

# Load face detection model files
base_dir = os.path.dirname(__file__)
faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load Age-Gender PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeGenderResNet().to(device)
model.load_state_dict(torch.load(os.path.join(base_dir, "age_gender_resnet18.pth"), map_location=device))
model.eval()

# Get preprocessing pipeline from torchvision weights
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
genderList = ['Male', 'Female']

padding = 20

if args.image:
    # Process single image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image {args.image}")
        exit(1)
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
    else:
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

            # Convert to RGB and preprocess
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_age, pred_gender = model(face_tensor)
                age = int(pred_age.item())
                gender_idx = torch.argmax(pred_gender, dim=1).item()
                gender = genderList[gender_idx]

            print(f"Gender: {gender}")
            print(f"Age: {age} years")

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Detecting age and gender", resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # Open webcam stream
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam")
        exit(1)

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            print("Error: No frame captured from webcam")
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            # No faces found; just show frame
            cv2.imshow("Detecting age and gender", resultImg)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
            continue

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_age, pred_gender = model(face_tensor)
                age = int(pred_age.item())
                gender_idx = torch.argmax(pred_gender, dim=1).item()
                gender = genderList[gender_idx]

            print(f"Gender: {gender}")
            print(f"Age: {age} years")

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detecting age and gender", resultImg)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break

    video.release()
    cv2.destroyAllWindows()
