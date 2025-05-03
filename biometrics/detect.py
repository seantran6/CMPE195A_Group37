import cv2
import torch
import argparse
import os
from data_preprocessing import preprocess_image  # Import the preprocess function from the previous code
from train_model import gender_model, age_model  # Import the trained models
from torchvision import models
import torch.nn as nn

# Define and load gender model
gender_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
gender_model.fc = nn.Linear(gender_model.fc.in_features, 2)
gender_model.load_state_dict(torch.load("gender_model.pth", map_location=torch.device('cpu')))
gender_model.eval()

# Define and load age model
age_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
age_model.fc = nn.Linear(age_model.fc.in_features, 8)
age_model.load_state_dict(torch.load("age_model.pth", map_location=torch.device('cpu')))
age_model.eval()

# Function for face detection (same as your original code)
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

# Argument parser for input image or webcam
parser = argparse.ArgumentParser()
parser.add_argument('--image', help="Path to the image or webcam (default 0 for webcam)", default=0)
args = parser.parse_args()

# Paths for face, gender, and age models
base_dir = os.path.dirname(__file__)
faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load pre-trained gender and age models (you will use PyTorch models here)
gender_model.eval()  # Set PyTorch models to evaluation mode
age_model.eval()

# Lists for gender and age classes
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# OpenCV for webcam or image input
video = cv2.VideoCapture(args.image if args.image else 0, cv2.CAP_DSHOW)

padding = 20
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Preprocess the face for PyTorch model
        image_tensor = preprocess_image(face)

        # Gender prediction using PyTorch model
        with torch.no_grad():
            gender_output = gender_model(image_tensor)
            gender = genderList[gender_output.argmax().item()]  # Get the class with the highest probability
        print(f'Gender: {gender}')

        # Age prediction using PyTorch model
        with torch.no_grad():
            age_output = age_model(image_tensor)
            age = ageList[age_output.argmax().item()]  # Get the class with the highest probability
        print(f'Age: {age}')  # Display age with parentheses

        # Display the predicted gender and age on the frame
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Detecting age and gender", resultImg)

video.release()
cv2.destroyAllWindows()
