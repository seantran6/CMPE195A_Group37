import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import threading
import time
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Enable template auto-reloading
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load the pre-trained models
base_dir = os.path.dirname(__file__)
# Paths to the face, gender, and age model files
faceProto = r"C:\Users\seann\Desktop\CMPE195A\Senior Project\CMPE195A_Group37\biometrics\opencv_face_detector.pbtxt"
faceModel = r"C:\Users\seann\Desktop\CMPE195A\Senior Project\CMPE195A_Group37\biometrics\opencv_face_detector_uint8.pb"

genderProto = r"C:\Users\seann\Desktop\CMPE195A\Senior Project\CMPE195A_Group37\biometrics\gender_deploy.prototxt"
genderModel = r"C:\Users\seann\Desktop\CMPE195A\Senior Project\CMPE195A_Group37\biometrics\gender_net.caffemodel"

ageProto = r"C:\Users\seann\Desktop\CMPE195A\Senior Project\CMPE195A_Group37\biometrics\age_deploy.prototxt"
ageModel = r"C:\Users\seann\Desktop\CMPE195A\Senior Project\CMPE195A_Group37\biometrics\age_net.caffemodel"


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

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
            # Draw bounding box for face
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Decode the base64 image and process it
def process_image(image_data):
    # Decode the image
    img_data = base64.b64decode(image_data.split(',')[1])  # Remove "data:image/jpeg;base64,"
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR

    # Process the image to detect faces
    resultImg, faceBoxes = highlightFace(faceNet, img)

    gender = "Unknown"
    age = "Unknown"

    if faceBoxes:
        # Assuming the first face is the one to detect
        face = resultImg[max(0, faceBoxes[0][1]):min(faceBoxes[0][3], resultImg.shape[0] - 1),
                         max(0, faceBoxes[0][0]):min(faceBoxes[0][2], resultImg.shape[1] - 1)]
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]  # Predict gender

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]  # Predict age

    return resultImg, gender, age


@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    image_data = data['image']
    resultImg, gender, age = process_image(image_data)

    # Convert result image back to base64 for sending it to frontend with bounding boxes (for live feed)
    _, buffer = cv2.imencode('.jpg', resultImg)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'gender': gender, 'age': age, 'image': result_image})





@app.route('/recognition', methods=['GET'])
def recognition():
    return render_template('recognition.html')

@app.route('/')
def home():
    return render_template('index.html')  # Ensure `index.html` exists

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)


