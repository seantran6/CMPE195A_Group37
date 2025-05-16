# biometrics/age_gender_predictor.py

import cv2
import torch
import torch.nn as nn
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


class AgeGenderPredictor:
    def __init__(self, model_path="age_gender_resnet18.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AgeGenderResNet().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()

    def predict(self, frame_bgr):
        """Takes an OpenCV BGR frame and returns (age, gender_str)"""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_age, pred_gender = self.model(img_tensor)
            age = int(pred_age.item())
            gender_idx = torch.argmax(pred_gender, dim=1).item()
            gender = {0: "Male", 1: "Female"}[gender_idx]

        return age, gender
