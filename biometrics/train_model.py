import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import load_data, prepare_gender_labels, prepare_age_labels
import numpy as np
import os
from tqdm import tqdm

# ----- Custom Dataset -----
class CustomDataset(Dataset):
    def __init__(self, image_paths, age_labels, gender_labels, transform=None):
        self.image_paths = image_paths
        self.age_labels = age_labels
        self.gender_labels = gender_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_name = self.image_paths[index]
        age_label = self.age_labels[index]
        gender_label = self.gender_labels[index]

        # Fix potential array wrapping
        age_label = int(age_label[0]) if isinstance(age_label, np.ndarray) else int(age_label)
        gender_label = int(gender_label[0]) if isinstance(gender_label, np.ndarray) else int(gender_label)

        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Skipping image {img_name}: {e}")
            index = (index + 1) % len(self.image_paths)  # Move to the next image
            return self.__getitem__(index)  # Recursively try the next image

        if self.transform:
            image = self.transform(image)

        return image, age_label, gender_label


# ----- Image Transform -----
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- Load Data -----
image_paths, photo_years, genders = load_data('wiki/wiki.mat', image_base_path='wiki/images')
gender_labels = prepare_gender_labels(genders)
age_labels = prepare_age_labels(genders, photo_years)

# ----- DataLoader -----
def create_dataloader(image_paths, age_labels, gender_labels, batch_size=32):
    dataset = CustomDataset(image_paths, age_labels, gender_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

dataloader = create_dataloader(image_paths, age_labels, gender_labels)

# ----- Load Models -----
def initialize_models():
    gender_model = models.resnet18(pretrained=True)
    age_model = models.resnet18(pretrained=True)
    gender_model.fc = nn.Linear(gender_model.fc.in_features, 2)  # 2 classes for gender
    age_model.fc = nn.Linear(age_model.fc.in_features, 8)  # 8 classes for age groups
    return gender_model, age_model

# ----- Device Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Checkpoint Functions -----
def save_checkpoint(epoch, gender_model, age_model, optimizer_gender, optimizer_age, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'gender_model_state': gender_model.state_dict(),
        'age_model_state': age_model.state_dict(),
        'optimizer_gender_state': optimizer_gender.state_dict(),
        'optimizer_age_state': optimizer_age.state_dict()
    }, filename)
    print(f"[INFO] Saved checkpoint at epoch {epoch}")

def load_checkpoint(gender_model, age_model, optimizer_gender, optimizer_age, filename='checkpoint.pth'):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        gender_model.load_state_dict(checkpoint['gender_model_state'])
        age_model.load_state_dict(checkpoint['age_model_state'])
        optimizer_gender.load_state_dict(checkpoint['optimizer_gender_state'])
        optimizer_age.load_state_dict(checkpoint['optimizer_age_state'])
        print(f"[INFO] Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    else:
        print("[INFO] No checkpoint found, starting from scratch.")
        return 0

# ----- Training Function -----
def train_model(num_epochs=10, checkpoint_path='checkpoint.pth', resume=True):
    gender_model, age_model = initialize_models()
    gender_model = gender_model.to(device)
    age_model = age_model.to(device)

    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()
    optimizer_gender = optim.Adam(gender_model.parameters(), lr=0.001)
    optimizer_age = optim.Adam(age_model.parameters(), lr=0.001)

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(gender_model, age_model, optimizer_gender, optimizer_age, checkpoint_path)

    print(f"[INFO] Starting training from epoch {start_epoch + 1}/{num_epochs}")

    try:
        for epoch in range(start_epoch, num_epochs):
            gender_model.train()
            age_model.train()

            running_loss_gender = 0.0
            running_loss_age = 0.0

            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            progress_bar.set_postfix({'Gender Loss': 'N/A', 'Age Loss': 'N/A'})

            for i, (inputs, age_targets, gender_targets) in progress_bar:
                inputs = inputs.to(device)
                age_targets = age_targets.to(device)
                gender_targets = gender_targets.to(device)

                # Gender model
                optimizer_gender.zero_grad()
                outputs_gender = gender_model(inputs)
                loss_gender = criterion_gender(outputs_gender, gender_targets)
                loss_gender.backward()
                optimizer_gender.step()

                # Age model
                optimizer_age.zero_grad()
                outputs_age = age_model(inputs)
                loss_age = criterion_age(outputs_age, age_targets)
                loss_age.backward()
                optimizer_age.step()

                running_loss_gender += loss_gender.item()
                running_loss_age += loss_age.item()

                if i % 10 == 9:
                    progress_bar.set_postfix({
                        'Gender Loss': f"{running_loss_gender / (i+1):.4f}",
                        'Age Loss': f"{running_loss_age / (i+1):.4f}"
                    })

            save_checkpoint(epoch + 1, gender_model, age_model, optimizer_gender, optimizer_age, checkpoint_path)

        print("[INFO] Training completed.")

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user. Saving checkpoint...")
        save_checkpoint(epoch, gender_model, age_model, optimizer_gender, optimizer_age, checkpoint_path)

    # Final model save
    torch.save(gender_model.state_dict(), 'gender_model_final.pth')
    torch.save(age_model.state_dict(), 'age_model_final.pth')
    print("[INFO] Final models saved.")


# ----- Entry Point -----
if __name__ == "__main__":
    train_model(num_epochs=10, checkpoint_path='checkpoint.pth', resume=True)
