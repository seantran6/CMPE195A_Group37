import scipy.io
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

MEAN_VALUES = [78.4263377603, 87.7689143744, 114.895847746]
AGE_CLASSES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def load_data(file_path, image_base_path='wiki/images'):
    wiki_data = scipy.io.loadmat(file_path)
    wiki_data = wiki_data['wiki'][0, 0]

    image_paths = [os.path.join(image_base_path, str(path[0])) for path in wiki_data['full_path'][0]]
    print("Image paths:", image_paths)  # Debugging line to see the paths

    photo_years = wiki_data['photo_taken'][0]
    genders = wiki_data['gender'][0]

    return image_paths, photo_years, genders

def prepare_gender_labels(gender_data):
    return np.nan_to_num(gender_data, nan=0).astype(int)

def prepare_age_labels(gender_data, photo_years):
    current_year = 2023
    age_labels = []

    for photo_year in photo_years:
        actual_age = current_year - photo_year
        age_class = 0  # Default to youngest bin
        for j, age_range in enumerate(AGE_CLASSES):
            age_min, age_max = map(int, age_range.strip('()').split('-'))
            if age_min <= actual_age <= age_max:
                age_class = j
                break
        age_labels.append(age_class)

    return np.array(age_labels)


class WikiDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found or unreadable at: {image_path}")
            image = cv2.resize(image, (224, 224))
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))

            image_tensor = torch.tensor(image, dtype=torch.float32)

            if self.transform:
                image_tensor = self.transform(image_tensor)

            return image_tensor, torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception as e:
            print(f"[WARNING] Skipping index {idx} due to error: {e}")
            return None  # Allow collate_fn to handle this gracefully

# Define the transformation
transform = transforms.Compose([
    transforms.Lambda(lambda x: x),  # Identity transform (data is already tensor)
])

# Custom collate function to filter out None samples
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def create_dataloader(image_paths, age_labels, gender_labels, batch_size=32):
    combined_labels = np.concatenate([age_labels, gender_labels.reshape(-1, 1)], axis=1)
    dataset = WikiDataset(image_paths, combined_labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
