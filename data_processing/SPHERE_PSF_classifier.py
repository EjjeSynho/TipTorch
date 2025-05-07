#%%
import re
import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from collections import defaultdict
from SPHERE_data_settings import SPHERE_DATA_FOLDER, device
from torch.utils.data import DataLoader, Dataset
from PIL import Image



#%%

# def read_labels_file(file_path):
#     label_dict = defaultdict(list)

#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             line = line.strip()
#             if not line:
#                 continue

#             # Extract ID (digits before the first "_")
#             match = re.match(r"(\d+)_", line)
#             if not match:
#                 continue  # Skip lines that do not match expected pattern

#             file_id = int(match.group(1))

#             # Extract labels (everything after ":")
#             labels = line.split(":")[1].strip().split(", ") if ":" in line else []
#             label_dict[file_id] = labels

#     return dict(label_dict)


def read_labels_file(file_path):
    label_dict = defaultdict(list)
    filename_dict = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Split the line at the colon to separate filename and labels
            if ":" in line:
                filename_part, labels_part = line.split(":", 1)
            else:
                filename_part = line
                labels_part = ""

            filename = filename_part.strip()  # Full filename with extension
            try:
                file_id = int(filename.split("_")[0])
            except ValueError:
                continue  # Skip if the file_id cannot be parsed

            # Extract labels if any
            labels = labels_part.strip().split(", ") if labels_part else []
            label_dict[file_id] = labels
            filename_dict[file_id] = filename

    return dict(label_dict), dict(filename_dict)


labels_dict, filenames_dict = read_labels_file(os.path.join(SPHERE_DATA_FOLDER, "labels.txt") )


def generate_dataset():
    image_dir = os.path.join(SPHERE_DATA_FOLDER, 'IRDIS_images')
    filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    init_images, ids = [], []

    for image_file in tqdm(filenames):
        image_path = os.path.join(image_dir, image_file)
        image = plt.imread(image_path)
        ids.append(int(image_file.split('_')[0]))
        
        H, W, _ = image.shape
        grayscale_image = np.dot(image[...,:3], [0.299, 0.587, 0.114])  
        init_images.append(grayscale_image)


    Ws = [img.shape[1] for img in init_images]
    Hs = [img.shape[0] for img in init_images]

    W, H = max(set(Ws), key=Ws.count), max(set(Hs), key=Hs.count)

    for i in range(len(init_images)):
        current_width = init_images[i].shape[1]
        if current_width > W:
            start = (current_width - W) // 2
            end = start + W
            init_images[i] = init_images[i][:, start:end]
            
    processed_images = {}
    for i in range(len(init_images)):
        image = init_images[i]
        H, W = image.shape
        downsampled = image[H-W:H, :][::9, ::9]
        
        if downsampled.shape[0] != 64: downsampled = downsampled[1:, :]
        if downsampled.shape[1] != 64: downsampled = downsampled[:, 1:]
            
        processed_images[ids[i]] = downsampled

    # Sort based on keys (IDs)
    processed_images = dict(sorted(processed_images.items()))
    processed_images_packed = np.stack([processed_images[key] for key in processed_images.keys()])
    images_ids = np.array([key for key in processed_images.keys()])

    # for id, labels in labels_dict.items():
    #     print(f"{id}: {labels}")

    with open(os.path.join(SPHERE_DATA_FOLDER, "PSF_classes.txt"), 'r', encoding='utf-8') as file:
        PSF_classes = file.readlines()
        
    for i in range(len(PSF_classes)):
        PSF_classes[i] = PSF_classes[i].strip()

    # Create a dictionary to store one-hot-vectors associated with each ID
    PSF_labels_dict = {}

    # Fill the dictionary with one-hot-vectors for each ID in labels_dict
    for id, labels in labels_dict.items():
        one_hot_vector = np.zeros(len(PSF_classes), dtype=np.uint8)
        for label in labels:
            if label in PSF_classes:
                one_hot_vector[PSF_classes.index(label)] = 1
        PSF_labels_dict[id] = one_hot_vector

    for id in ids:
        if id not in PSF_labels_dict:
            PSF_labels_dict[id] = np.zeros(len(PSF_classes), dtype=np.uint8)


    PSF_labels_dict = dict(sorted(PSF_labels_dict.items()))

    # Convert the dictionary back to a NumPy array if needed
    PSF_labels_packed = np.array([PSF_labels_dict[id] for id in PSF_labels_dict.keys()])
    PSF_labels_ids = np.array([key for key in PSF_labels_dict.keys()])

    assert np.all(PSF_labels_ids == images_ids)

    dataset_dict = {
        'id': images_ids,
        'iamges': processed_images_packed,
        'labels': PSF_labels_packed,
        'classes': PSF_classes,
    }

    # Write to pickle file
    with open(os.path.join(SPHERE_DATA_FOLDER, "PSF_classifier_dataset.pkl"), 'wb') as file:
        pickle.dump(dataset_dict, file)


# generate_dataset()

#%%
class PSFDataset(Dataset):
    def __init__(self, pickle_path, transform=None, filter_zero_labels=True):
        """
        Loads data from a pickle file and initializes the dataset.
        
        Args:
            pickle_path (str): Path to the pickle file containing the dataset.
            transform (callable, optional): A function/transform to apply to images.
            filter_zero_labels (bool): If True, remove samples where the sum of labels is 0.
        """
        # Load dataset from pickle file
        with open(pickle_path, 'rb') as file:
            dataset_dict = pickle.load(file)

        # Extract data
        self.image_ids = np.array(dataset_dict['id'])  # Image IDs
        self.images = np.array(dataset_dict['images'])  # Image data
        self.labels = np.array(dataset_dict['labels'])  # Labels (assumed multi-hot vectors or class labels)
        self.classes = dataset_dict['classes']  # Class names
        self.transform = transform

        # Filter out samples where sum of labels vector is 0 (i.e., unlabeled images)
        if filter_zero_labels:
            non_zero_mask = np.sum(self.labels, axis=1) > 0  # Boolean mask
            self.image_ids = self.image_ids[non_zero_mask]
            self.images = self.images[non_zero_mask]
            self.labels = self.labels[non_zero_mask]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): The transformed image.
            label (Tensor): The class label.
            image_id (str): The corresponding image ID.
        """
        image = self.images[idx]  # Image data (N, H, W)
        label = self.labels[idx]  # Label (vector or class index)
        image_id = self.image_ids[idx]  # Corresponding ID

        # Convert NumPy array to PIL image (if needed)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, label, image_id


#%%
    # Define a sample transformation (optional)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (if grayscale)
])

# Path to dataset pickle file
pickle_path = os.path.join(SPHERE_DATA_FOLDER, "PSF_classifier_dataset.pkl")

# Create dataset instance with filtered non-zero labels
dataset = PSFDataset(pickle_path, transform=transform, filter_zero_labels=True)
# Create separate dataset for samples which are yet to be predicted
zero_label_dataset = PSFDataset(pickle_path, transform=transform, filter_zero_labels=False)

# Compute mask for zero-label images
zero_label_mask = np.sum(zero_label_dataset.labels, axis=1) == 0
zero_label_dataset.image_ids = zero_label_dataset.image_ids[zero_label_mask]
zero_label_dataset.images = zero_label_dataset.images[zero_label_mask]
zero_label_dataset.labels = zero_label_dataset.labels[zero_label_mask]

print(f"Total dataset size (filtered): {len(dataset)}")
print(f"Zero-label dataset size: {len(zero_label_dataset)}")

BATCH_SIZE = 32

dataloader_train = DataLoader(dataset,            batch_size=BATCH_SIZE, shuffle=True)
dataloader_pred  = DataLoader(zero_label_dataset, batch_size=BATCH_SIZE, shuffle=True)

for images, labels, image_ids in dataloader_train:
    print("Batch size:", images.shape)
    print("Labels:", labels)
    print("Image IDs:", image_ids)
    break


#%%
# Load a pre-trained ResNet18
# resnet18 = models.resnet18(pretrained=True)

# Define a simple CNN model for classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_size=64):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # output: 16 x (input_size/2) x (input_size/2)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # output: 32 x (input_size/4) x (input_size/4)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # output: 64 x (input_size/8) x (input_size/8)
        )
        
        feature_size = input_size // 8  # after three poolings
        self.classifier = nn.Sequential(
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten feature maps into a vector
        x = self.classifier(x)
        return x

#%%
# Hyperparameters
num_classes = len(dataset.classes)
input_size = dataset.images.shape[-1]
batch_size = BATCH_SIZE
learning_rate = 0.001
num_epochs = 200

# Example transforms: convert numpy array to tensor and normalize.
transform = transforms.Compose([
    transforms.ToTensor(),                # converts (H, W) numpy array to (C, H, W)
    transforms.Normalize((0.5,), (0.5,))  # normalization for grayscale images
])

#%%
# Initialize the model, loss function, and optimizer.
model = SimpleCNN(num_classes, input_size=input_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%
# Training loop.
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels, _ in dataloader_train:
        # Add a channel dimension if not already present.
        # images should be of shape (B, 1, H, W)
        if images.ndim == 3:
            images = images.unsqueeze(1)
    
        images = images.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

#%%
# Predicting missing labels
model.eval()
val_running_loss = 0.0
predicted_labels = {}

with torch.no_grad():
    for images, labels, ids in dataloader_pred:
        # Add a channel dimension if not already present. Images should be of shape (B, 1, H, W)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        outputs = model(images)

        res = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy().round(1)
        res_classes = np.zeros_like(res)
        res_classes[res >= 0.3] = 1
        
        ids = ids.cpu().detach().numpy().tolist()
        
        for id in ids:
           predicted_labels[id] = res_classes[ids.index(id)].astype(np.uint8)


predicted_classes = {}

for id in sorted(predicted_labels.keys()):
    predicted_classes[id] = [dataset.classes[i] for i, value in enumerate(predicted_labels[id]) if value == 1]

#%%

filenames_img = os.listdir(os.path.join(SPHERE_DATA_FOLDER, "IRDIS_images"))
filenames_img = [f for f in filenames_img if f.endswith(".png")]

filenames_dict = {}

for filename in filenames_img:
    filenames_dict[int(filename.split("_")[0])] = filename
    
    
#%%
lines = []

for id in predicted_classes.keys():
    labels_str = "".join([c+', ' for c in predicted_classes[id]])
    labels_str = labels_str[:-2]
    final_str = filenames_dict[id] + ": " + labels_str
    lines.append(final_str)

old_labels = []
with open(os.path.join(SPHERE_DATA_FOLDER, "labels.txt"), 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        else:
            old_labels.append(line)

lines = old_labels + lines

# Write to file
with open(os.path.join(SPHERE_DATA_FOLDER, "labels_pred.txt"), "w") as f:
    for line in lines:
        f.write(line + "\n")


        
# %%
