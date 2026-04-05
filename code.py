import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

class UnderwaterDataset(Dataset):
    def __init__(self, input_dir, gt_dir=None, transform=None):
        self.input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]) if gt_dir else None
        self.transform = transform
    def __len__(self):
        return len(self.input_paths)
    def __getitem__(self, idx):
        img = Image.open(self.input_paths[idx]).convert("RGB")
        if self.gt_paths:
            gt = Image.open(self.gt_paths[idx]).convert("RGB")
        else:
            gt = img
        if self.transform:
            img = self.transform(img)
            gt = self.transform(gt)
        return img, gt

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Example placeholders for dataset
train_dataset = UnderwaterDataset(image_paths=["path1.png","path2.png"],
                                  labels=[0,1],
                                  transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


class QuantumInspiredCNN(nn.Module):
    def __init__(self):
        super(QuantumInspiredCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Residual block
        self.res_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Output layer
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Quantum-inspired encoding (simple approximation)
        x = torch.cos(np.pi * x) + torch.sin(np.pi * x)
        # Convolution + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Residual block
        res = F.relu(self.res_conv1(x))
        res = self.res_conv2(res)
        x = x + res
        # Output reconstruction
        x = torch.sigmoid(self.conv_out(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantumInspiredCNN().to(device)
criterion_mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2): 
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        # Forward pass
        outputs = model(images)
        # Compute MSE loss (placeholder target = images)
        loss = criterion_mse(outputs, images)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
