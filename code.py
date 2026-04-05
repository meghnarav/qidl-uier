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


# DATASET
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

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor()
])

# QUANTUM-INSPIRED ENCODING

class QuantumEncoding(nn.Module):
    def forward(self, x):
        theta = math.pi * x
        real = torch.cos(theta)
        imag = torch.sin(theta)
        return torch.cat([real, imag], dim=1)

# ATTENTION MODULES
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//8),
            nn.ReLU(),
            nn.Linear(channels//8, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,h,w = x.size()
        y = x.mean(dim=(2,3))
        y = self.fc(y).view(b,c,1,1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max,_ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg,max], dim=1)
        attn = torch.sigmoid(self.conv(x_cat))
        return x * attn

# TRANSFORMER BLOCK
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        b,c,h,w = x.shape
        x_flat = x.view(b, c, -1).permute(0,2,1)
        attn_out,_ = self.attn(x_flat, x_flat, x_flat)
        x = self.norm1(x_flat + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        x = x.permute(0,2,1).view(b,c,h,w)
        return x

# FULL QIDL MODEL
class QIDLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumEncoding()
        self.conv1 = nn.Conv2d(6, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.transformer = TransformerBlock(128)
        self.ca = ChannelAttention(128)
        self.sa = SpatialAttention()
        self.res1 = nn.Conv2d(128,128,3,padding=1)
        self.res2 = nn.Conv2d(128,128,3,padding=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,3,3,padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.quantum(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.transformer(x)
        x = self.ca(x)
        x = self.sa(x)
        res = F.relu(self.res1(x))
        res = self.res2(res)
        x = x + res
        out = self.decoder(x)
        return out

# LOSS FUNCTIONS 
def ssim_loss(img1, img2):
    return 1 - torch.mean((2*img1*img2 + 0.01)/(img1**2 + img2**2 + 0.01))
def perceptual_loss(x, y):
    return torch.mean(torch.abs(x - y))
def total_loss(out, gt):
    mse = F.mse_loss(out, gt)
    ssim = ssim_loss(out, gt)
    perc = perceptual_loss(out, gt)
    return 0.5*mse + 0.3*ssim + 0.2*perc

# METRICS
def psnr(mse):
    return 10 * torch.log10(1 / mse)
def compute_metrics(out, gt):
    mse = F.mse_loss(out, gt)
    return {
        "MSE": mse.item(),
        "PSNR": psnr(mse).item()
    }

# TRAINING LOOP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QIDLModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_dataset = UnderwaterDataset("data/input", "data/gt", transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
loss_history = []
psnr_history = []
for epoch in range(10):
    model.train()
    total_loss_epoch = 0
    for img, gt in train_loader:
        img, gt = img.to(device), gt.to(device)
        out = model(img)
        loss = total_loss(out, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
    avg_loss = total_loss_epoch / len(train_loader)
    loss_history.append(avg_loss)
    metrics = compute_metrics(out, gt)
    psnr_history.append(metrics["PSNR"])
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | PSNR: {metrics['PSNR']:.2f}")

# GRAPHS
plt.plot(loss_history)
plt.title("Training Loss")
plt.savefig("loss.png")
plt.plot(psnr_history)
plt.title("PSNR Curve")
plt.savefig("psnr.png")

# TEST / INFERENCE
def enhance_image(model, image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
    return out.squeeze().cpu()


'''

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
            print(f"Epoch [{epoch+1}], Batch [{batch_idx}], Loss: {loss.item():.4f}") '''
