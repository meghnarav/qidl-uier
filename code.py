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
val_dataset = UnderwaterDataset("data/val_input", "data/val_gt", transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataset = UnderwaterDataset("data/test_input", "data/test_gt", transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
loss_history = []
psnr_history = []
val_loss_history = []
ssim_history = []

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
    # VALIDATION
    model.eval()
    val_loss = 0
    total_ssim = 0
    with torch.no_grad():
        for img, gt in val_loader:
            img, gt = img.to(device), gt.to(device)
            out = model(img)
            loss = total_loss(out, gt)
            val_loss += loss.item()
            total_ssim += compute_ssim(out, gt)
    avg_val_loss = val_loss / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    val_loss_history.append(avg_val_loss)
    ssim_history.append(avg_ssim)
    print(f"Epoch {epoch+1} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | SSIM: {avg_ssim:.4f}")

# GRAPHS
# 1 Loss
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss")
plt.savefig("graph1_loss.png")
# 2 Validation Loss
plt.figure()
plt.plot(val_loss_history)
plt.title("Validation Loss")
plt.savefig("graph2_val_loss.png")
# 3 PSNR
plt.figure()
plt.plot(psnr_history)
plt.title("PSNR Curve")
plt.savefig("graph3_psnr.png")
# 4 SSIM
plt.figure()
plt.plot(ssim_history)
plt.title("SSIM Curve")
plt.savefig("graph4_ssim.png")
# 5 Combined Loss vs Val
plt.figure()
plt.plot(loss_history, label="Train")
plt.plot(val_loss_history, label="Val")
plt.legend()
plt.title("Train vs Val Loss")
plt.savefig("graph5_compare.png")
# 6 PSNR distribution
plt.figure()
plt.hist(psnr_list)
plt.title("PSNR Distribution")
plt.savefig("graph6_hist_psnr.png")
# 7 SSIM distribution
plt.figure()
plt.hist(ssim_list)
plt.title("SSIM Distribution")
plt.savefig("graph7_hist_ssim.png")
# 8 UIQM
plt.figure()
plt.plot(uiqm_list)
plt.title("UIQM Scores")
plt.savefig("graph8_uiqm.png")
# 9 UCIQE
plt.figure()
plt.plot(uciqe_list)
plt.title("UCIQE Scores")
plt.savefig("graph9_uciqe.png")

# TEST / INFERENCE
def enhance_image(model, image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
    return out.squeeze().cpu()
    
