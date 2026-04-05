"""
QIDL: Quantum-Inspired Deep Learning Framework for Underwater Image Enhancement
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_sk
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────

class UnderwaterDataset(Dataset):
    """
    Loads paired underwater / ground-truth images from disk.
    If no gt_dir is given, the input image acts as its own target
    (useful for unpaired datasets such as RUIE / EUVP).
    """
    def __init__(self, input_dir, gt_dir=None, transform=None):
        self.input_paths = sorted(
            [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        )
        self.gt_paths = (
            sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            if gt_dir else None
        )
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        img = Image.open(self.input_paths[idx]).convert("RGB")
        gt  = (Image.open(self.gt_paths[idx]).convert("RGB")
               if self.gt_paths else img.copy())
        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed); torch.manual_seed(seed)
            img = self.transform(img)
            random.seed(seed); torch.manual_seed(seed)
            gt  = self.transform(gt)
        return img, gt


# ─────────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# ─────────────────────────────────────────────
# 3. QUANTUM-INSPIRED FEATURE ENCODING
# ─────────────────────────────────────────────

class QuantumEncoding(nn.Module):
    """
    Maps each pixel intensity p ∈ [0,1] to a two-component
    quantum-inspired state:  α = cos(π·p),  β = sin(π·p).
    This doubles the channel count (RGB → 6-ch) and adds
    a superposition-like non-linear prior before convolution.
    """
    def forward(self, x):
        # x is normalised to [0,1] for encoding
        x_norm = (x + 1.0) / 2.0          # undo [-1,1] normalisation
        theta   = math.pi * x_norm
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)


# ─────────────────────────────────────────────
# 4. ATTENTION MODULES
# ─────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.fc(self.avg_pool(x).view(b, c))
        max_y = self.fc(self.max_pool(x).view(b, c))
        y = self.sigmoid(avg_y + max_y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial attention using average and max pooling across channels."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ─────────────────────────────────────────────
# 5. TRANSFORMER BLOCK
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Lightweight Vision-Transformer block.
    Pixels are flattened to sequence tokens; multi-head
    self-attention captures global spatial dependencies.
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(0, 2, 1)   # (B, HW, C)
        n   = self.norm1(seq)
        a, _ = self.attn(n, n, n)
        seq  = seq + a
        seq  = seq + self.mlp(self.norm2(seq))
        return seq.permute(0, 2, 1).view(b, c, h, w)


# ─────────────────────────────────────────────
# 6. RESIDUAL BLOCK
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    """Standard residual block with CBAM attention."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.cbam = CBAM(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.cbam(self.block(x)))


# ─────────────────────────────────────────────
# 7. COMPLETE QIDL MODEL
# ─────────────────────────────────────────────

class QIDLModel(nn.Module):
    """
    Quantum-Inspired Deep Learning (QIDL) model for underwater image
    enhancement.  Architecture:

        Input (3-ch)
          ↓  Quantum Encoding  →  6-ch
          ↓  Encoder (CNN)     →  64 → 128 ch
          ↓  Transformer Block (global attention)
          ↓  Residual Blocks × 4 + CBAM
          ↓  Decoder (CNN)     →  64 → 3 ch  (Sigmoid)
        Output (3-ch enhanced image)
    """
    def __init__(self, base_ch=64, n_res=4):
        super().__init__()

        # ── Quantum encoding ──────────────────────────────────
        self.quantum = QuantumEncoding()

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
        )

        # ── Transformer (global context) ───────────────────────
        self.transformer = TransformerBlock(base_ch * 2)

        # ── Residual blocks ────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResBlock(base_ch * 2) for _ in range(n_res)]
        )

        # ── Bottleneck CBAM ────────────────────────────────────
        self.cbam = CBAM(base_ch * 2)

        # ── Decoder ────────────────────────────────────────────
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch, 3, 3, padding=1),
            nn.Tanh(),              # output in [-1,1], same as normalised GT
        )

    def forward(self, x):
        # Quantum-inspired encoding
        xq = self.quantum(x)          # (B, 6, H, W)

        # Encoder
        f1 = self.enc1(xq)            # (B, 64, H, W)
        f2 = self.enc2(f1)            # (B, 128, H, W)

        # Global attention via transformer
        ft = self.transformer(f2)     # (B, 128, H, W)

        # Residual feature refinement
        fr = self.res_blocks(ft)      # (B, 128, H, W)
        fr = self.cbam(fr)

        # Decoder
        d1 = self.dec1(fr)            # (B, 64, H, W)
        out = self.dec2(d1)           # (B, 3, H, W)
        return out


# ─────────────────────────────────────────────
# 8. LOSS FUNCTIONS (Physics-Guided)
# ─────────────────────────────────────────────

def compute_ssim_loss(x, y, C1=0.01**2, C2=0.03**2):
    """Differentiable SSIM loss (1 − SSIM)."""
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sig_x  = F.avg_pool2d(x**2, 3, 1, 1) - mu_x2
    sig_y  = F.avg_pool2d(y**2, 3, 1, 1) - mu_y2
    sig_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2))
    return 1.0 - ssim_map.mean()


def compute_color_loss(out, gt):
    """Physics-guided color consistency: L1 in each channel."""
    return sum(F.l1_loss(out[:, i], gt[:, i]) for i in range(3)) / 3.0


def compute_gradient_loss(out, gt):
    """Edge-preserving gradient loss to retain structural sharpness."""
    def grad(t):
        dx = t[:, :, :, 1:] - t[:, :, :, :-1]
        dy = t[:, :, 1:, :] - t[:, :, :-1, :]
        return dx, dy
    dx_o, dy_o = grad(out)
    dx_g, dy_g = grad(gt)
    return F.l1_loss(dx_o, dx_g) + F.l1_loss(dy_o, dy_g)


def total_loss(out, gt, lam=(0.5, 0.3, 0.1, 0.1)):
    """
    Composite physics-guided loss:
      L = λ1·L_MSE + λ2·L_SSIM + λ3·L_Color + λ4·L_Grad
    """
    l_mse   = F.mse_loss(out, gt)
    l_ssim  = compute_ssim_loss(out, gt)
    l_color = compute_color_loss(out, gt)
    l_grad  = compute_gradient_loss(out, gt)
    return (lam[0] * l_mse + lam[1] * l_ssim +
            lam[2] * l_color + lam[3] * l_grad)


# ─────────────────────────────────────────────
# 9. METRICS
# ─────────────────────────────────────────────

def psnr_metric(out, gt):
    """PSNR in dB. Tensors in [-1,1]."""
    mse = F.mse_loss(out, gt).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(4.0 / mse)    # max range is 2 (i.e., −1 to 1)


def ssim_metric(out, gt):
    """Per-image SSIM averaged over batch, computed in numpy."""
    out_np = ((out.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    gt_np  = ((gt.detach().cpu().numpy()  + 1) / 2).clip(0, 1)
    scores = []
    for i in range(out_np.shape[0]):
        o = out_np[i].transpose(1, 2, 0)
        g = gt_np[i].transpose(1, 2, 0)
        scores.append(ssim_sk(o, g, channel_axis=2, data_range=1.0))
    return float(np.mean(scores))


def uiqm_metric(img_tensor):
    """
    Simplified UIQM estimate:
        UIQM = c1·UICM + c2·UISM + c3·UIConM
    c1=0.0282, c2=0.2953, c3=3.5753 (standard coefficients).
    """
    img = ((img_tensor.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    scores = []
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    for b in range(img.shape[0]):
        im = img[b].transpose(1, 2, 0)
        r, g, bl = im[..., 0], im[..., 1], im[..., 2]
        # UICM: chroma mean
        uicm  = float(np.std(r - g) + np.std(r - bl))
        # UISM: sharpness via gradient magnitude
        dx    = im[:, 1:, :] - im[:, :-1, :]
        dy    = im[1:, :, :] - im[:-1, :, :]
        uism  = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))
        # UIConM: contrast
        uicon = float(im.max() - im.min())
        scores.append(c1 * uicm + c2 * uism + c3 * uicon)
    return float(np.mean(scores))


def uciqe_metric(img_tensor):
    """
    Simplified UCIQE:
        UCIQE = c1·σ_c + c2·con_l + c3·μ_s
    """
    img = ((img_tensor.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    scores = []
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    for b in range(img.shape[0]):
        im = img[b].transpose(1, 2, 0)
        # Approximate in HSV
        r, g, bl = im[..., 0], im[..., 1], im[..., 2]
        chroma = np.sqrt((r - g)**2 + (r - bl)**2 + (g - bl)**2)
        sigma_c = float(np.std(chroma))
        lum = 0.299 * r + 0.587 * g + 0.114 * bl
        con_l = float(lum.max() - lum.min())
        mu_s  = float(np.mean(chroma))
        scores.append(c1 * sigma_c + c2 * con_l + c3 * mu_s)
    return float(np.mean(scores))


# ─────────────────────────────────────────────
# 10. TRAINING FUNCTION
# ─────────────────────────────────────────────

def train_qidl(model, train_loader, val_loader,
               optimizer, scheduler, device,
               epochs=30, early_stop_patience=7,
               save_path="qidl_best.pth"):
    """Full training loop with validation, early stopping, and metric logging."""
    history = {
        "train_loss": [], "val_loss": [],
        "psnr": [], "ssim": [], "uiqm": [], "uciqe": [],
    }
    best_psnr   = 0.0
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for img, gt in train_loader:
            img, gt = img.to(device), gt.to(device)
            out      = model(img)
            loss     = total_loss(out, gt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        psnr_sum, ssim_sum, uiqm_sum, uciqe_sum = 0.0, 0.0, 0.0, 0.0
        n_batches = len(val_loader)

        with torch.no_grad():
            for img, gt in val_loader:
                img, gt = img.to(device), gt.to(device)
                out      = model(img)
                val_loss += total_loss(out, gt).item()
                psnr_sum  += psnr_metric(out, gt)
                ssim_sum  += ssim_metric(out, gt)
                uiqm_sum  += uiqm_metric(out)
                uciqe_sum += uciqe_metric(out)

        val_loss  /= n_batches
        psnr_avg   = psnr_sum  / n_batches
        ssim_avg   = ssim_sum  / n_batches
        uiqm_avg   = uiqm_sum  / n_batches
        uciqe_avg  = uciqe_sum / n_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["psnr"].append(psnr_avg)
        history["ssim"].append(ssim_avg)
        history["uiqm"].append(uiqm_avg)
        history["uciqe"].append(uciqe_avg)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PSNR: {psnr_avg:.2f} | SSIM: {ssim_avg:.4f} | "
              f"UIQM: {uiqm_avg:.4f} | UCIQE: {uciqe_avg:.4f}")

        # Early stopping + checkpoint
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            torch.save(model.state_dict(), save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    return history


# ─────────────────────────────────────────────
# 11. EVALUATION ON TEST SET
# ─────────────────────────────────────────────

def evaluate(model, test_loader, device):
    """Returns per-image metric lists for statistical analysis."""
    model.eval()
    psnr_list, ssim_list, uiqm_list, uciqe_list = [], [], [], []

    with torch.no_grad():
        for img, gt in test_loader:
            img, gt = img.to(device), gt.to(device)
            out      = model(img)
            psnr_list.append(psnr_metric(out, gt))
            ssim_list.append(ssim_metric(out, gt))
            uiqm_list.append(uiqm_metric(out))
            uciqe_list.append(uciqe_metric(out))

    return psnr_list, ssim_list, uiqm_list, uciqe_list


# ─────────────────────────────────────────────
# 12. GRAPH GENERATION (9 Graphs)
# ─────────────────────────────────────────────

def save_graphs(history, psnr_list, ssim_list, uiqm_list, uciqe_list,
                out_dir="graphs"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    style  = {"linewidth": 2, "marker": "o", "markersize": 4}

    # Graph 1 — Training Loss Curve
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], color="#2196F3", **style)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Graph 1: Training Loss Curve")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph1_train_loss.png", dpi=150)
    plt.close()

    # Graph 2 — Validation Loss Curve
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["val_loss"], color="#E91E63", **style)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Graph 2: Validation Loss Curve")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph2_val_loss.png", dpi=150)
    plt.close()

    # Graph 3 — Train vs Val Loss (Combined)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], label="Train", color="#2196F3", **style)
    plt.plot(epochs, history["val_loss"],   label="Validation", color="#E91E63", **style)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Graph 3: Train vs. Validation Loss")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph3_train_val_loss.png", dpi=150)
    plt.close()

    # Graph 4 — PSNR Curve (per epoch)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["psnr"], color="#4CAF50", **style)
    plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)")
    plt.title("Graph 4: PSNR over Training Epochs")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph4_psnr_curve.png", dpi=150)
    plt.close()

    # Graph 5 — SSIM Curve (per epoch)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["ssim"], color="#FF9800", **style)
    plt.xlabel("Epoch"); plt.ylabel("SSIM")
    plt.title("Graph 5: SSIM over Training Epochs")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph5_ssim_curve.png", dpi=150)
    plt.close()

    # Graph 6 — PSNR Distribution (test set)
    plt.figure(figsize=(7, 4))
    plt.hist(psnr_list, bins=20, color="#2196F3", edgecolor="white")
    plt.xlabel("PSNR (dB)"); plt.ylabel("Frequency")
    plt.title("Graph 6: PSNR Distribution (Test Set)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph6_psnr_distribution.png", dpi=150)
    plt.close()

    # Graph 7 — SSIM Distribution (test set)
    plt.figure(figsize=(7, 4))
    plt.hist(ssim_list, bins=20, color="#9C27B0", edgecolor="white")
    plt.xlabel("SSIM"); plt.ylabel("Frequency")
    plt.title("Graph 7: SSIM Distribution (Test Set)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph7_ssim_distribution.png", dpi=150)
    plt.close()

    # Graph 8 — UIQM Curve (per epoch)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["uiqm"], color="#009688", **style)
    plt.xlabel("Epoch"); plt.ylabel("UIQM")
    plt.title("Graph 8: UIQM over Training Epochs")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph8_uiqm_curve.png", dpi=150)
    plt.close()

    # Graph 9 — UCIQE Curve (per epoch)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["uciqe"], color="#FF5722", **style)
    plt.xlabel("Epoch"); plt.ylabel("UCIQE")
    plt.title("Graph 9: UCIQE over Training Epochs")
    plt.tight_layout(); plt.savefig(f"{out_dir}/graph9_uciqe_curve.png", dpi=150)
    plt.close()

    print(f"All 9 graphs saved to '{out_dir}/'")


# ─────────────────────────────────────────────
# 13. INFERENCE / SINGLE-IMAGE ENHANCEMENT
# ─────────────────────────────────────────────

def enhance_image(model, image_path, device, save_path=None):
    """
    Enhance a single underwater image.
    Returns a PIL Image of the enhanced result.
    """
    model.eval()
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img  = Image.open(image_path).convert("RGB")
    orig_size = img.size
    inp  = t(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)                   # (1, 3, 256, 256) in [-1, 1]

    out_np = ((out.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)
    out_np = (out_np * 255).astype(np.uint8).transpose(1, 2, 0)
    result = Image.fromarray(out_np).resize(orig_size, Image.BICUBIC)

    if save_path:
        result.save(save_path)
        print(f"Enhanced image saved to {save_path}")
    return result


# ─────────────────────────────────────────────
# 14. ABLATION UTILITIES
# ─────────────────────────────────────────────

class QIDLNoQuantum(nn.Module):
    """Ablation: QIDL without quantum encoding (plain RGB encoder)."""
    def __init__(self, base_ch=64, n_res=4):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*2), nn.ReLU(inplace=True))
        self.transformer = TransformerBlock(base_ch*2)
        self.res_blocks  = nn.Sequential(*[ResBlock(base_ch*2) for _ in range(n_res)])
        self.cbam        = CBAM(base_ch*2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh())

    def forward(self, x):
        f1  = self.enc1(x)
        f2  = self.enc2(f1)
        ft  = self.transformer(f2)
        fr  = self.cbam(self.res_blocks(ft))
        return self.dec2(self.dec1(fr))


class QIDLNoAttention(nn.Module):
    """Ablation: QIDL without CBAM attention."""
    def __init__(self, base_ch=64, n_res=4):
        super().__init__()
        self.quantum = QuantumEncoding()
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*2), nn.ReLU(inplace=True))
        self.transformer = TransformerBlock(base_ch*2)
        # Plain residual (no CBAM)
        plain_res = []
        for _ in range(n_res):
            plain_res += [
                nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch*2), nn.ReLU(inplace=True)
            ]
        self.res_blocks = nn.Sequential(*plain_res)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh())

    def forward(self, x):
        xq = self.quantum(x)
        f1 = self.enc1(xq); f2 = self.enc2(f1)
        ft = self.transformer(f2)
        fr = self.res_blocks(ft)
        return self.dec2(self.dec1(fr))


class QIDLNoTransformer(nn.Module):
    """Ablation: QIDL without transformer (pure CNN)."""
    def __init__(self, base_ch=64, n_res=4):
        super().__init__()
        self.quantum    = QuantumEncoding()
        self.enc1       = nn.Sequential(
            nn.Conv2d(6, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.enc2       = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*2), nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(*[ResBlock(base_ch*2) for _ in range(n_res)])
        self.cbam       = CBAM(base_ch*2)
        self.dec1       = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.dec2       = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh())

    def forward(self, x):
        xq = self.quantum(x)
        f1 = self.enc1(xq); f2 = self.enc2(f1)
        fr = self.cbam(self.res_blocks(f2))
        return self.dec2(self.dec1(fr))


# ─────────────────────────────────────────────
# 15. MAIN — DEMO WITH SYNTHETIC DATA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("QIDL: Quantum-Inspired Deep Learning for UIE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Synthetic demo dataset (random tensors) ────────────────
    class SyntheticDataset(Dataset):
        def __init__(self, n=200, size=256):
            self.n = n; self.s = size
        def __len__(self): return self.n
        def __getitem__(self, _):
            # Simulate degraded underwater (cyan-shifted, low contrast)
            img = torch.rand(3, self.s, self.s)
            img[0] *= 0.4; img[1] *= 0.7    # suppress red, partial green
            # Ground truth: balanced channels
            gt  = torch.rand(3, self.s, self.s)
            # Normalise to [-1, 1]
            img = img * 2 - 1
            gt  = gt  * 2 - 1
            return img, gt

    ds    = SyntheticDataset(n=240)
    n_tr  = int(0.7 * len(ds))
    n_val = int(0.15 * len(ds))
    n_te  = len(ds) - n_tr - n_val
    tr_ds, val_ds, te_ds = torch.utils.data.random_split(ds, [n_tr, n_val, n_te])

    tr_loader  = DataLoader(tr_ds,  batch_size=8, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    te_loader  = DataLoader(te_ds,  batch_size=1, shuffle=False, num_workers=0)

    # ── Model, optimiser, scheduler ───────────────────────────
    model     = QIDLModel(base_ch=64, n_res=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Train ─────────────────────────────────────────────────
    history = train_qidl(
        model, tr_loader, val_loader, optimizer, scheduler,
        device, epochs=30, early_stop_patience=7,
        save_path="qidl_best.pth"
    )

    # ── Test evaluation ────────────────────────────────────────
    model.load_state_dict(torch.load("qidl_best.pth", map_location=device))
    psnr_list, ssim_list, uiqm_list, uciqe_list = evaluate(model, te_loader, device)

    print("\n─── Test Set Results ───")
    print(f"PSNR  : {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"SSIM  : {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print(f"UIQM  : {np.mean(uiqm_list):.4f} ± {np.std(uiqm_list):.4f}")
    print(f"UCIQE : {np.mean(uciqe_list):.4f} ± {np.std(uciqe_list):.4f}")

    # ── Save graphs ─────────────────────────────────────────
    save_graphs(history, psnr_list, ssim_list, uiqm_list, uciqe_list)
    print("\nQIDL pipeline complete.")
