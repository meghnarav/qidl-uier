"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QIDL: Quantum-Inspired Deep Learning for Underwater Image Enhancement       ║
║  Author : Meghna Ravikumar                                                   ║
║  Repo   : https://github.com/meghnarav/qidl-uier                             ║
║  Python : 3.9+  |  PyTorch : 2.1+                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

QUICK START
-----------
  # 1. Install dependencies
      pip install torch torchvision scikit-image matplotlib pillow numpy

  # 2. Demo (no dataset needed — runs on synthetic data)
      python qidl_final.py

  # 3. Real dataset training
      python qidl_final.py --mode train \\
          --train_input data/train/input --train_gt data/train/gt \\
          --val_input   data/val/input   --val_gt   data/val/gt   \\
          --test_input  data/test/input  --test_gt  data/test/gt

  # 4. Enhance a single image
      python qidl_final.py --mode infer \\
          --checkpoint qidl_best.pth    \\
          --input_image your_image.jpg  \\
          --output_image enhanced.jpg
"""

# ── standard library ──────────────────────────────────────────────────────────
import os, math, random, argparse, warnings
warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── torch ─────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── scikit-image (optional) ───────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as _ssim_sk
    SKIMAGE = True
except ImportError:
    SKIMAGE = False

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — DATASET
# ══════════════════════════════════════════════════════════════════════════════

class UnderwaterDataset(Dataset):
    """
    Paired underwater image dataset loader.

    Parameters
    ----------
    input_dir : str   – degraded underwater images
    gt_dir    : str   – ground-truth/reference images (None for unpaired)
    transform : callable
    """
    def __init__(self, input_dir, gt_dir=None, transform=None):
        self.inputs = sorted(
            [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if f.lower().endswith(IMG_EXTS)]
        )
        self.gts = (
            sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                    if f.lower().endswith(IMG_EXTS)])
            if gt_dir and os.path.isdir(gt_dir) else None
        )
        self.transform = transform
        if self.gts:
            assert len(self.inputs) == len(self.gts), (
                f"Mismatch: {len(self.inputs)} inputs vs {len(self.gts)} GT images."
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img = Image.open(self.inputs[idx]).convert("RGB")
        gt  = Image.open(self.gts[idx]).convert("RGB") if self.gts else img.copy()
        if self.transform:
            seed = random.randint(0, 2**31)
            random.seed(seed); torch.manual_seed(seed); img = self.transform(img)
            random.seed(seed); torch.manual_seed(seed); gt  = self.transform(gt)
        return img, gt


# ── transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),   # → [-1, 1]
]) if TORCH_AVAILABLE else None

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
]) if TORCH_AVAILABLE else None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — QUANTUM-INSPIRED FEATURE ENCODING  (Core novelty)
# ══════════════════════════════════════════════════════════════════════════════

class QuantumEncoding(nn.Module):
    """
    Quantum-inspired amplitude encoding (no learnable parameters).

    Each normalised pixel intensity p ∈ [0,1] is mapped to:
        α = cos(π·p)   — real amplitude
        β = sin(π·p)   — imaginary amplitude

    Input:  (B, 3, H, W)  in [-1, 1]
    Output: (B, 6, H, W)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x01   = (x + 1.0) / 2.0        # re-map to [0,1]
        theta = math.pi * x01
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — ATTENTION MODULES  (CBAM)
# ══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size(0), x.size(1)
        y = self.sig(
            self.mlp(self.avg(x).view(b, c)) +
            self.mlp(self.mx(x).view(b, c))
        ).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial attention from channel-pooled descriptors."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel → spatial)."""
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — VISION TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Lightweight ViT block for global spatial context modelling.
    Tokens = flattened spatial positions of the feature map.
    """
    def __init__(self, dim: int, num_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(0, 2, 1)   # (B, HW, C)
        n = self.norm1(seq)
        a, _ = self.attn(n, n, n)
        seq = seq + a
        seq = seq + self.mlp(self.norm2(seq))
        return seq.permute(0, 2, 1).view(b, c, h, w)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — RESIDUAL BLOCK WITH CBAM
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.cbam = CBAM(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.cbam(self.body(x)))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — FULL QIDL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class QIDLModel(nn.Module):
    """
    Quantum-Inspired Deep Learning (QIDL) model.

    Pipeline
    --------
    Input (3-ch) → QuantumEncoding (6-ch)
      → Encoder: Conv×2 (6→64→128)
      → TransformerBlock (global attention)
      → ResBlock×n_res with CBAM (local refinement)
      → Final CBAM
      → Decoder: Conv×2 (128→64→3) + Tanh
    Output (3-ch) in [-1,1]
    """
    def __init__(self, base_ch: int = 64, n_res: int = 4):
        super().__init__()
        ch2 = base_ch * 2

        self.quantum = QuantumEncoding()

        self.enc1 = nn.Sequential(
            nn.Conv2d(6, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, ch2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch2), nn.ReLU(inplace=True),
        )
        self.transformer = TransformerBlock(ch2)
        self.res_blocks  = nn.Sequential(*[ResBlock(ch2) for _ in range(n_res)])
        self.cbam        = CBAM(ch2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch2, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = self.quantum(x)
        f1 = self.enc1(xq)
        f2 = self.enc2(f1)
        ft = self.transformer(f2)
        fr = self.cbam(self.res_blocks(ft))
        return self.dec2(self.dec1(fr))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — ABLATION VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

class QIDLNoQuantum(nn.Module):
    """Ablation: standard RGB input, no quantum encoding."""
    def __init__(self, base_ch=64, n_res=4):
        super().__init__()
        ch2 = base_ch * 2
        self.enc1 = nn.Sequential(nn.Conv2d(3, base_ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, ch2, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(ch2), nn.ReLU(inplace=True))
        self.transformer = TransformerBlock(ch2)
        self.res_blocks  = nn.Sequential(*[ResBlock(ch2) for _ in range(n_res)])
        self.cbam = CBAM(ch2)
        self.dec1 = nn.Sequential(nn.Conv2d(ch2, base_ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh())
    def forward(self, x):
        f2 = self.enc2(self.enc1(x))
        return self.dec2(self.dec1(self.cbam(self.res_blocks(self.transformer(f2)))))


class QIDLNoTransformer(nn.Module):
    """Ablation: quantum encoding + CBAM, no transformer block."""
    def __init__(self, base_ch=64, n_res=4):
        super().__init__()
        ch2 = base_ch * 2
        self.quantum = QuantumEncoding()
        self.enc1 = nn.Sequential(nn.Conv2d(6, base_ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, ch2, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(ch2), nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(*[ResBlock(ch2) for _ in range(n_res)])
        self.cbam = CBAM(ch2)
        self.dec1 = nn.Sequential(nn.Conv2d(ch2, base_ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh())
    def forward(self, x):
        f2 = self.enc2(self.enc1(self.quantum(x)))
        return self.dec2(self.dec1(self.cbam(self.res_blocks(f2))))


class QIDLNoCBAM(nn.Module):
    """Ablation: quantum encoding + transformer, no CBAM."""
    def __init__(self, base_ch=64, n_res=4):
        super().__init__()
        ch2 = base_ch * 2
        self.quantum = QuantumEncoding()
        self.enc1 = nn.Sequential(nn.Conv2d(6, base_ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, ch2, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(ch2), nn.ReLU(inplace=True))
        self.transformer = TransformerBlock(ch2)
        plain = []
        for _ in range(n_res):
            plain += [nn.Conv2d(ch2, ch2, 3, padding=1, bias=False),
                      nn.BatchNorm2d(ch2), nn.ReLU(inplace=True)]
        self.res_blocks = nn.Sequential(*plain)
        self.dec1 = nn.Sequential(nn.Conv2d(ch2, base_ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch, 3, 3, padding=1), nn.Tanh())
    def forward(self, x):
        f2 = self.enc2(self.enc1(self.quantum(x)))
        return self.dec2(self.dec1(self.res_blocks(self.transformer(f2))))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION H — LOSS FUNCTIONS  (Physics-Guided Composite)
# ══════════════════════════════════════════════════════════════════════════════

def loss_mse(out, gt):
    """L2 pixel reconstruction loss."""
    return F.mse_loss(out, gt)


def loss_ssim(out, gt, C1=1e-4, C2=9e-4):
    """Differentiable SSIM loss = 1 - SSIM."""
    mu_x  = F.avg_pool2d(out, 3, 1, 1)
    mu_y  = F.avg_pool2d(gt,  3, 1, 1)
    sg_x  = F.avg_pool2d(out**2, 3, 1, 1) - mu_x**2
    sg_y  = F.avg_pool2d(gt**2,  3, 1, 1) - mu_y**2
    sg_xy = F.avg_pool2d(out*gt, 3, 1, 1) - mu_x*mu_y
    num   = (2*mu_x*mu_y + C1) * (2*sg_xy + C2)
    den   = (mu_x**2 + mu_y**2 + C1) * (sg_x + sg_y + C2)
    return 1.0 - (num / den.clamp(min=1e-8)).mean()


def loss_colour(out, gt):
    """Per-channel L1 colour consistency (physics: wavelength absorption)."""
    return sum(F.l1_loss(out[:, c], gt[:, c]) for c in range(3)) / 3.0


def loss_gradient(out, gt):
    """Gradient-domain edge preservation loss."""
    dx_o = out[:, :, :, 1:] - out[:, :, :, :-1]
    dy_o = out[:, :, 1:, :] - out[:, :, :-1, :]
    dx_g = gt[:, :, :, 1:]  - gt[:, :, :, :-1]
    dy_g = gt[:, :, 1:, :]  - gt[:, :, :-1, :]
    return F.l1_loss(dx_o, dx_g) + F.l1_loss(dy_o, dy_g)


def total_loss(out, gt, w=(0.5, 0.3, 0.1, 0.1)):
    """
    Physics-guided composite loss:
        L = w0·L_MSE + w1·L_SSIM + w2·L_Colour + w3·L_Gradient
    """
    return (w[0] * loss_mse(out, gt) +
            w[1] * loss_ssim(out, gt) +
            w[2] * loss_colour(out, gt) +
            w[3] * loss_gradient(out, gt))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION I — METRICS
# ══════════════════════════════════════════════════════════════════════════════

def metric_psnr(out: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR in dB. Tensors in [-1,1] → max range = 2."""
    mse = F.mse_loss(out, gt).item()
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(4.0 / mse)


def metric_ssim(out: torch.Tensor, gt: torch.Tensor) -> float:
    """Per-image SSIM averaged over batch."""
    out_np = ((out.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    gt_np  = ((gt.detach().cpu().numpy()  + 1) / 2).clip(0, 1)
    if SKIMAGE:
        scores = [
            _ssim_sk(out_np[i].transpose(1, 2, 0),
                     gt_np[i].transpose(1, 2, 0),
                     channel_axis=2, data_range=1.0)
            for i in range(out_np.shape[0])
        ]
    else:
        # Fast differentiable approximation when skimage unavailable
        with torch.no_grad():
            scores = [1.0 - loss_ssim(out, gt).item()]
    return float(np.mean(scores))


def metric_uiqm(img: torch.Tensor) -> float:
    """
    UIQM = c1·UICM + c2·UISM + c3·UIConM
    Coefficients: Panetta et al. (2016).
    """
    arr = ((img.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    scores = []
    for b in range(arr.shape[0]):
        im = arr[b].transpose(1, 2, 0)
        r, g, bl = im[..., 0], im[..., 1], im[..., 2]
        uicm  = float(np.std(r - g) + np.std(r - bl))
        dx, dy = im[:, 1:, :] - im[:, :-1, :], im[1:, :, :] - im[:-1, :, :]
        uism  = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))
        uicon = float(im.max() - im.min())
        scores.append(c1*uicm + c2*uism + c3*uicon)
    return float(np.mean(scores))


def metric_uciqe(img: torch.Tensor) -> float:
    """UCIQE = c1·σ_c + c2·con_l + c3·μ_s"""
    arr = ((img.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    scores = []
    for b in range(arr.shape[0]):
        im = arr[b].transpose(1, 2, 0)
        r, g, bl = im[..., 0], im[..., 1], im[..., 2]
        chroma = np.sqrt((r-g)**2 + (r-bl)**2 + (g-bl)**2)
        lum    = 0.299*r + 0.587*g + 0.114*bl
        scores.append(c1*np.std(chroma) + c2*(lum.max()-lum.min()) + c3*np.mean(chroma))
    return float(np.mean(scores))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION J — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_qidl(model, train_loader, val_loader,
               optimizer, scheduler, device,
               epochs=30, patience=7,
               ckpt="qidl_best.pth"):
    """
    Full training loop with:
        • physics-guided composite loss
        • gradient clipping (norm 1.0)
        • ReduceLROnPlateau scheduling
        • early stopping on validation PSNR
        • best-model checkpointing
    Returns history dict.
    """
    history = {k: [] for k in
               ("train_loss","val_loss","psnr","ssim","uiqm","uciqe")}
    best_psnr, wait = 0.0, 0

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for img, gt in train_loader:
            img, gt = img.to(device), gt.to(device)
            loss = total_loss(model(img), gt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        v_loss = psnr_s = ssim_s = uiqm_s = uciqe_s = 0.0
        n = len(val_loader)
        with torch.no_grad():
            for img, gt in val_loader:
                img, gt = img.to(device), gt.to(device)
                out = model(img)
                v_loss  += total_loss(out, gt).item()
                psnr_s  += metric_psnr(out, gt)
                ssim_s  += metric_ssim(out, gt)
                uiqm_s  += metric_uiqm(out)
                uciqe_s += metric_uciqe(out)
        v_loss  /= n
        psnr_avg = psnr_s  / n
        ssim_avg = ssim_s  / n
        uiqm_avg = uiqm_s  / n
        uciqe_avg= uciqe_s / n

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["psnr"].append(psnr_avg)
        history["ssim"].append(ssim_avg)
        history["uiqm"].append(uiqm_avg)
        history["uciqe"].append(uciqe_avg)

        scheduler.step(v_loss)
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train {t_loss:.4f} | Val {v_loss:.4f} | "
              f"PSNR {psnr_avg:.2f} dB | SSIM {ssim_avg:.4f} | "
              f"UIQM {uiqm_avg:.4f} | UCIQE {uciqe_avg:.4f}")

        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            torch.save(model.state_dict(), ckpt)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    print(f"\nBest val PSNR: {best_psnr:.2f} dB  (checkpoint: {ckpt})")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION K — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device):
    """Returns per-image metric lists for statistical reporting."""
    model.eval()
    psnr_l, ssim_l, uiqm_l, uciqe_l = [], [], [], []
    with torch.no_grad():
        for img, gt in loader:
            img, gt = img.to(device), gt.to(device)
            out = model(img)
            psnr_l.append(metric_psnr(out, gt))
            ssim_l.append(metric_ssim(out, gt))
            uiqm_l.append(metric_uiqm(out))
            uciqe_l.append(metric_uciqe(out))
    return psnr_l, ssim_l, uiqm_l, uciqe_l


# ══════════════════════════════════════════════════════════════════════════════
# SECTION L — SINGLE-IMAGE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def enhance_image(model, image_path: str, device,
                  save_path: str = None) -> Image.Image:
    """
    Enhance a single degraded underwater image.
    Returns PIL Image at original resolution.
    """
    model.eval()
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img  = Image.open(image_path).convert("RGB")
    orig = img.size
    inp  = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    out_np = ((out.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)
    result = Image.fromarray(
        (out_np * 255).astype(np.uint8).transpose(1, 2, 0)
    ).resize(orig, Image.BICUBIC)
    if save_path:
        result.save(save_path)
        print(f"Saved: {save_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION M — GRAPH GENERATION  (9 plots for paper)
# ══════════════════════════════════════════════════════════════════════════════

def save_graphs(history, psnr_list, ssim_list,
                uiqm_list, uciqe_list, out_dir="graphs"):
    os.makedirs(out_dir, exist_ok=True)
    ep = range(1, len(history["train_loss"]) + 1)
    kw = dict(linewidth=2.0, marker="o", markersize=3)

    curves = [
        ("graph1_train_loss.png", history["train_loss"], "#1565C0", "Training Loss"),
        ("graph2_val_loss.png",   history["val_loss"],   "#C62828", "Validation Loss"),
        ("graph4_psnr_curve.png", history["psnr"],       "#2E7D32", "PSNR (dB)"),
        ("graph5_ssim_curve.png", history["ssim"],       "#E65100", "SSIM"),
        ("graph8_uiqm_curve.png", history["uiqm"],       "#00695C", "UIQM"),
        ("graph9_uciqe_curve.png",history["uciqe"],      "#BF360C", "UCIQE"),
    ]
    for fname, data, col, ylabel in curves:
        plt.figure(figsize=(7, 4))
        plt.plot(ep, data, color=col, **kw)
        plt.xlabel("Epoch"); plt.ylabel(ylabel)
        plt.title(f"{ylabel} over Epochs — QIDL"); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150); plt.close()

    # Graph 3: combined
    plt.figure(figsize=(7, 4))
    plt.plot(ep, history["train_loss"], label="Train",      color="#1565C0", **kw)
    plt.plot(ep, history["val_loss"],   label="Validation", color="#C62828",
             linestyle="--", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Train vs. Validation Loss — QIDL"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph3_train_val.png"), dpi=150); plt.close()

    # Graphs 6–7: test distributions
    for fname, data, col, xlabel in [
        ("graph6_psnr_dist.png", psnr_list, "#1565C0", "PSNR (dB)"),
        ("graph7_ssim_dist.png", ssim_list, "#7B1FA2", "SSIM"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.hist(data, bins=20, color=col, edgecolor="white", alpha=0.85)
        plt.axvline(np.mean(data), color="red", linestyle="--",
                    label=f"Mean={np.mean(data):.4f}")
        plt.xlabel(xlabel); plt.ylabel("Frequency")
        plt.title(f"{xlabel} Distribution — Test Set"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150); plt.close()

    print(f"9 graphs saved → {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION N — SYNTHETIC DATASET  (for demo without real data)
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticUnderwaterDataset(Dataset):
    """
    Synthetic demo dataset that simulates underwater colour degradation.
    Requires no disk data — used to verify the full pipeline runs correctly.
    """
    def __init__(self, n: int = 200, size: int = 256):
        self.n, self.s = n, size

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        # Simulate degraded underwater image (strong blue-green, weak red)
        img = torch.rand(3, self.s, self.s)
        img[0] = img[0] * 0.35 - 0.3    # red channel heavily attenuated
        img[1] = img[1] * 0.65 - 0.1    # green partially attenuated
        img[2] = img[2] * 0.90           # blue dominant
        img = img.clamp(-1, 1)

        # Ground truth: balanced, high-quality image
        gt = torch.rand(3, self.s, self.s) * 2 - 1
        return img, gt


# ══════════════════════════════════════════════════════════════════════════════
# SECTION O — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="QIDL: Quantum-Inspired Deep Learning for UIE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--mode", default="demo",
                   choices=["demo", "train", "infer"],
                   help="demo=synthetic | train=real dataset | infer=single image")
    # Dataset paths (used in train mode)
    p.add_argument("--train_input", default="data/train/input")
    p.add_argument("--train_gt",    default="data/train/gt")
    p.add_argument("--val_input",   default="data/val/input")
    p.add_argument("--val_gt",      default="data/val/gt")
    p.add_argument("--test_input",  default="data/test/input")
    p.add_argument("--test_gt",     default="data/test/gt")
    # Training hyperparameters
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--n_res",       type=int,   default=4)
    p.add_argument("--patience",    type=int,   default=7)
    p.add_argument("--checkpoint",  default="qidl_best.pth")
    p.add_argument("--graph_dir",   default="graphs")
    # Inference
    p.add_argument("--input_image",  default=None)
    p.add_argument("--output_image", default="enhanced.jpg")
    return p.parse_args()


def main():
    args = get_args()

    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch not found. Install with:")
        print("        pip install torch torchvision")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  QIDL: Quantum-Inspired Deep Learning for UIE")
    print(f"  Device : {device}")
    print(f"  Mode   : {args.mode}")
    print(f"{'='*60}\n")

    # ── INFER MODE ──────────────────────────────────────────────────────────
    if args.mode == "infer":
        assert args.input_image, "--input_image required for infer mode"
        model = QIDLModel(args.base_ch, args.n_res).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        enhance_image(model, args.input_image, device, args.output_image)
        return

    # ── BUILD DATA LOADERS ──────────────────────────────────────────────────
    if args.mode == "demo":
        print("Running synthetic demo (no dataset required).")
        ds  = SyntheticUnderwaterDataset(n=200)
        n_tr, n_va = int(0.7*len(ds)), int(0.15*len(ds))
        n_te = len(ds) - n_tr - n_va
        tr_ds, va_ds, te_ds = torch.utils.data.random_split(
            ds, [n_tr, n_va, n_te],
            generator=torch.Generator().manual_seed(42)
        )
        kw = dict(num_workers=0, pin_memory=False)
    else:
        tr_ds = UnderwaterDataset(args.train_input, args.train_gt, train_transform)
        va_ds = UnderwaterDataset(args.val_input,   args.val_gt,   eval_transform)
        te_ds = UnderwaterDataset(args.test_input,  args.test_gt,  eval_transform)
        kw    = dict(num_workers=4, pin_memory=device.type == "cuda")

    tr_loader = DataLoader(tr_ds, args.batch_size, shuffle=True,  **kw)
    va_loader = DataLoader(va_ds, args.batch_size, shuffle=False, **kw)
    te_loader = DataLoader(te_ds, 1,               shuffle=False, **kw)

    # ── MODEL ───────────────────────────────────────────────────────────────
    model = QIDLModel(args.base_ch, args.n_res).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False)

    # ── TRAIN ───────────────────────────────────────────────────────────────
    history = train_qidl(
        model, tr_loader, va_loader, optimizer, scheduler, device,
        epochs=args.epochs, patience=args.patience,
        ckpt=args.checkpoint,
    )

    # ── EVALUATE ────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    psnr_l, ssim_l, uiqm_l, uciqe_l = evaluate(model, te_loader, device)

    print("\n" + "─"*60)
    print("  Test Set Results")
    print("─"*60)
    print(f"  PSNR  : {np.mean(psnr_l):.2f} ± {np.std(psnr_l):.2f} dB")
    print(f"  SSIM  : {np.mean(ssim_l):.4f} ± {np.std(ssim_l):.4f}")
    print(f"  UIQM  : {np.mean(uiqm_l):.4f} ± {np.std(uiqm_l):.4f}")
    print(f"  UCIQE : {np.mean(uciqe_l):.4f} ± {np.std(uciqe_l):.4f}")
    print("─"*60 + "\n")

    # ── GRAPHS ──────────────────────────────────────────────────────────────
    save_graphs(history, psnr_l, ssim_l, uiqm_l, uciqe_l, args.graph_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
