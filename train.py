import torch
from torch.utils.data import DataLoader, random_split
from dataset import UIEBDataset
from models import QIDL
from losses import total
from metrics import psnr, ssim
from config import Config

# ─────────────────────────────────────────────
# DEVICE FIX (IMPORTANT)
# ─────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[INFO] Using device: {device}")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
ds = UIEBDataset("data/train/input", "data/train/gt")

n = len(ds)
tr_len = int(0.8 * n)
va_len = int(0.1 * n)
te_len = n - tr_len - va_len

tr_ds, va_ds, _ = random_split(
    ds,
    [tr_len, va_len, te_len],
    generator=torch.Generator().manual_seed(42)
)

tr = DataLoader(tr_ds, batch_size=Config.batch_size, shuffle=True)
va = DataLoader(va_ds, batch_size=Config.batch_size, shuffle=False)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = QIDL().to(device)

opt = torch.optim.Adam(model.parameters(), lr=Config.lr)

# ─────────────────────────────────────────────
# TRAIN LOOP
# ─────────────────────────────────────────────
for ep in range(Config.epochs):
    model.train()
    tl = 0

    for x, y in tr:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = total(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tl += loss.item()

    # ─────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────
    model.eval()
    pv, sv = 0, 0

    with torch.no_grad():
        for x, y in va:
            x, y = x.to(device), y.to(device)
            out = model(x)

            pv += psnr(out, y)
            sv += ssim(out, y)

    pv /= len(va)
    sv /= len(va)

    print(
        f"Epoch {ep+1}/{Config.epochs} | "
        f"Loss: {tl/len(tr):.4f} | "
        f"PSNR: {pv:.2f} | SSIM: {sv:.4f}"
    )

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
torch.save(model.state_dict(), "qidl.pth")
print("[INFO] Model saved → qidl.pth")