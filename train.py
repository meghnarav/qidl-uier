import torch
from torch.utils.data import DataLoader, random_split

from config  import Config
from dataset import UIEBDataset
from models  import QIDL
from losses  import total
from metrics import psnr, ssim
from utils   import set_seed, save_model, count_params

# ── Reproducibility ───────────────────────────────────────────
set_seed(Config.seed)

# ── Device ────────────────────────────────────────────────────
# BUG FIX: original used Config.device (string "cuda") directly;
# torch.device() must wrap it for proper device-move semantics.
device = torch.device(Config.device)
print(f"[INFO] Device : {device}")

# ── Dataset ───────────────────────────────────────────────────
full_ds = UIEBDataset(
    "data/train/input",
    "data/train/gt",
    augment=True,         # random flip/rotation on training set
)
n      = len(full_ds)
n_tr   = int(0.8 * n)
n_va   = int(0.1 * n)
n_te   = n - n_tr - n_va

tr_ds, va_ds, _ = random_split(
    full_ds, [n_tr, n_va, n_te],
    generator=torch.Generator().manual_seed(Config.seed),
)

kw = dict(num_workers=Config.num_workers,
          pin_memory=(device.type == "cuda"))

tr_loader = DataLoader(tr_ds, Config.batch_size, shuffle=True,  **kw)
va_loader = DataLoader(va_ds, Config.batch_size, shuffle=False, **kw)

# ── Model ─────────────────────────────────────────────────────
model = QIDL(base=Config.base_ch, n_res=Config.n_res).to(device)
print(f"[INFO] Parameters: {count_params(model):,}")

opt       = torch.optim.Adam(model.parameters(),
                             lr=Config.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=3, verbose=False,
)

# ── Training loop ─────────────────────────────────────────────
best_psnr = 0.0
wait      = 0

for ep in range(1, Config.epochs + 1):

    # ── train ──────────────────────────────────────────────
    model.train()
    t_loss = 0.0
    for x, y in tr_loader:
        x, y = x.to(device), y.to(device)
        loss = total(model(x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        t_loss += loss.item()
    t_loss /= len(tr_loader)

    # ── validate ───────────────────────────────────────────
    model.eval()
    pv = sv = 0.0
    with torch.no_grad():
        for x, y in va_loader:
            x, y = x.to(device), y.to(device)
            out   = model(x)
            pv   += psnr(out, y)
            sv   += ssim(out, y)
    pv /= len(va_loader)
    sv /= len(va_loader)

    scheduler.step(t_loss)

    print(f"Epoch {ep:3d}/{Config.epochs} | "
          f"Loss {t_loss:.4f} | PSNR {pv:.2f} dB | SSIM {sv:.4f}")

    # ── early stopping + checkpoint ────────────────────────
    if pv > best_psnr:
        best_psnr = pv
        save_model(model, Config.checkpoint)
        wait = 0
    else:
        wait += 1
        if wait >= Config.patience:
            print(f"[INFO] Early stopping at epoch {ep}.")
            break

print(f"[INFO] Best PSNR : {best_psnr:.2f} dB")
print(f"[INFO] Model saved → {Config.checkpoint}")
