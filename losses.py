import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# ── VGG perceptual feature extractor ─────────────────────────
# BUG FIX 1: pretrained=True is deprecated → use weights=
# BUG FIX 2: vgg defined at module level on CPU; calling
#            perceptual() with GPU tensors caused device-mismatch
#            RuntimeError. Now moved lazily to correct device.
_vgg_cache: dict = {}

def _get_vgg(device: torch.device) -> nn.Module:
    key = str(device)
    if key not in _vgg_cache:
        vgg = tvm.vgg16(
            weights=tvm.VGG16_Weights.DEFAULT   # replaces pretrained=True
        ).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        _vgg_cache[key] = vgg.to(device)
    return _vgg_cache[key]


# ── Individual loss functions ─────────────────────────────────
def mse_loss(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Pixel-level L2 reconstruction loss."""
    return F.mse_loss(out, gt)


def ssim_loss(out: torch.Tensor, gt: torch.Tensor,
              C1: float = 1e-4, C2: float = 9e-4) -> torch.Tensor:
    """Differentiable SSIM loss = 1 − SSIM(out, gt)."""
    mu_x  = F.avg_pool2d(out, 3, 1, 1)
    mu_y  = F.avg_pool2d(gt,  3, 1, 1)
    sg_x  = F.avg_pool2d(out ** 2, 3, 1, 1) - mu_x ** 2
    sg_y  = F.avg_pool2d(gt  ** 2, 3, 1, 1) - mu_y ** 2
    sg_xy = F.avg_pool2d(out * gt, 3, 1, 1) - mu_x * mu_y
    num   = (2 * mu_x * mu_y + C1) * (2 * sg_xy + C2)
    den   = (mu_x ** 2 + mu_y ** 2 + C1) * (sg_x + sg_y + C2)
    return 1.0 - (num / den.clamp(min=1e-8)).mean()


def colour_loss(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-channel L1 colour consistency loss."""
    return sum(F.l1_loss(out[:, c], gt[:, c]) for c in range(3)) / 3.0


def gradient_loss(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Gradient-domain edge preservation loss."""
    dx_o = out[:, :, :, 1:] - out[:, :, :, :-1]
    dy_o = out[:, :, 1:, :] - out[:, :, :-1, :]
    dx_g = gt[:, :, :, 1:]  - gt[:, :, :, :-1]
    dy_g = gt[:, :, 1:, :]  - gt[:, :, :-1, :]
    return F.l1_loss(dx_o, dx_g) + F.l1_loss(dy_o, dy_g)


def perceptual_loss(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """VGG-16 perceptual loss (feature-level L1)."""
    vgg = _get_vgg(out.device)   # always same device as inputs
    return F.l1_loss(vgg(out), vgg(gt))


# ── Composite physics-guided total loss ───────────────────────
def total(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    L = 0.5·MSE + 0.2·SSIM + 0.2·Colour + 0.1·Perceptual
    """
    return (0.5 * mse_loss(out, gt)
          + 0.2 * ssim_loss(out, gt)
          + 0.2 * colour_loss(out, gt)
          + 0.1 * perceptual_loss(out, gt))
