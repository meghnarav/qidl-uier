import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Load once, move to device lazily in perceptual()
_vgg = None

def _get_vgg(device):
    global _vgg
    if _vgg is None:
        _vgg = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT   # replaces pretrained=True
        ).features[:16].eval()
        for p in _vgg.parameters():
            p.requires_grad = False
    return _vgg.to(device)   # moves to same device as input

def mse(out, gt):
    return F.mse_loss(out, gt)

def l1(out, gt):
    return F.l1_loss(out, gt)

def ssim_loss(out, gt):
    mu1 = F.avg_pool2d(out, 3, 1, 1)
    mu2 = F.avg_pool2d(gt,  3, 1, 1)
    return 1 - torch.mean((2 * mu1 * mu2 + 0.01) / (mu1**2 + mu2**2 + 0.01))

def perceptual(out, gt):
    vgg = _get_vgg(out.device)   # always on same device as tensors
    return F.l1_loss(vgg(out), vgg(gt))

def total(out, gt):
    return (0.5 * mse(out, gt)
          + 0.2 * l1(out, gt)
          + 0.2 * ssim_loss(out, gt)
          + 0.1 * perceptual(out, gt))