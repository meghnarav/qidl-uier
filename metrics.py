import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn

def psnr(out, gt):
    mse = F.mse_loss(out, gt).item()
    return 100 if mse < 1e-10 else 10 * np.log10(4 / mse)

def ssim(out, gt):
    o = ((out + 1) / 2).clamp(0, 1).detach().cpu().numpy()
    g = ((gt  + 1) / 2).clamp(0, 1).detach().cpu().numpy()
    return float(np.mean([
        ssim_fn(
            o[i].transpose(1, 2, 0),
            g[i].transpose(1, 2, 0),
            channel_axis=2,
            data_range=1.0   # was missing — caused wrong values
        )
        for i in range(o.shape[0])
    ]))