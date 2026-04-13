import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.metrics import structural_similarity as _ssim_sk
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


def psnr(out: torch.Tensor, gt: torch.Tensor) -> float:
    """
    PSNR in dB. Tensors expected in [-1, 1] so max range = 2.
    PSNR = 10 * log10(max_val² / MSE)  where max_val = 2.
    """
    mse = F.mse_loss(out.detach(), gt.detach()).item()
    return 100.0 if mse < 1e-10 else 10.0 * math.log10(4.0 / mse)


def ssim(out: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Per-image SSIM averaged over the batch.
    BUG FIX: added data_range=1.0 (was missing → wrong values in
    newer scikit-image which cannot infer range from float arrays).
    Falls back to a fast differentiable approximation if skimage
    is not installed.
    """
    # Convert [-1,1] → [0,1] and clamp for safety
    out_np = out.detach().cpu().float()
    gt_np  = gt.detach().cpu().float()
    out_np = ((out_np + 1.0) / 2.0).clamp(0, 1).numpy()
    gt_np  = ((gt_np  + 1.0) / 2.0).clamp(0, 1).numpy()

    if _SKIMAGE:
        scores = [
            _ssim_sk(
                out_np[i].transpose(1, 2, 0),
                gt_np[i].transpose(1, 2, 0),
                channel_axis=2,
                data_range=1.0,           # ← the missing argument
            )
            for i in range(out_np.shape[0])
        ]
        return float(np.mean(scores))

    # Fallback: differentiable approximation (no skimage needed)
    out_t = torch.from_numpy(out_np)
    gt_t  = torch.from_numpy(gt_np)
    mu_x  = F.avg_pool2d(out_t, 3, 1, 1)
    mu_y  = F.avg_pool2d(gt_t,  3, 1, 1)
    sg_x  = F.avg_pool2d(out_t**2, 3, 1, 1) - mu_x**2
    sg_y  = F.avg_pool2d(gt_t**2,  3, 1, 1) - mu_y**2
    sg_xy = F.avg_pool2d(out_t*gt_t, 3, 1, 1) - mu_x*mu_y
    C1, C2 = 1e-4, 9e-4
    num = (2*mu_x*mu_y + C1) * (2*sg_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sg_x + sg_y + C2)
    return float((num / den.clamp(min=1e-8)).mean().item())
