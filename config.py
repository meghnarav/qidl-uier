import torch


class Config:
    img_size    = 256          # spatial resolution (H = W)
    batch_size  = 8
    lr          = 1e-4
    epochs      = 30
    num_workers = 4
    seed        = 42
    base_ch     = 64
    n_res       = 4
    patience    = 7            # early-stopping patience (epochs)
    checkpoint  = "qidl.pth"  # where to save the best model

    # BUG FIX: original hardcoded "cuda" — crashes on CPU-only machines.
    # Now auto-detected.
    device = "cuda" if torch.cuda.is_available() else "cpu"
