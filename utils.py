import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def save_model(model: torch.nn.Module, path: str = "qidl.pth") -> None:
    torch.save(model.state_dict(), path)
    print(f"[INFO] Checkpoint saved → {path}")


def load_model(model: torch.nn.Module, path: str,
               device: torch.device) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)   # BUG FIX: exist_ok avoids race condition
