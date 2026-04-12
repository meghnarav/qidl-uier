import torch
import random
import numpy as np
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)