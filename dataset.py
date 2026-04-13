import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config


# BUG FIX 1: was hardcoded to Resize(128,128) — should use Config.img_size (256)
# BUG FIX 2: missing Normalize — model expects [-1,1] but dataset returned [0,1]

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _sorted_images(folder: str) -> list:
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(_EXTS)
    )


class UIEBDataset(Dataset):
    """
    Paired underwater image dataset.

    Parameters
    ----------
    inp_dir : str   – folder of degraded/raw underwater images
    gt_dir  : str   – folder of ground-truth/reference images
    transform       – if None, uses default train transform
    augment : bool  – apply random flips/rotation (train only)
    """

    def __init__(self, inp_dir: str, gt_dir: str,
                 transform=None, augment: bool = False):
        self.inp_paths = _sorted_images(inp_dir)
        self.gt_paths  = _sorted_images(gt_dir)
        self.augment   = augment

        assert len(self.inp_paths) == len(self.gt_paths), (
            f"Mismatch: {len(self.inp_paths)} inputs vs "
            f"{len(self.gt_paths)} ground-truth files."
        )

        # Default transform: resize to Config.img_size, normalise to [-1,1]
        self.transform = transform or transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Augmentation applied identically to both images via shared seed
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.inp_paths)

    def __getitem__(self, idx: int):
        inp = Image.open(self.inp_paths[idx]).convert("RGB")
        gt  = Image.open(self.gt_paths[idx]).convert("RGB")

        if self.aug is not None:
            # Same random seed → identical spatial augmentation on both images
            seed = random.randint(0, 2 ** 31)
            random.seed(seed); torch.manual_seed(seed)
            inp = self.aug(inp)
            random.seed(seed); torch.manual_seed(seed)
            gt  = self.aug(gt)

        return self.transform(inp), self.transform(gt)
