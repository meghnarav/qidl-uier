import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class UIEBDataset(Dataset):
    def __init__(self, inp_dir, gt_dir, transform=None):
        self.inp_paths = sorted([
            os.path.join(inp_dir, f)
            for f in os.listdir(inp_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.gt_paths = sorted([
            os.path.join(gt_dir, f)
            for f in os.listdir(gt_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.inp_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.inp_paths[idx]).convert("RGB")
        gt  = Image.open(self.gt_paths[idx]).convert("RGB")

        inp = self.transform(inp)
        gt  = self.transform(gt)

        return inp, gt