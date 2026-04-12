import os, random
from PIL import Image
from torch.utils.data import Dataset

class UIEBDataset(Dataset):
    def __init__(self, inp, gt, transform=None):
        self.inp = sorted(os.listdir(inp))
        self.gt = sorted(os.listdir(gt))
        self.inp = [os.path.join(inp, x) for x in self.inp]
        self.gt = [os.path.join(gt, x) for x in self.gt]
        self.t = transform

    def __len__(self): return len(self.inp)

    def __getitem__(self, i):
        img = Image.open(self.inp[i]).convert("RGB")
        gt  = Image.open(self.gt[i]).convert("RGB")
        if self.t:
            seed = random.randint(0, 99999)
            random.seed(seed)
            img = self.t(img)
            random.seed(seed)
            gt = self.t(gt)
        return img, gt