import torch
from torch.utils.data import DataLoader, random_split
from dataset import UIEBDataset
from models import QIDL
from losses import total
from metrics import psnr, ssim
from config import Config

device = Config.device

ds = UIEBDataset("data/input","data/gt")

n=len(ds)
tr,va,_ = random_split(ds,[int(0.8*n),int(0.1*n),n-int(0.9*n)])

tr = DataLoader(tr,batch_size=Config.batch_size,shuffle=True)
va = DataLoader(va,batch_size=Config.batch_size)

model = QIDL().to(device)
opt = torch.optim.Adam(model.parameters(),lr=Config.lr)

for ep in range(Config.epochs):
    model.train()
    tl=0

    for x,y in tr:
        x,y=x.to(device),y.to(device)
        out=model(x)
        loss=total(out,y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tl+=loss.item()

    model.eval()
    pv=0
    sv=0

    with torch.no_grad():
        for x,y in va:
            x,y=x.to(device),y.to(device)
            out=model(x)
            pv+=psnr(out,y)
            sv+=ssim(out,y)

    print(ep, tl/len(tr), pv/len(va), sv/len(va))

torch.save(model.state_dict(),"qidl.pth")