import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Quantum Encoding ----------
class QuantumEncoding(nn.Module):
    def forward(self, x):
        x = (x + 1) / 2
        theta = math.pi * x
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)


# ---------- CBAM ----------
class ChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c, c//8),
            nn.ReLU(),
            nn.Linear(c//8, c),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,_,_ = x.shape
        y = x.mean((2,3))
        y = self.mlp(y).view(b,c,1,1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3)

    def forward(self,x):
        avg = x.mean(1,True)
        mx,_ = x.max(1,True)
        x = torch.cat([avg,mx],1)
        return x * torch.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()
    def forward(self,x):
        return self.sa(self.ca(x))


# ---------- Transformer ----------
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim,4,batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim)
        )

    def forward(self,x):
        b,c,h,w = x.shape
        x = F.adaptive_avg_pool2d(x, (32, 32))
        x = x.view(b,c,h*w).permute(0,2,1)
        a,_ = self.attn(x,x,x)
        x = self.ln1(x+a)
        x = self.ln2(x + self.mlp(x))
        x = x.permute(0,2,1).view(b,c,32, 32)
        return F.interpolate(x, size=(h, w))

# ---------- Residual ----------
class ResBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c,c,3,1,1),
            nn.ReLU(),
            nn.Conv2d(c,c,3,1,1)
        )
        self.cbam = CBAM(c)

    def forward(self,x):
        return x + self.cbam(self.block(x))


# ---------- Full QIDL ----------
class QIDL(nn.Module):
    def __init__(self, base=64, n_res=4, C=64):
        super().__init__()
        self.q = QuantumEncoding()

        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)

        self.tr = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 128→64x64
            TransformerBlock(128),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)  # back
        )   
        self.res = nn.Sequential(*[ResBlock(base*2) for _ in range(n_res)])

        self.dec = nn.Sequential(
            nn.Conv2d(base*2,base,3,1,1),
            nn.ReLU(),
            nn.Conv2d(base,3,3,1,1),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.q(x)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.tr(x)
        x = self.res(x)
        return self.dec(x)