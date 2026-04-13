import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Quantum Encoding ──────────────────────────────────────────
class QuantumEncoding(nn.Module):
    """
    Parameter-free quantum-inspired amplitude encoding.
    Maps each pixel p ∈ [0,1] to [cos(π·p), sin(π·p)].
    Input : (B, 3, H, W)  in [-1, 1]
    Output: (B, 6, H, W)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x01   = (x + 1.0) / 2.0          # remap [-1,1] → [0,1]
        theta  = math.pi * x01
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)


# ── CBAM ──────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        # BUG FIX: use max(c//8, 1) to avoid 0-dim Linear when c < 8
        mid = max(c // 8, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # BUG FIX: use BOTH avg and max pool (original only used avg)
        avg = self.mlp(self.avg_pool(x).view(b, c))
        mx  = self.mlp(self.max_pool(x).view(b, c))
        y   = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg    = x.mean(dim=1, keepdim=True)
        mx, _  = x.max(dim=1, keepdim=True)
        # BUG FIX: original overwrote x before multiplying
        mask   = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * mask


class CBAM(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ── Transformer Block ─────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    Lightweight ViT block for global spatial context modelling.
    BUG FIX: original used self.ln1/ln2/mlp in __init__ but
    referenced self.norm1/norm2/ff in forward → AttributeError crash.
    Names are now consistent throughout.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq        = x.flatten(2).transpose(1, 2)   # (B, H*W, C)
        attn, _    = self.attn(seq, seq, seq)
        seq        = self.norm1(seq + attn)
        seq        = self.norm2(seq + self.ff(seq))
        return seq.transpose(1, 2).reshape(b, c, h, w)


# ── Residual Block ────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.cbam = CBAM(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.cbam(self.block(x)))


# ── Full QIDL Model ───────────────────────────────────────────
class QIDL(nn.Module):
    """
    Quantum-Inspired Deep Learning model for underwater image enhancement.

    Pipeline
    --------
    Input (3-ch, [-1,1])
      → QuantumEncoding            → 6-ch
      → enc1 Conv(6→64) + ReLU
      → enc2 Conv(64→128) + ReLU
      → Downsample → TransformerBlock → Upsample
      → ResBlock × n_res  (each with CBAM)
      → Decoder Conv(128→64→3) + Tanh
    Output (3-ch, [-1,1])

    BUG FIX: ConvTranspose2d output size must exactly match pre-downsample
    size. Using kernel=4, stride=2, padding=1 gives:
        out = (in - 1)*stride - 2*padding + kernel
            = (64-1)*2 - 2 + 4 = 128  ✓
    """
    def __init__(self, base: int = 64, n_res: int = 4):
        super().__init__()
        ch2 = base * 2   # 128

        self.q    = QuantumEncoding()

        self.enc1 = nn.Sequential(
            nn.Conv2d(6, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, ch2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )

        # Downsample → Transformer (on 64×64 tokens) → Upsample
        self.down = nn.Conv2d(ch2, ch2, kernel_size=3, stride=2, padding=1)
        self.tr   = TransformerBlock(ch2)
        self.up   = nn.ConvTranspose2d(ch2, ch2, kernel_size=4,
                                       stride=2, padding=1)

        self.res  = nn.Sequential(*[ResBlock(ch2) for _ in range(n_res)])

        self.dec  = nn.Sequential(
            nn.Conv2d(ch2, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.up(self.tr(self.down(x)))
        x = self.res(x)
        return self.dec(x)
