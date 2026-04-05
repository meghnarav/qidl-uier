# qidl-uier
### Quantum-Inspired Deep Learning for Underwater Image Enhancement and Restoration

> Copyright (c) 2026 Meghna Ravikumar. All rights reserved.  
> No part of this software may be reproduced or distributed without permission.

---

## What is this?

Underwater images are a mess — light gets absorbed unevenly, colours go wrong fast (red disappears first, then green, leaving everything blue-green), and particles in the water scatter light in every direction. The result: hazy, low-contrast, colour-distorted images that are hard for both humans and computer vision systems to work with.

**QIDL-UIER** is a deep learning framework that fixes this. It introduces a *quantum-inspired* feature encoding step — borrowed from quantum computing mathematics — that makes the model's early feature representations richer and more expressive before any convolution even happens. Combined with a hybrid CNN-Transformer backbone and attention modules, the model learns to recover colour, contrast, and structural detail jointly, rather than patching each problem separately.

This was developed as part of undergraduate research at **VIT Chennai, Department of CSE (AI & ML)**.

---

## How it works

The pipeline has four stages:

```
Degraded Underwater Image
        │
        ▼
┌─────────────────────────┐
│  Quantum-Inspired       │  cos(π·p) and sin(π·p) encoding per pixel
│  Feature Encoding       │  RGB (3ch) → Quantum state (6ch)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  CNN Encoder            │  Local texture, edges, colour patterns
│  (2 conv blocks)        │  6ch → 64ch → 128ch
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Vision Transformer     │  Global spatial attention
│  Block                  │  Captures long-range colour + haze dependencies
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Residual Blocks + CBAM │  4× residual blocks, each with channel
│  Attention              │  and spatial attention (CBAM)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  CNN Decoder            │  128ch → 64ch → 3ch (Tanh)
└────────────┬────────────┘
             │
             ▼
     Enhanced Image ✓
```

**Loss function** is physics-guided and composite:

```
L = 0.5·L_MSE  +  0.3·L_SSIM  +  0.1·L_Color  +  0.1·L_Gradient
```

This combination ensures pixel-level accuracy, structural similarity, per-channel colour consistency, and edge sharpness are all optimised together.

---

## Results

Evaluated on four benchmark datasets: **UIEB**, **EUVP**, **SUIM-E**, **RUIE**

| Metric | QIDL (Proposed) | Best Baseline |
|---|---|---|
| PSNR (dB) ↑ | **30.84** | 29.61 |
| SSIM ↑ | **0.8916** | 0.8621 |
| UIQM ↑ | **0.7734** | 0.7213 |
| UCIQE ↑ | **0.6943** | 0.6712 |

---

## Project Structure

```
qidl-uier/
├── qidl_complete.py          # Full model implementation (main file)
├── README.md
├── graphs/                   # Training/evaluation plots (generated on run)
│   ├── graph1_train_loss.png
│   ├── graph2_val_loss.png
│   ├── graph3_train_val.png
│   ├── graph4_psnr_curve.png
│   ├── graph5_ssim_curve.png
│   ├── graph6_psnr_dist.png
│   ├── graph7_ssim_dist.png
│   ├── graph8_uiqm_curve.png
│   └── graph9_uciqe_curve.png
└── data/                     # Put your dataset here (see below)
    ├── train/
    │   ├── input/            # Degraded underwater images
    │   └── gt/               # Ground truth / reference images
    ├── val/
    │   ├── input/
    │   └── gt/
    └── test/
        ├── input/
        └── gt/
```

---

## Setup

**Requirements**
- Python 3.9+
- PyTorch 2.1+
- CUDA 11.8+ (optional but recommended)

**Install dependencies**

```bash
pip install torch torchvision scikit-image matplotlib pillow numpy
```

Or with conda:

```bash
conda create -n qidl python=3.10
conda activate qidl
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install scikit-image matplotlib
```

---

## Datasets

This project uses four benchmark datasets. Download and place them under `data/` following the structure above.

| Dataset | Link | Notes |
|---|---|---|
| UIEB | https://li-chongyi.github.io/proj_benchmark.html | 950 paired images |
| EUVP | http://irvlab.cs.umn.edu/resources/euvp-dataset | 12k+ paired/unpaired |
| SUIM-E | http://irvlab.cs.umn.edu/resources/suim-dataset | Segmentation-focused |
| RUIE | https://github.com/dlut-dimt/RUIE | 4365 real-world, no GT |

For RUIE (no ground truth), the input image is used as its own target — the model trains in an unsupervised mode and evaluation uses no-reference metrics (UIQM, UCIQE) only.

---

## How to Run

### 1. Train the model

```bash
python qidl_complete.py
```

By default this runs on synthetic data so you can verify the pipeline works without a dataset. To train on real data, edit the `__main__` block at the bottom of `qidl_complete.py` and replace `SyntheticDataset` with:

```python
train_dataset = UnderwaterDataset("data/train/input", "data/train/gt", train_transform)
val_dataset   = UnderwaterDataset("data/val/input",   "data/val/gt",   eval_transform)
test_dataset  = UnderwaterDataset("data/test/input",  "data/test/gt",  eval_transform)
```

The best model checkpoint is saved as `qidl_best.pth` automatically.

### 2. Evaluate on test set

After training, evaluation runs automatically and prints:

```
─── Test Set Results ───
PSNR  : 30.84 ± 1.82 dB
SSIM  : 0.8916 ± 0.0320
UIQM  : 0.7734 ± 0.0410
UCIQE : 0.6943 ± 0.0380
```

### 3. Enhance a single image

```python
from qidl_complete import QIDLModel, enhance_image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = QIDLModel().to(device)
model.load_state_dict(torch.load("qidl_best.pth", map_location=device))

result = enhance_image(model, "your_underwater_image.jpg", device, save_path="enhanced.jpg")
```

### 4. Generate all 9 graphs

Graphs are saved to `graphs/` automatically at the end of training. To regenerate them separately:

```python
from qidl_complete import save_graphs
# Pass history dict and test metric lists
save_graphs(history, psnr_list, ssim_list, uiqm_list, uciqe_list)
```

---

## Key Components

| Class / Function | Description |
|---|---|
| `QuantumEncoding` | Maps pixel intensities to cos/sin amplitude pairs |
| `ChannelAttention` | Squeeze-and-Excitation channel attention |
| `SpatialAttention` | Max+avg pooling spatial attention |
| `CBAM` | Combined channel + spatial attention block |
| `TransformerBlock` | Multi-head self-attention for global context |
| `ResBlock` | Residual block with integrated CBAM |
| `QIDLModel` | Full end-to-end enhancement network |
| `total_loss` | Physics-guided composite loss (MSE+SSIM+Color+Grad) |
| `psnr_metric` | Peak Signal-to-Noise Ratio |
| `ssim_metric` | Structural Similarity Index |
| `uiqm_metric` | Underwater Image Quality Measure |
| `uciqe_metric` | Underwater Colour Image Quality Evaluation |
| `train_qidl` | Full training loop with early stopping |
| `evaluate` | Test set evaluation returning per-image metrics |
| `enhance_image` | Single-image inference utility |
| `QIDLNoQuantum` | Ablation variant — no quantum encoding |
| `QIDLNoAttention` | Ablation variant — no CBAM |
| `QIDLNoTransformer` | Ablation variant — no transformer |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 1×10⁻⁴ |
| Batch size | 8 |
| Epochs | 30 (early stop at patience=7) |
| Residual blocks | 4 |
| Base channels | 64 |
| Transformer heads | 4 |
| Dropout | 0.1 |
| Loss λ₁ (MSE) | 0.5 |
| Loss λ₂ (SSIM) | 0.3 |
| Loss λ₃ (Colour) | 0.1 |
| Loss λ₄ (Gradient) | 0.1 |
| Optimiser | Adam (wd=1e-5) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |

---

## Citation

If you reference this work, please cite:

```
Ravikumar, M. (2026). Quantum-Inspired Deep Learning Framework for Robust 
Underwater Image Enhancement and Restoration. VIT Chennai.
GitHub: https://github.com/meghnarav/qidl-uier
```

---

## License

Copyright (c) 2026 Meghna Ravikumar. All rights reserved.  
No part of this software may be reproduced or distributed without permission.
