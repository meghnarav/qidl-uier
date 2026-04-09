# qidl-uier
## Quantum-Inspired Deep Learning for Underwater Image Enhancement and Restoration

> **Copyright (c) 2026 Meghna Ravikumar. All rights reserved.**  
> No part of this software may be reproduced or distributed without permission.

---

## Overview

Underwater images degrade severely due to wavelength-selective light absorption, forward/backward scattering, non-uniform illumination, and depth-dependent contrast loss — producing colour-distorted, hazy, low-contrast imagery. **QIDL-UIER** addresses this through three architectural innovations:

1. **Quantum-Inspired Pixel Encoding** — maps each pixel `p` to `[cos(π·p), sin(π·p)]` before any convolution, adding structured non-linear representations inspired by quantum superposition
2. **Hybrid CNN-Transformer + CBAM** — local texture (CNN) + global colour context (Vision Transformer) + selective spatial attention (CBAM)
3. **Physics-Guided Composite Loss** — MSE + SSIM + colour consistency + gradient-edge preservation

Developed as undergraduate research at VIT Chennai, School of CSE (AI & ML).

---

## Results

| Metric | QIDL (Proposed) | Best Baseline | Improvement |
|---|---|---|---|
| PSNR (dB) ↑ | **30.84** | 29.61 | +1.23 dB |
| SSIM ↑ | **0.8916** | 0.8585 | +0.031 |
| UIQM ↑ | **0.7734** | 0.7213 | +0.052 |
| UCIQE ↑ | **0.6943** | 0.6712 | +0.023 |

Evaluated on: UIEB · EUVP · SUIM-E · RUIE

---

## Architecture

```
Input (3-ch)
    → QuantumEncoding  [cos(πp) ‖ sin(πp)]   → 6-ch  (no learnable params)
    → CNN Encoder      [6→64→128]
    → TransformerBlock [global self-attention]
    → ResBlock×4 + CBAM [local refinement + selective attention]
    → CNN Decoder      [128→64→3 + Tanh]
Output (3-ch enhanced image)

Loss = 0.5·MSE + 0.3·SSIM + 0.1·Colour + 0.1·Gradient
```

---

## Setup

```bash
pip install torch torchvision scikit-image matplotlib pillow numpy
```

---

## Datasets

| Dataset | Link | Notes |
|---|---|---|
| UIEB | https://li-chongyi.github.io/proj_benchmark.html | 950 paired images |
| EUVP | http://irvlab.cs.umn.edu/resources/euvp-dataset | 12k+ paired/unpaired |
| SUIM-E | http://irvlab.cs.umn.edu/resources/suim-dataset | Segmentation-focused |
| RUIE | https://github.com/dlut-dimt/RUIE | 4365 real-world, no GT |

Place under `data/train/`, `data/val/`, `data/test/` with `input/` and `gt/` subfolders.

---

## How to Run

### 1. Demo — no dataset needed

```bash
python qidl_final.py
```
Runs on synthetic data, confirms full pipeline works, saves 9 graphs to `graphs/`.

### 2. Train on real data

```bash
python qidl_final.py --mode train \
    --train_input data/train/input --train_gt data/train/gt \
    --val_input   data/val/input   --val_gt   data/val/gt   \
    --test_input  data/test/input  --test_gt  data/test/gt  \
    --epochs 30 --batch_size 8
```

### 3. Enhance a single image

```bash
python qidl_final.py --mode infer \
    --checkpoint qidl_best.pth \
    --input_image underwater.jpg \
    --output_image enhanced.jpg
```

### 4. Python API

```python
import torch
from qidl_final import QIDLModel, enhance_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = QIDLModel().to(device)
model.load_state_dict(torch.load("qidl_best.pth", map_location=device))
result = enhance_image(model, "underwater.jpg", device, "enhanced.jpg")
```

---

## Project Structure

```
qidl-uier/
├── qidl_final.py      ← Complete implementation (single file)
├── README.md
├── requirements.txt
├── .gitignore
├── data/              ← Datasets (gitignored)
└── graphs/            ← 9 training/evaluation plots (commit these)
```

---

## Citation

```
Ravikumar, M. (2026). Quantum-Inspired Deep Learning Framework for Robust
Underwater Image Enhancement and Restoration. VIT Chennai.
https://github.com/meghnarav/qidl-uier
```

---

**License:** Copyright (c) 2026 Meghna Ravikumar. All rights reserved.
