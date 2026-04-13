"""
Single-image inference.

Usage
-----
    python infer.py                              # enhances test.jpg → out.jpg
    python infer.py --input path/to/img.jpg --output enhanced.jpg
"""
import argparse
import torch
from PIL import Image
from torchvision import transforms

from config import Config
from models import QIDL


def enhance(input_path: str, output_path: str,
            checkpoint: str = Config.checkpoint) -> None:

    # BUG FIX: original used Config.device (string) without torch.device()
    # and also had no map_location → crashed on CPU-only machines.
    device = torch.device(Config.device)

    model = QIDL(base=Config.base_ch, n_res=Config.n_res).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device)   # map_location added
    )
    model.eval()

    t = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    img  = Image.open(input_path).convert("RGB")
    orig = img.size                               # (W, H)
    x    = t(img).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)

    # Convert [-1,1] → [0,255] uint8, restore original resolution
    y_np = ((y.squeeze().cpu() + 1.0) / 2.0).clamp(0, 1).numpy()
    result = Image.fromarray(
        (y_np.transpose(1, 2, 0) * 255).astype("uint8")
    ).resize(orig, Image.BICUBIC)

    result.save(output_path)
    print(f"[INFO] Enhanced image saved → {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default="test.jpg")
    p.add_argument("--output",     default="out.jpg")
    p.add_argument("--checkpoint", default=Config.checkpoint)
    args = p.parse_args()
    enhance(args.input, args.output, args.checkpoint)
