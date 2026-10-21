#!/usr/bin/env python3
"""Deblur images with a released WaveFusion-Net checkpoint.

Examples (run from the repository root):
    python scripts/infer.py --input blurry.png --output results/
    python scripts/infer.py --input photos/ --output results/ --dataset gopro
    python scripts/infer.py --input blurry.png --weights my_checkpoint.pth

The three released checkpoints (GoPro, HIDE, RealBlur-J) are attached to the v1.0 release
and are downloaded to ~/.cache/wavefusion/ on first use. Each one was trained on its own
dataset, so pick the one closest to your images; ``realblurj`` is the default.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavefusion import WaveFusionNet  # noqa: E402

RELEASE = "https://github.com/praann07/wavefusion-net-nigercon2026/releases/download/v1.0/"
WEIGHTS = {
    "gopro": "wavefusion_gopro_best.pth",
    "hide": "wavefusion_hide_best.pth",
    "realblurj": "wavefusion_realblurj_best.pth",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_weights(dataset: str, path: str | None) -> Path:
    if path:
        return Path(path)
    cache = Path.home() / ".cache" / "wavefusion"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / WEIGHTS[dataset]
    if not target.exists():
        print(f"Downloading {WEIGHTS[dataset]} (about 110 MB) ...")
        torch.hub.download_url_to_file(RELEASE + WEIGHTS[dataset], str(target))
    return target


def load_model(weights: Path, device: torch.device) -> WaveFusionNet:
    model = WaveFusionNet()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def deblur(model: WaveFusionNet, image: Image.Image, device: torch.device) -> Image.Image:
    x = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255.0 + 0.5).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="an image file or a folder of images")
    parser.add_argument("--output", default="results", help="output folder (default: results)")
    parser.add_argument("--dataset", choices=sorted(WEIGHTS), default="realblurj",
                        help="which released checkpoint to use (default: realblurj)")
    parser.add_argument("--weights", help="path to your own checkpoint instead of a released one")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    source = Path(args.input)
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if source.is_dir() else [source]
    if not files:
        sys.exit(f"No images found in {source}")

    model = load_model(get_weights(args.dataset, args.weights), device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        result = deblur(model, Image.open(path), device)
        target = out_dir / f"{path.stem}_deblurred.png"
        result.save(target)
        print(f"{path.name} -> {target}")


if __name__ == "__main__":
    main()
