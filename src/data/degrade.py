import argparse
import io
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def degrade(img, scale=4, blur=1.2, noise_std=6.0, jpeg_quality=60, seed=0):
    """Transforme une image HR en image LR dégradée."""
    rng = np.random.default_rng(seed)
    img = img.convert("RGB")
    w, h = img.size
    # dimensions divisibles par scale
    w, h = w - w % scale, h - h % scale  
    img = img.crop((0, 0, w, h))

    img = img.filter(ImageFilter.GaussianBlur(blur))
    img = img.resize((w // scale, h // scale), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0, noise_std, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="dossier des images HR")
    p.add_argument("--output", required=True, help="dossier des images LR")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--blur", type=float, default=1.2)
    p.add_argument("--noise", type=float, default=6.0)
    p.add_argument("--jpeg", type=int, default=60)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input).glob("*.png"))
    for f in files:
        # graine stable par image
        seed = zlib.crc32(f.name.encode())  
        lr = degrade(Image.open(f), args.scale, args.blur, args.noise, args.jpeg, seed)
        lr.save(out / f.name)
    print(f"{len(files)} images dégradées dans {out}")


if __name__ == "__main__":
    main()