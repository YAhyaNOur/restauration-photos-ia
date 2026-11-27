"""Évaluation SRGAN vs SRResNet : PSNR, SSIM, LPIPS, FID (+ bicubique comme repère).

Les TROIS méthodes sont évaluées par le même code, sur les mêmes patchs du split test : la comparaison est
équitable (PSNR/SSIM moyennés image par image pour tous). Les anciens chiffres de metrics/test_metrics.json
(PSNR moyenné par lot) ne sont utilisés qu'en secours si le checkpoint baseline est introuvable.

Sorties (mêmes noms qu'avant, lus par log_srgan_to_mlflow.py) :
  metrics/srgan_test_metrics.json          scores du SRGAN
  metrics/srresnet_test_metrics_full.json  scores du SRResNet recalculés (avec LPIPS et FID)
  metrics/comparison_report.json           comparaison complète
  metrics/examples/example_XX.png          exemples avant/après

Le FID utilise torchmetrics (Inception FID standard, pip install torchmetrics torch-fidelity).
Avec seulement 100 images il est très bruité : --crops-per-image 4 (défaut) ou plus le stabilise.
Indiquez dans le rapport le protocole utilisé.
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset import create_dataset_from_config
from src.models.srresnet import create_model_from_config
from src.utils.metrics import calculate_psnr, calculate_ssim


class MultiCropEvalDataset(Dataset):
    """Plusieurs patchs déterministes par image (le 1er est le patch central, comme la baseline)."""

    def __init__(self, base, crops_per_image):
        self.base = base
        self.k = crops_per_image

    def __len__(self):
        return len(self.base.filenames) * self.k

    def __getitem__(self, idx):
        b = self.base
        name, k = b.filenames[idx // self.k], idx % self.k
        lr_img = Image.open(b.lr_dir / name).convert("RGB")
        hr_img = Image.open(b.hr_dir / name).convert("RGB")
        p, s = b.patch_size_lr, b.scale
        w, h = lr_img.size
        if k == 0:
            x, y = max(0, (w - p) // 2), max(0, (h - p) // 2)
        else:
            rng = np.random.default_rng(zlib.crc32(name.encode()) + k)
            x = int(rng.integers(0, max(1, w - p + 1)))
            y = int(rng.integers(0, max(1, h - p + 1)))
        lr = lr_img.crop((x, y, x + p, y + p))
        hr = hr_img.crop((x * s, y * s, x * s + p * s, y * s + p * s))
        return TF.to_tensor(lr), TF.to_tensor(hr)


def load_generator(checkpoint_path, config_path, device):
    """Charge un générateur (SRResNet ou SRGAN) : formats model_state_dict / generator_state_dict / brut."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    model = create_model_from_config(config_path).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("generator_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    print(f"Modèle chargé depuis : {checkpoint_path}")
    return model.eval()


def build_lpips(device, pretrained=True):
    import lpips  # pip install lpips

    fn = lpips.LPIPS(net="alex", pnet_rand=not pretrained, verbose=False).to(device).eval()
    for p in fn.parameters():
        p.requires_grad = False
    return fn


@torch.no_grad()
def evaluate_method(name, upscale_fn, loader, device, lpips_fn, fid_features):
    """PSNR/SSIM/LPIPS moyens par image + FID global pour une méthode d'agrandissement."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=fid_features, normalize=True).to(device)
    psnr = ssim = lp = 0.0
    n = 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = torch.clamp(upscale_fn(lr), 0.0, 1.0)
        for i in range(sr.size(0)):
            psnr += calculate_psnr(sr[i : i + 1], hr[i : i + 1])
            ssim += calculate_ssim(sr[i : i + 1], hr[i : i + 1])
        lp += lpips_fn(sr, hr, normalize=True).sum().item()
        fid.update(hr, real=True)
        fid.update(sr, real=False)
        n += sr.size(0)
    n = max(1, n)
    res = {"model": name, "psnr": psnr / n, "ssim": ssim / n, "lpips": lp / n, "fid": float(fid.compute().item())}
    print(
        f"[{name:9s}] PSNR {res['psnr']:.2f} dB | SSIM {res['ssim']:.4f} | "
        f"LPIPS {res['lpips']:.4f} | FID {res['fid']:.2f}"
    )
    return res


def _label(tensor, text):
    img = TF.to_pil_image(torch.clamp(tensor.cpu(), 0, 1))
    ImageDraw.Draw(img).text((6, 6), text, fill=(255, 255, 0))
    return img


@torch.no_grad()
def save_examples(methods, dataset, device, out_dir, n=6):
    """Images [Bicubique | SRResNet | SRGAN | HR] pour illustrer le compromis PSNR / netteté."""
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(dataset) // n)
    paths = []
    for j, idx in enumerate(list(range(0, len(dataset), step))[:n]):
        lr, hr = dataset[idx]
        lr_b, hr_b = lr.unsqueeze(0).to(device), hr.unsqueeze(0).to(device)
        cols = []
        for name, fn in methods.items():
            cols.append(_label(torch.clamp(fn(lr_b), 0, 1)[0], name))
        cols.append(_label(hr_b[0], "HR (verite)"))
        canvas = Image.new("RGB", (sum(c.width for c in cols), cols[0].height))
        x = 0
        for c in cols:
            canvas.paste(c, (x, 0))
            x += c.width
        path = out_dir / f"example_{j:02d}.png"
        canvas.save(path)
        paths.append(path)
    return paths


def _delta(a, b):
    return None if a is None or b is None else float(a - b)


def main():
    parser = argparse.ArgumentParser(description="Évaluation SRGAN et comparaison avec SRResNet baseline")
    parser.add_argument("--config", default="params.yaml", help="Chemin du fichier params.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/srgan_generator_best.pt", help="Checkpoint SRGAN")
    parser.add_argument("--baseline-checkpoint", default="checkpoints/srresnet_best.pt", help="Checkpoint SRResNet")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crops-per-image", type=int, default=4, help="Patchs par image (1 = patch central seul)")
    parser.add_argument("--fid-features", type=int, default=2048, choices=[64, 192, 768, 2048])
    parser.add_argument("--max-images", type=int, default=None, help="Limite d'images (tests rapides)")
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--examples-dir", default="metrics/examples")
    parser.add_argument(
        "--baseline-metrics", default="metrics/test_metrics.json", help="Secours si pas de checkpoint baseline"
    )
    parser.add_argument("--metrics-out", default="metrics/srgan_test_metrics.json")
    parser.add_argument("--baseline-out", default="metrics/srresnet_test_metrics_full.json")
    parser.add_argument("--comparison-out", default="metrics/comparison_report.json")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Test rapide sans poids ImageNet (pas de téléchargement)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique : {device}")

    base_ds = create_dataset_from_config(args.split, args.config)
    dataset = base_ds if args.crops_per_image == 1 else MultiCropEvalDataset(base_ds, args.crops_per_image)
    limit = 10 if args.smoke_test else args.max_images
    if limit:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    scale = base_ds.scale
    fid_features = 64 if args.smoke_test else args.fid_features
    print(f"Évaluation sur '{args.split}' : {len(dataset)} patchs ({args.crops_per_image} par image)")

    lpips_fn = build_lpips(device, pretrained=not args.smoke_test)

    methods = {"Bicubique": lambda x: F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)}
    baseline_available = Path(args.baseline_checkpoint).exists()
    if baseline_available:
        srresnet = load_generator(args.baseline_checkpoint, args.config, device)
        methods["SRResNet"] = lambda x, m=srresnet: m(x)
    else:
        print(
            f"ATTENTION : checkpoint baseline introuvable ({args.baseline_checkpoint}) "
            "-> repli sur les anciens chiffres."
        )
    srgan = load_generator(args.checkpoint, args.config, device)
    methods["SRGAN"] = lambda x, m=srgan: m(x)

    results = {}
    for name, fn in methods.items():
        results[name] = evaluate_method(name, fn, loader, device, lpips_fn, fid_features)

    # --- Baseline : recalculée (idéal) ou reprise de l'ancien fichier (secours) ---
    if baseline_available:
        baseline = results["SRResNet"]
        baseline_note = "SRResNet recalculé avec le même protocole que le SRGAN (PSNR/SSIM par image, LPIPS, FID)."
    else:
        old = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
        baseline = {
            "model": "SRResNet",
            "psnr": old["model"]["psnr"],
            "ssim": old["model"]["ssim"],
            "lpips": None,
            "fid": None,
        }
        baseline_note = "Anciens chiffres (PSNR/SSIM moyennés par lot) : NON comparables strictement, sans LPIPS/FID."
    srgan_res = results["SRGAN"]

    protocol = {
        "split": args.split,
        "crops_per_image": args.crops_per_image,
        "n_patches": len(dataset),
        "fid_features": fid_features,
    }

    def dump(path, obj):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return p

    dump(args.metrics_out, {"split": args.split, **srgan_res, "protocol": protocol})
    if baseline_available:
        dump(args.baseline_out, {"split": args.split, **baseline, "protocol": protocol})

    gain_perceptual = (
        baseline["lpips"] is not None
        and (srgan_res["lpips"] < baseline["lpips"] or srgan_res["fid"] < baseline["fid"])
    )
    if baseline["lpips"] is None:
        verdict = "Comparaison perceptuelle impossible sans le checkpoint de la baseline."
    elif srgan_res["psnr"] < baseline["psnr"] and gain_perceptual:
        verdict = (
            "Comportement attendu : le SRGAN perd en PSNR/SSIM (mesures pixel à pixel) mais gagne en LPIPS/FID "
            "(qualité perceptuelle) : images plus nettes. À montrer avec metrics/examples/."
        )
    elif gain_perceptual:
        verdict = "Le SRGAN améliore à la fois PSNR/SSIM et les métriques perceptuelles."
    else:
        verdict = (
            "Le SRGAN ne dépasse pas la baseline en perceptuel : vérifier le nombre d'époques GAN, "
            "adversarial_loss_weight (trop bas/haut) et l'initialisation depuis SRResNet."
        )

    comparison = {
        "protocol": protocol,
        "bicubic": results["Bicubique"],
        "baseline_srresnet": {**baseline, "note": baseline_note},
        "srgan": srgan_res,
        "comparison": {
            "psnr_delta": _delta(srgan_res["psnr"], baseline["psnr"]),
            "ssim_delta": _delta(srgan_res["ssim"], baseline["ssim"]),
            "lpips_delta": _delta(srgan_res["lpips"], baseline["lpips"]),
            "fid_delta": _delta(srgan_res["fid"], baseline["fid"]),
            "verdict": verdict,
            "note": (
                "PSNR/SSIM delta positif = SRGAN meilleur ; LPIPS et FID delta NÉGATIF = SRGAN meilleur "
                "(plus bas = meilleur)."
            ),
        },
    }
    dump(args.comparison_out, comparison)

    examples_methods = {k: v for k, v in methods.items()}
    save_examples(examples_methods, dataset, device, Path(args.examples_dir), args.n_examples)

    # --- Tableau comparatif ---
    def fmt(v, p):
        return f"{v:.{p}f}" if v is not None else "—"

    print("\n" + "=" * 78)
    print(f"{'Méthode':<12} {'PSNR (dB)':>10} {'SSIM':>8} {'LPIPS ↓':>9} {'FID ↓':>9}")
    print("=" * 78)
    print(f"{'Bicubique':<12} {fmt(results['Bicubique']['psnr'], 2):>10} {fmt(results['Bicubique']['ssim'], 4):>8} "
          f"{fmt(results['Bicubique']['lpips'], 4):>9} {fmt(results['Bicubique']['fid'], 2):>9}")
    print(f"{'SRResNet':<12} {fmt(baseline['psnr'], 2):>10} {fmt(baseline['ssim'], 4):>8} "
          f"{fmt(baseline['lpips'], 4):>9} {fmt(baseline['fid'], 2):>9}")
    print(f"{'SRGAN':<12} {fmt(srgan_res['psnr'], 2):>10} {fmt(srgan_res['ssim'], 4):>8} "
          f"{fmt(srgan_res['lpips'], 4):>9} {fmt(srgan_res['fid'], 2):>9}")
    print("=" * 78)
    print(verdict)
    print(f"Rapport : {args.comparison_out} | exemples : {args.examples_dir}/")


if __name__ == "__main__":
    main()
