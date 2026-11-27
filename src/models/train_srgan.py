"""Entraînement SRGAN – GAN pour la super-résolution x4.

Person 2 : construit sur le travail de Person 1 (SRResNet baseline).

Pipeline :
  1. Le générateur est initialisé depuis le checkpoint SRResNet (srgan.pretrained_generator). S'il est
     introuvable, `warmup_epochs` époques d'échauffement en L1 sont faites avant la phase GAN.
  2. Phase GAN : perte G = pixel_loss_weight·L1 + content_loss_weight·VGG + adversarial_loss_weight·GAN.
  3. Le « meilleur » modèle est choisi sur la LPIPS de validation (select_metric), pas sur le PSNR :
     un GAN perd en PSNR alors que ses images sont plus nettes ; choisir le PSNR reviendrait à garder
     l'époque la plus proche de la baseline.

Le checkpoint `checkpoints/srgan_generator_best.pt` contient UNIQUEMENT le générateur, sous les clés
`model_state_dict` (format de la baseline : compatible predict.py / evaluate.py / API / export ONNX)
et `generator_state_dict` (ancien format). `checkpoints/srgan_last.pt` contient tout l'état (reprise).

Lancement :
    python -m src.models.train_srgan --config params.yaml
    python -m src.models.train_srgan --config params.yaml --resume checkpoints/srgan_last.pt
    python -m src.models.train_srgan --config params.yaml --smoke-test      # test rapide (2 lots)
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import yaml
import mlflow
from torch.utils.data import ConcatDataset, DataLoader

# Résolution des modules depuis la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset import create_dataset_from_config
from src.models.srresnet import create_model_from_config
from src.models.discriminator import Discriminator
from src.utils.metrics import calculate_psnr, calculate_ssim


def set_seed(seed=42):
    """Fixe la graine aléatoire pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VGGFeatureExtractor(nn.Module):
    """Extracteur de features VGG19 pour la perte perceptuelle (VGG5,4)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        vgg = models.vgg19(weights=weights)
        # features[:36] = jusqu'à conv5_4 + ReLU (VGG5,4 « après activation », comme dans l'article SRGAN)
        self.features = nn.Sequential(*list(vgg.features[:36]))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)


class PerceptualLoss(nn.Module):
    """Perte perceptuelle (MSE) entre les features VGG19 de l'image SR et de l'image HR."""

    def __init__(self, device, pretrained=True):
        super().__init__()
        self.vgg = VGGFeatureExtractor(pretrained).to(device).eval()
        self.criterion = nn.MSELoss()
        # Normalisation ImageNet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        # Le VGG reste en mode eval quoi qu'il arrive (figé)
        super().train(mode)
        self.vgg.eval()
        return self

    def forward(self, sr, hr):
        sr_norm = (sr - self.mean.to(sr.device)) / self.std.to(sr.device)
        hr_norm = (hr - self.mean.to(hr.device)) / self.std.to(hr.device)
        sr_features = self.vgg(sr_norm)
        with torch.no_grad():  # la cible n'a pas besoin de gradient (économise de la mémoire)
            hr_features = self.vgg(hr_norm)
        return self.criterion(sr_features, hr_features)


class LPIPSScorer:
    """LPIPS (AlexNet) pour la sélection du meilleur modèle. Plus bas = meilleur."""

    def __init__(self, device, pretrained=True):
        import lpips  # dépendance à mettre dans requirements.txt : pip install lpips

        self.device = device
        self.fn = lpips.LPIPS(net="alex", pnet_rand=not pretrained, verbose=False).to(device).eval()
        for p in self.fn.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, sr, hr):
        # normalize=True : entrées dans [0, 1]
        return float(self.fn(sr.to(self.device), hr.to(self.device), normalize=True).mean().item())


def train_warmup_epoch(generator, dataloader, optimizer_g, device, max_batches=None):
    """Une époque d'échauffement : générateur seul, perte L1 (si pas de checkpoint baseline)."""
    generator.train()
    criterion = nn.L1Loss()
    total, n = 0.0, 0
    for i, (lr, hr) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        lr, hr = lr.to(device), hr.to(device)
        optimizer_g.zero_grad()
        loss = criterion(generator(lr), hr)
        loss.backward()
        optimizer_g.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


def train_one_epoch(
    generator,
    discriminator,
    dataloader,
    perceptual_loss,
    optimizer_g,
    optimizer_d,
    device,
    content_weight,
    adversarial_weight,
    pixel_weight=0.0,
    max_batches=None,
):
    """Exécute une époque d'entraînement SRGAN. Retourne les pertes moyennes."""
    generator.train()
    discriminator.train()

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    sums = {"g": 0.0, "d": 0.0, "content": 0.0, "adv": 0.0, "pixel": 0.0, "d_real": 0.0, "d_fake": 0.0}
    num_batches = 0

    for i, (lr, hr) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        lr, hr = lr.to(device), hr.to(device)

        # Un seul passage du générateur par lot : réutilisé pour D (detach) puis pour G
        sr = generator(lr)

        # --- Entraînement du Discriminateur ---
        optimizer_d.zero_grad()
        real_pred = discriminator(hr)
        fake_pred = discriminator(sr.detach())
        d_loss_real = bce_loss(real_pred, torch.ones_like(real_pred))
        d_loss_fake = bce_loss(fake_pred, torch.zeros_like(fake_pred))
        d_loss = (d_loss_real + d_loss_fake) / 2.0
        d_loss.backward()
        optimizer_d.step()

        # --- Entraînement du Générateur ---
        optimizer_g.zero_grad()
        gen_pred = discriminator(sr)
        content_loss = perceptual_loss(sr, hr)
        adv_loss = bce_loss(gen_pred, torch.ones_like(gen_pred))
        pixel_loss = l1_loss(sr, hr)
        g_loss = content_weight * content_loss + adversarial_weight * adv_loss + pixel_weight * pixel_loss
        g_loss.backward()
        optimizer_g.step()

        sums["g"] += g_loss.item()
        sums["d"] += d_loss.item()
        sums["content"] += content_loss.item()
        sums["adv"] += adv_loss.item()
        sums["pixel"] += pixel_loss.item()
        sums["d_real"] += torch.sigmoid(real_pred).mean().item()
        sums["d_fake"] += torch.sigmoid(fake_pred).mean().item()
        num_batches += 1

    return {k: v / max(1, num_batches) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, dataloader, device, lpips_scorer=None, max_batches=None):
    """Évalue le générateur sur la validation : PSNR, SSIM et LPIPS moyens."""
    model.eval()
    total_psnr = total_ssim = total_lpips = 0.0
    n = 0
    for i, (lr, hr) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        lr, hr = lr.to(device), hr.to(device)
        sr = torch.clamp(model(lr), 0.0, 1.0)
        total_psnr += calculate_psnr(sr, hr)
        total_ssim += calculate_ssim(sr, hr)
        if lpips_scorer is not None:
            total_lpips += lpips_scorer(sr, hr)
        n += 1
    n = max(1, n)
    return total_psnr / n, total_ssim / n, total_lpips / n


@torch.no_grad()
def log_sample_comparison(model, dataloader, device, epoch, output_dir=Path("samples_srgan")):
    """Génère et sauvegarde une image comparative [LR bicubique | SRGAN | HR] pour le suivi visuel."""
    model.eval()
    output_dir.mkdir(exist_ok=True)

    for lr, hr in dataloader:
        lr_img = lr[0:1].to(device)
        hr_img = hr[0:1].to(device)
        sr_img = torch.clamp(model(lr_img), 0.0, 1.0)
        lr_bicubic = torch.clamp(
            torch.nn.functional.interpolate(lr_img, size=hr_img.shape[-2:], mode="bicubic"), 0.0, 1.0
        )
        comparison = torch.cat([lr_bicubic[0], sr_img[0], hr_img[0]], dim=-1)
        pil_img = TF.to_pil_image(comparison.cpu())

        save_path = output_dir / f"srgan_epoch_{epoch:03d}.png"
        pil_img.save(save_path)
        mlflow.log_artifact(str(save_path), artifact_path="visual_samples")
        break


def load_generator_weights(generator, path, device):
    """Charge un checkpoint de générateur (formats model_state_dict / generator_state_dict / brut)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("generator_state_dict", checkpoint))
    generator.load_state_dict(state_dict)


def is_better(metric, new, best):
    """val_lpips : plus bas = meilleur ; val_psnr / val_ssim : plus haut = meilleur."""
    if best is None:
        return True
    return new < best if metric == "val_lpips" else new > best


def main():
    parser = argparse.ArgumentParser(description="Entraînement SRGAN avec suivi MLflow")
    parser.add_argument("--config", default="params.yaml", help="Chemin du fichier de configuration")
    parser.add_argument("--resume", default=None, help="Reprise depuis checkpoints/srgan_last.pt")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Test rapide : 2 lots, 1 époque, sans poids ImageNet (pas de téléchargement)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Configuration SRGAN depuis params.yaml
    srgan_cfg = cfg["srgan"]
    train_cfg = cfg["train"]

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement SRGAN sur : {device}")

    epochs = srgan_cfg.get("epochs", 50)
    warmup_epochs = int(srgan_cfg.get("warmup_epochs", 5))
    select_metric = srgan_cfg.get("select_metric", "val_lpips")
    if select_metric not in ("val_lpips", "val_psnr", "val_ssim"):
        raise ValueError("srgan.select_metric doit valoir val_lpips, val_psnr ou val_ssim")
    pretrained_backbones = True
    max_batches = None
    if args.smoke_test:
        epochs, warmup_epochs, pretrained_backbones, max_batches = 1, 1, False, 2

    # --- Chargement des données ---
    train_dataset = create_dataset_from_config("train", args.config)
    train_dataset = ConcatDataset([train_dataset] * train_cfg.get("repeat", 1))
    val_dataset = create_dataset_from_config("val", args.config)

    batch_size = srgan_cfg.get("batch_size", train_cfg["batch_size"])
    num_workers = train_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,  # évite un dernier lot minuscule (BatchNorm du discriminateur)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Modèles ---
    generator = create_model_from_config(args.config).to(device)
    discriminator = Discriminator(in_channels=cfg["model"].get("num_channels", 3)).to(device)

    pretrained_path = srgan_cfg.get("pretrained_generator", "checkpoints/srresnet_best.pt")
    needs_warmup = False
    if args.resume:
        pass  # l'état complet est restauré plus bas
    elif not args.smoke_test and Path(pretrained_path).exists():
        load_generator_weights(generator, pretrained_path, device)
        print(f"Générateur initialisé depuis le checkpoint SRResNet : {pretrained_path}")
    else:
        needs_warmup = True
        print(
            f"ATTENTION : checkpoint SRResNet introuvable ({pretrained_path}). "
            f"{warmup_epochs} époque(s) d'échauffement L1 avant la phase GAN."
        )

    # --- Optimiseurs avec LR configurables ---
    gen_lr = float(srgan_cfg["generator_lr"])
    disc_lr = float(srgan_cfg["discriminator_lr"])
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.9, 0.999))

    # --- Schedulers (optionnels) ---
    scheduler_cfg = srgan_cfg.get("scheduler", {})
    scheduler_enabled = scheduler_cfg.get("enabled", False)
    scheduler_g = scheduler_d = None
    if scheduler_enabled:
        step_size = scheduler_cfg.get("step_size", 25)
        gamma = scheduler_cfg.get("gamma", 0.5)
        scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=step_size, gamma=gamma)
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=step_size, gamma=gamma)
        print(f"Scheduler activé : StepLR(step_size={step_size}, gamma={gamma})")

    # --- Pertes ---
    perceptual_loss = PerceptualLoss(device, pretrained=pretrained_backbones)
    lpips_scorer = LPIPSScorer(device, pretrained=pretrained_backbones)
    content_weight = float(srgan_cfg.get("content_loss_weight", 1.0))
    adversarial_weight = float(srgan_cfg.get("adversarial_loss_weight", 0.001))
    pixel_weight = float(srgan_cfg.get("pixel_loss_weight", 0.01))

    # --- Répertoires de sortie ---
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_path = checkpoint_dir / "srgan_generator_best.pt"
    last_path = checkpoint_dir / "srgan_last.pt"
    best_value = None
    start_epoch = 1

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(ck["generator_state_dict"])
        discriminator.load_state_dict(ck["discriminator_state_dict"])
        optimizer_g.load_state_dict(ck["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(ck["optimizer_d_state_dict"])
        if scheduler_g is not None and ck.get("scheduler_g_state_dict"):
            scheduler_g.load_state_dict(ck["scheduler_g_state_dict"])
            scheduler_d.load_state_dict(ck["scheduler_d_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_value = ck.get("best_value")
        print(f"Reprise depuis {args.resume} (époque {ck['epoch']}, meilleur {select_metric}={best_value})")

    # --- MLflow ---
    mlflow.set_experiment("SRGAN_SuperResolution")
    start_time = time.time()

    with mlflow.start_run(run_name="srgan_div2k_x4"):
        mlflow.set_tags({"model_type": "srgan", "task": "super-resolution-x4", "dataset": "DIV2K"})
        mlflow.log_params(
            {
                "model_name": "SRGAN",
                "learning_rate": gen_lr,  # clé standard (identique à la baseline)
                "generator_lr": gen_lr,
                "discriminator_lr": disc_lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "scale": cfg["degrade"]["scale"],
                "num_blocks": cfg["model"]["num_blocks"],
                "num_features": cfg["model"]["num_features"],
                "seed": train_cfg["seed"],
                "content_loss_weight": content_weight,
                "adversarial_loss_weight": adversarial_weight,
                "pixel_loss_weight": pixel_weight,
                "pretrained_generator": pretrained_path,
                "warmup_epochs": warmup_epochs if needs_warmup else 0,
                "select_metric": select_metric,
                "scheduler_enabled": scheduler_enabled,
            }
        )
        if scheduler_enabled:
            mlflow.log_params(
                {
                    "scheduler_step_size": scheduler_cfg.get("step_size", 25),
                    "scheduler_gamma": scheduler_cfg.get("gamma", 0.5),
                }
            )
        if Path(args.config).exists():
            mlflow.log_artifact(args.config, artifact_path="config")

        # --- Échauffement L1 (seulement si pas de checkpoint baseline) ---
        if needs_warmup:
            for w in range(1, warmup_epochs + 1):
                w_loss = train_warmup_epoch(generator, train_loader, optimizer_g, device, max_batches)
                mlflow.log_metric("warmup_loss_l1", w_loss, step=w)
                print(f"Échauffement [{w}/{warmup_epochs}] - Perte L1: {w_loss:.4f}")

        # --- Phase GAN ---
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            losses = train_one_epoch(
                generator,
                discriminator,
                train_loader,
                perceptual_loss,
                optimizer_g,
                optimizer_d,
                device,
                content_weight,
                adversarial_weight,
                pixel_weight,
                max_batches,
            )
            val_psnr, val_ssim, val_lpips = evaluate(generator, val_loader, device, lpips_scorer, max_batches)

            current_gen_lr = optimizer_g.param_groups[0]["lr"]
            current_disc_lr = optimizer_d.param_groups[0]["lr"]

            mlflow.log_metrics(
                {
                    "generator_loss": losses["g"],
                    "discriminator_loss": losses["d"],
                    "content_loss": losses["content"],
                    "adversarial_loss": losses["adv"],
                    "pixel_loss": losses["pixel"],
                    "d_prob_real": losses["d_real"],
                    "d_prob_fake": losses["d_fake"],
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "val_lpips": val_lpips,
                    "generator_lr": current_gen_lr,
                    "discriminator_lr": current_disc_lr,
                    "epoch_time_seconds": time.time() - t0,
                },
                step=epoch,
            )

            print(
                f"Époque [{epoch:02d}/{epochs:02d}] - G_Loss: {losses['g']:.4f} | D_Loss: {losses['d']:.4f} | "
                f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f} | "
                f"D(réel)={losses['d_real']:.2f} D(faux)={losses['d_fake']:.2f}"
            )

            if scheduler_g:
                scheduler_g.step()
            if scheduler_d:
                scheduler_d.step()

            if epoch % 5 == 0 or epoch == epochs:
                log_sample_comparison(generator, val_loader, device, epoch)

            # Sélection du meilleur modèle (LPIPS par défaut)
            current = {"val_lpips": val_lpips, "val_psnr": val_psnr, "val_ssim": val_ssim}[select_metric]
            if is_better(select_metric, current, best_value):
                best_value = current
                gen_state = generator.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        # Générateur seul, sous les DEUX clés (même tenseurs, stockés une seule fois)
                        "model_state_dict": gen_state,
                        "generator_state_dict": gen_state,
                        "model_type": "srgan",
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "val_lpips": val_lpips,
                        "config": cfg,
                    },
                    best_path,
                )
                mlflow.log_metric(f"best_{select_metric}", current, step=epoch)

            # État complet pour reprendre un entraînement interrompu
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scheduler_g_state_dict": scheduler_g.state_dict() if scheduler_g else None,
                    "scheduler_d_state_dict": scheduler_d.state_dict() if scheduler_d else None,
                    "best_value": best_value,
                },
                last_path,
            )

        # Le meilleur générateur (léger) est envoyé UNE seule fois dans MLflow
        if best_path.exists():
            mlflow.log_artifact(str(best_path), artifact_path="models")

        training_duration = time.time() - start_time
        mlflow.log_metric("training_duration_seconds", training_duration)
        print(
            f"\nEntraînement SRGAN terminé en {training_duration:.0f}s. "
            f"Meilleur {select_metric}: {best_value} -> {best_path}"
        )


if __name__ == "__main__":
    main()
