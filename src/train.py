import argparse
import random
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import mlflow

# Assure la résolution des modules depuis la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset import create_dataset_from_config
from src.models.srresnet import create_model_from_config
from src.utils.metrics import calculate_psnr, calculate_ssim


def set_seed(seed=42):
    """Fixe la graine aléatoire pour assurer la reproductibilité des résultats."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Exécute une époque d'entraînement avec la fonction de perte L1."""
    model.train()
    running_loss = 0.0

    for lr, hr in dataloader:
        lr, hr = lr.to(device), hr.to(device)

        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(dataloader))


def evaluate(model, dataloader, device):
    """Évalue le modèle sur le jeu de validation et calcule le PSNR et le SSIM moyens."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)

            total_psnr += calculate_psnr(sr, hr)
            total_ssim += calculate_ssim(sr, hr)

    avg_psnr = total_psnr / max(1, num_batches)
    avg_ssim = total_ssim / max(1, num_batches)
    return avg_psnr, avg_ssim


def log_sample_comparison(model, dataloader, device, epoch, output_dir=Path("samples")):
    """Génère et sauvegarde une image comparative (Basse Résolution, Prédiction SR, Vérité Terrain)."""
    model.eval()
    output_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for lr, hr in dataloader:
            lr_img = lr[0:1].to(device)
            hr_img = hr[0:1].to(device)
            sr_img = torch.clamp(model(lr_img), 0.0, 1.0)

            # Redimensionne le LR pour l'affichage comparatif côte à côte
            lr_bicubic = torch.nn.functional.interpolate(lr_img, size=hr_img.shape[-2:], mode="bicubic")

            # Concatène horizontalement : [LR bicubique | SRResNet | HR Originale]
            comparison = torch.cat([lr_bicubic[0], sr_img[0], hr_img[0]], dim=-1)
            pil_img = TF.to_pil_image(comparison.cpu())

            save_path = output_dir / f"epoch_{epoch:03d}.png"
            pil_img.save(save_path)
            mlflow.log_artifact(str(save_path), artifact_path="visual_samples")
            break


def main():
    parser = argparse.ArgumentParser(description="Entraînement SRResNet avec suivi MLflow")
    parser.add_argument("--config", default="params.yaml", help="Chemin du fichier de configuration")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Initialisation
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur le périphérique : {device}")

    # Chargement des données
    train_dataset = create_dataset_from_config("train", args.config)
    val_dataset = create_dataset_from_config("val", args.config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"]
    )

    # Modèle, optimiseur et fonction de coût L1
    model = create_model_from_config(args.config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    criterion = nn.L1Loss()

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_psnr = -1.0

    # Lancement du suivi MLflow
    mlflow.set_experiment("SRResNet_Baseline")
    with mlflow.start_run(run_name="srresnet_div2k_x4"):
        # Journalisation des paramètres
        mlflow.log_params({
            "learning_rate": cfg["train"]["lr"],
            "batch_size": cfg["train"]["batch_size"],
            "epochs": cfg["train"]["epochs"],
            "scale": cfg["degrade"]["scale"],
            "num_blocks": cfg["model"]["num_blocks"],
            "num_features": cfg["model"]["num_features"],
            "seed": cfg["train"]["seed"]
        })

        epochs = cfg["train"]["epochs"]
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_psnr, val_ssim = evaluate(model, val_loader, device)

            # Enregistrement des métriques dans MLflow
            mlflow.log_metrics({
                "train_loss_l1": train_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim
            }, step=epoch)

            print(f"Époque [{epoch:02d}/{epochs:02d}] - Perte L1: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

            # Sauvegarde d'échantillons visuels toutes les 5 époques
            if epoch % 5 == 0 or epoch == epochs:
                log_sample_comparison(model, val_loader, device, epoch)

            # Sauvegarde du meilleur modèle selon le PSNR
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = checkpoint_dir / "srresnet_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "config": cfg
                }, best_path)
                mlflow.log_artifact(str(best_path), artifact_path="models")

    print(f"Entraînement terminé. Meilleur PSNR: {best_psnr:.2f} dB sauvegardé dans {checkpoint_dir / 'srresnet_best.pt'}")


if __name__ == "__main__":
    main()
