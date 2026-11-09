import argparse
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Assure la résolution des modules depuis la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset import create_dataset_from_config
from src.models.srresnet import create_model_from_config
from src.utils.metrics import calculate_psnr, calculate_ssim


def load_model(checkpoint_path=None, config_path="params.yaml", device="cpu"):
    """Instancie le modèle SRResNet et charge les poids si un checkpoint est fourni."""
    model = create_model_from_config(config_path).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Checkpoint chargé depuis : {checkpoint_path}")
    else:
        print("Note : Aucun checkpoint fourni ou fichier introuvable. Évaluation avec les poids initiaux.")

    model.eval()
    return model


def evaluate_dataset(model, dataloader, device):
    """Calcule le PSNR et le SSIM moyens sur le DataLoader fourni."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            sr = torch.clamp(model(lr), 0.0, 1.0)

            total_psnr += calculate_psnr(sr, hr)
            total_ssim += calculate_ssim(sr, hr)

    avg_psnr = total_psnr / max(1, num_batches)
    avg_ssim = total_ssim / max(1, num_batches)
    return avg_psnr, avg_ssim


def export_to_onnx(model, onnx_path, device="cpu", input_size=(64, 64)):
    """Exporte le modèle PyTorch au format ONNX avec axes dynamiques pour le batch et la résolution."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    onnx_file = Path(onnx_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_file),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    )
    print(f"Modèle exporté avec succès en ONNX : {onnx_file}")


def main():
    parser = argparse.ArgumentParser(description="Évaluation de SRResNet et export ONNX")
    parser.add_argument("--config", default="params.yaml", help="Chemin du fichier params.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/srresnet_best.pt", help="Chemin du checkpoint .pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Jeu de données à évaluer")
    parser.add_argument("--export-onnx", default=None, help="Chemin de sortie pour l'export ONNX (ex: checkpoints/srresnet.onnx)")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille des lots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique : {device}")

    # Chargement du modèle
    model = load_model(args.checkpoint, args.config, device)

    # Chargement du dataset (test se lit dans train_LR)
    dataset = create_dataset_from_config(args.split, args.config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Évaluation sur le split '{args.split}' ({len(dataset)} paires)...")
    psnr, ssim = evaluate_dataset(model, dataloader, device)
    print(f"Résultats [{args.split}] - PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    # Export ONNX si demandé
    if args.export_onnx:
        export_to_onnx(model, args.export_onnx, device=device)


if __name__ == "__main__":
    main()

