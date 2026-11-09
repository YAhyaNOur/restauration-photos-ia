import argparse
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Assure la résolution des modules depuis la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.srresnet import create_model_from_config


def load_model(checkpoint_path=None, config_path="params.yaml", device=None):
    """Initialise l'architecture SRResNet et charge les poids du checkpoint s'il existe."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model_from_config(config_path).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Poids chargés avec succès depuis : {checkpoint_path}")
    else:
        print("Note : Aucun checkpoint trouvé ou spécifié. Le modèle tourne avec ses poids initiaux (mode test).")

    model.eval()
    return model, device


def predict(image, model, device):
    """Restaure une image basse résolution et retourne une image PIL haute résolution (facteur x4).

    Args:
        image (PIL.Image): Image basse résolution d'entrée.
        model (torch.nn.Module): Modèle SRResNet en mode eval().
        device (torch.device): CPU ou GPU.

    Returns:
        PIL.Image: Image super-résolue x4.
    """
    img_rgb = image.convert("RGB")
    lr_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

    # Reconversion du tenseur (1, 3, H, W) en image PIL
    sr_image = TF.to_pil_image(sr_tensor.squeeze(0).cpu())
    return sr_image


def main():
    parser = argparse.ArgumentParser(description="Inférence de super-résolution x4 avec SRResNet")
    parser.add_argument("--input", required=True, help="Chemin de l'image d'entrée basse résolution")
    parser.add_argument("--output", required=True, help="Chemin où sauvegarder l'image restaurée")
    parser.add_argument("--checkpoint", default="checkpoints/srresnet_best.pt", help="Chemin du fichier de poids .pt")
    parser.add_argument("--config", default="params.yaml", help="Chemin du fichier params.yaml")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"L'image d'entrée n'existe pas : {input_path}")

    model, device = load_model(args.checkpoint, args.config)
    lr_image = Image.open(input_path)
    print(f"Taille de l'image d'entrée (LR) : {lr_image.size}")

    sr_image = predict(lr_image, model, device)
    print(f"Taille de l'image restaurée (HR x4) : {sr_image.size}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sr_image.save(output_path)
    print(f"Image restaurée sauvegardée dans : {output_path}")


if __name__ == "__main__":
    main()
