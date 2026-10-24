import random
from pathlib import Path
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def random_crop_pair(lr_img, hr_img, patch_size_lr=64, scale=4, is_train=True):
    """Extrait un patch LR et son patch HR correspondant exactement à la même position."""
    w_lr, h_lr = lr_img.size
    patch_size_hr = patch_size_lr * scale

    if is_train:
        # Position aléatoire dans l'image LR
        x_lr = random.randint(0, max(0, w_lr - patch_size_lr))
        y_lr = random.randint(0, max(0, h_lr - patch_size_lr))
    else:
        # Position centrale pour une validation reproductible
        x_lr = max(0, (w_lr - patch_size_lr) // 2)
        y_lr = max(0, (h_lr - patch_size_lr) // 2)

    x_hr = x_lr * scale
    y_hr = y_lr * scale

    # Découpage du patch LR (64x64) et HR (256x256)
    lr_patch = lr_img.crop((x_lr, y_lr, x_lr + patch_size_lr, y_lr + patch_size_lr))
    hr_patch = hr_img.crop((x_hr, y_hr, x_hr + patch_size_hr, y_hr + patch_size_hr))
    return lr_patch, hr_patch


def augment_pair(lr_patch, hr_patch):
    """Applique les mêmes rotations et retournements au patch LR et au patch HR."""
    # Retournement horizontal aléatoire
    if random.random() > 0.5:
        lr_patch = TF.hflip(lr_patch)
        hr_patch = TF.hflip(hr_patch)

    # Retournement vertical aléatoire
    if random.random() > 0.5:
        lr_patch = TF.vflip(lr_patch)
        hr_patch = TF.vflip(hr_patch)

    # Rotation aléatoire de 90 degrés
    if random.random() > 0.5:
        lr_patch = TF.rotate(lr_patch, 90)
        hr_patch = TF.rotate(hr_patch, 90)

    return lr_patch, hr_patch


class DIV2KDataset(Dataset):
    """Dataset PyTorch pour charger les paires d'images (LR dégradée, HR originale)."""

    def __init__(self, lr_dir, hr_dir, split_file, patch_size_lr=64, scale=4, is_train=True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size_lr = patch_size_lr
        self.scale = scale
        self.is_train = is_train

        # Lecture de la liste des noms d'images du split
        with open(split_file, "r", encoding="utf-8") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        lr_path = self.lr_dir / filename
        hr_path = self.hr_dir / filename

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        # 1. Extraction des patchs synchronisés
        lr_patch, hr_patch = random_crop_pair(
            lr_img, hr_img, self.patch_size_lr, self.scale, self.is_train
        )

        # 2. Augmentation de données (uniquement en train)
        if self.is_train:
            lr_patch, hr_patch = augment_pair(lr_patch, hr_patch)

        # 3. Conversion en tenseur PyTorch normalisé dans [0, 1]
        lr_tensor = TF.to_tensor(lr_patch)
        hr_tensor = TF.to_tensor(hr_patch)

        return lr_tensor, hr_tensor


def create_dataset_from_config(split="train", config_path="params.yaml"):
    """Crée une instance de DIV2KDataset à partir des paramètres du fichier params.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    is_train = (split == "train")
    hr_dir = cfg["data"]["raw_train_hr"] if is_train else cfg["data"]["raw_valid_hr"]
    lr_dir = cfg["data"]["processed_train_lr"] if is_train else cfg["data"]["processed_valid_lr"]
    split_file = Path(cfg["data"]["splits_dir"]) / f"{split}.txt"

    return DIV2KDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        split_file=split_file,
        patch_size_lr=cfg["data"]["patch_size_lr"],
        scale=cfg["degrade"]["scale"],
        is_train=is_train
    )
