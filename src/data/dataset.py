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
        # Position centrale pour une évaluation déterministe et reproductible
        x_lr = max(0, (w_lr - patch_size_lr) // 2)
        y_lr = max(0, (h_lr - patch_size_lr) // 2)

    x_hr = x_lr * scale
    y_hr = y_lr * scale

    lr_patch = lr_img.crop((x_lr, y_lr, x_lr + patch_size_lr, y_lr + patch_size_lr))
    hr_patch = hr_img.crop((x_hr, y_hr, x_hr + patch_size_hr, y_hr + patch_size_hr))
    return lr_patch, hr_patch


def augment_pair(img_a, img_b):
    """Applique les mêmes transformations géométriques aléatoires aux deux images."""
    if random.random() > 0.5:
        img_a = TF.hflip(img_a)
        img_b = TF.hflip(img_b)

    if random.random() > 0.5:
        img_a = TF.vflip(img_a)
        img_b = TF.vflip(img_b)

    if random.random() > 0.5:
        img_a = TF.rotate(img_a, 90)
        img_b = TF.rotate(img_b, 90)

    return img_a, img_b


class SRResNetDataset(Dataset):
    """Dataset PyTorch pour SRResNet (LR natif 64x64 -> HR vérité terrain 256x256 avec facteur x4)."""

    def __init__(self, lr_dir, hr_dir, split_file, patch_size_lr=64, scale=4, is_train=True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size_lr = patch_size_lr
        self.scale = scale
        self.is_train = is_train

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

        # Extraction des patchs synchronisés : LR (64x64) et HR (256x256)
        lr_patch, hr_patch = random_crop_pair(
            lr_img, hr_img, self.patch_size_lr, self.scale, self.is_train
        )

        if self.is_train:
            lr_patch, hr_patch = augment_pair(lr_patch, hr_patch)

        lr_tensor = TF.to_tensor(lr_patch)
        hr_tensor = TF.to_tensor(hr_patch)
        return lr_tensor, hr_tensor


class BicubicUpscaleDataset(Dataset):
    """Dataset PyTorch avec patchs 128x128 où l'entrée LR est pré-agrandie en bicubique à la taille HR.
    
    Utile pour les modèles classiques (ex. SRCNN, VDSR) où l'entrée et la sortie
    partagent la même résolution spatiale (128x128).
    """

    def __init__(self, lr_dir, hr_dir, split_file, patch_size=128, scale=4, is_train=True):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train

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

        patch_lr_size = self.patch_size // self.scale
        w_lr, h_lr = lr_img.size

        if self.is_train:
            x_lr = random.randint(0, max(0, w_lr - patch_lr_size))
            y_lr = random.randint(0, max(0, h_lr - patch_lr_size))
        else:
            x_lr = max(0, (w_lr - patch_lr_size) // 2)
            y_lr = max(0, (h_lr - patch_lr_size) // 2)

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale

        lr_crop = lr_img.crop((x_lr, y_lr, x_lr + patch_lr_size, y_lr + patch_lr_size))
        hr_crop = hr_img.crop((x_hr, y_hr, x_hr + self.patch_size, y_hr + self.patch_size))

        # Agrandissement bicubique du patch LR vers la taille du patch HR (128x128)
        lr_upscaled = lr_crop.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        if self.is_train:
            lr_upscaled, hr_crop = augment_pair(lr_upscaled, hr_crop)

        lr_tensor = TF.to_tensor(lr_upscaled)
        hr_tensor = TF.to_tensor(hr_crop)
        return lr_tensor, hr_tensor


# Alias pour rétro-compatibilité
DIV2KDataset = SRResNetDataset


def create_dataset_from_config(split="train", config_path="params.yaml", dataset_type="srresnet"):
    """Crée une instance de Dataset selon params.yaml.
    
    Attention : les images de test (test.txt) proviennent des 800 images du train DIV2K
    (train_HR / train_LR), tandis que val.txt utilise le dossier DIV2K_valid (valid_HR / valid_LR).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    is_train = (split == "train")

    # train et test se lisent dans les dossiers train (DIV2K_train_HR et train_LR)
    if split in ("train", "test"):
        hr_dir = cfg["data"]["raw_train_hr"]
        lr_dir = cfg["data"]["processed_train_lr"]
    else:  # val
        hr_dir = cfg["data"]["raw_valid_hr"]
        lr_dir = cfg["data"]["processed_valid_lr"]

    split_file = Path(cfg["data"]["splits_dir"]) / f"{split}.txt"

    if dataset_type == "bicubic":
        return BicubicUpscaleDataset(
            lr_dir=lr_dir,
            hr_dir=hr_dir,
            split_file=split_file,
            patch_size=128,
            scale=cfg["degrade"]["scale"],
            is_train=is_train
        )

    return SRResNetDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        split_file=split_file,
        patch_size_lr=cfg["data"].get("patch_size_lr", 64),
        scale=cfg["degrade"]["scale"],
        is_train=is_train
    )
