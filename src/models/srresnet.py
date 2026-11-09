from pathlib import Path
import yaml
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Bloc résiduel classique : Conv -> BN -> PReLU -> Conv -> BN + Skip Connection."""

    def __init__(self, num_features=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """Bloc de sur-échantillonnage x2 via PixelShuffle : Conv(64 -> 256) -> PixelShuffle(2) -> PReLU."""

    def __init__(self, num_features=64, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class SRResNet(nn.Module):
    """Architecture SRResNet pour la super-résolution x4."""

    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=16, scale=4):
        super().__init__()

        # 1. Couche d'entrée
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # 2. Corps résiduel : B blocs résiduels consécutifs
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        # 3. Saut résiduel global (global skip connection)
        self.mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features)
        )

        # 4. Sur-échantillonnage (x4 = 2 blocs x2)
        num_upsample_blocks = 2 if scale == 4 else 1
        self.upsample = nn.Sequential(
            *[UpsampleBlock(num_features, upscale_factor=2) for _ in range(num_upsample_blocks)]
        )

        # 5. Couche de sortie ramenant vers le nombre de canaux souhaité (RGB)
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        initial_features = self.head(x)
        res_features = self.res_blocks(initial_features)
        mid_features = self.mid(res_features) + initial_features
        upscaled_features = self.upsample(mid_features)
        return self.tail(upscaled_features)


def create_model_from_config(config_path="params.yaml"):
    """Instancie le modèle SRResNet en chargeant les hyperparamètres depuis params.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    return SRResNet(
        in_channels=model_cfg.get("num_channels", 3),
        out_channels=model_cfg.get("num_channels", 3),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 16),
        scale=model_cfg.get("scale", 4)
    )
