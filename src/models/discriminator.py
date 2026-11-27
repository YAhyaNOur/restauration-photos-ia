"""Discriminateur SRGAN – classifie les patchs HR réels vs générés.

Architecture selon Ledig et al. (2017) : 8 blocs Conv-BN-LeakyReLU
avec doublement progressif des filtres et réduction spatiale par stride 2.
La sortie est UN logit par image (discriminateur global, pas un PatchGAN) :
pas de sigmoïde, on utilise BCEWithLogitsLoss.
"""

import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """Bloc Conv(3×3) -> BN -> LeakyReLU(0.2)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """Discriminateur global du SRGAN : image HR (ou SR) -> logit « réelle ».

    Les noms de couches (features / classifier) sont inchangés : les anciens checkpoints restent chargeables.
    """

    def __init__(self, in_channels=3, base_features=64):
        super().__init__()

        self.features = nn.Sequential(
            # Première couche sans BN
            nn.Conv2d(in_channels, base_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Blocs avec augmentation progressive des filtres
            DiscriminatorBlock(base_features, base_features, stride=2),
            DiscriminatorBlock(base_features, base_features * 2, stride=1),
            DiscriminatorBlock(base_features * 2, base_features * 2, stride=2),
            DiscriminatorBlock(base_features * 2, base_features * 4, stride=1),
            DiscriminatorBlock(base_features * 4, base_features * 4, stride=2),
            DiscriminatorBlock(base_features * 4, base_features * 8, stride=1),
            DiscriminatorBlock(base_features * 8, base_features * 8, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_features * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
