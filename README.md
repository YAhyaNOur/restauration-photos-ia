# Restauration de Photos Anciennes (Super-Résolution x4)

Projet étudiant pour le module **Deep Learning Avancé & MLOps** (TEK-UP).

## 1. Objectif du projet
Restaurer des images basse résolution (anciennes / dégradées) vers une haute résolution avec un facteur d'agrandissement de **x4**, en utilisant le jeu de données **DIV2K**.

### Périmètre de cette partie (Data & Baseline)
Ce dépôt contient le pipeline de données et le modèle de référence (**SRResNet**) :
- Préparation et dégradation contrôlée des données (HR -> LR) via **DVC**.
- Chargement des données par patchs (LR 64x64 -> HR 256x256).
- Modèle baseline **SRResNet** (blocs résiduels + PixelShuffle x4).
- Suivi des métriques (PSNR, SSIM) et des entraînements avec **MLflow**.
- Inférence simple sur une image.

*(Les parties GAN, API REST, Frontend, Docker et CI/CD sont développées par les autres membres du groupe).*

## 2. Installation de l'environnement

```bash
# 1. Créer un environnement virtuel (recommandé)
python -m venv .venv

# 2. Activer l'environnement
# Sur Windows (PowerShell) :
.venv\Scripts\Activate.ps1
# Sur Linux / macOS :
# source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## 3. Structure du projet

```text
├── data/               # Données (ignorées par Git, suivies par DVC)
├── splits/             # Fichiers de répartition (train.txt, val.txt, test.txt)
├── src/
│   ├── data/           # Préparation, dégradation et Dataset PyTorch
│   ├── models/         # Architecture SRResNet
│   ├── utils/          # Calcul des métriques (PSNR, SSIM)
│   ├── train.py        # Entraînement et suivi MLflow
│   └── predict.py      # Inférence
├── dvc.yaml            # Pipeline DVC
├── params.yaml         # Hyperparamètres centralisés
└── requirements.txt    # Dépendances autorisées
```
