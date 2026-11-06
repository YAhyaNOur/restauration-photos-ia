# Restauration de Photos Anciennes (Super-Résolution x4)

Projet étudiant pour le module **Deep Learning Avancé & MLOps** (TEK-UP).

---

## 1. Objectif du projet

Restaurer des images basse résolution (anciennes / dégradées) vers une haute résolution avec un facteur d'agrandissement de **x4**, en s'appuyant sur le jeu de données **DIV2K**.

### Périmètre de cette branche (Data & Baseline SRResNet)
- Pipeline de dégradation et versionnement des données avec **DVC** (stockage distant Google Drive).
- Dataset PyTorch avec patchs synchronisés (support natif SRResNet 64x64 $\rightarrow$ 256x256 et support bicubique 128x128).
- Architecture baseline **SRResNet** (blocs résiduels profonds + PixelShuffle x4).
- Suivi des entraînements et des métriques (PSNR, SSIM, perte L1) avec **MLflow**.
- Évaluation sur le split de test et export du modèle au format **ONNX**.
- Script d'inférence pour la restauration d'images individuelles.

---

## 2. Installation de l'environnement

```bash
# 1. Cloner le dépôt et se placer dans le projet
git clone <URL_DU_DEPOT>
cd restauration-photos-ia

# 2. Créer un environnement virtuel Python
python -m venv .venv

# 3. Activer l'environnement
# Sur Windows (PowerShell) :
.venv\Scripts\Activate.ps1
# Sur Linux / macOS :
# source .venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

---

## 3. Configuration DVC et récupération des données

Le stockage distant des données est hébergé sur Google Drive via DVC. Pour des raisons de sécurité, les identifiants OAuth Google Drive ne sont **jamais** versionnés dans Git et doivent être configurés en local.

### 3.1 Configuration locale des identifiants OAuth

Exécutez les commandes suivantes en remplaçant `<VOTRE_CLIENT_ID>` et `<VOTRE_CLIENT_SECRET>` par vos identifiants Google Cloud Console fournis dans le guide DVC :

```bash
# Configuration locale (enregistrée dans .dvc/config.local, ignorée par Git)
dvc remote modify --local storage gdrive_client_id <VOTRE_CLIENT_ID>
dvc remote modify --local storage gdrive_client_secret <VOTRE_CLIENT_SECRET>
```

> ⚠️ **Important :** Ne committez jamais votre `gdrive_client_secret`. Le flag `--local` garantit que ces paramètres sont stockés dans `.dvc/config.local`, qui figure déjà dans `.dvc/.gitignore`.

### 3.2 Téléchargement des données

Une fois les identifiants configurés, récupérez les données :

```bash
dvc pull
```

*Note : Lors du premier `dvc pull`, une page web s'ouvrira dans votre navigateur pour valider l'autorisation d'accès Google Drive.*

### 3.3 Pipeline DVC (Reproduction et synchronisation)

Pour exécuter ou reproduire le pipeline de dégradation des données :

```bash
# Exécuter les étapes du pipeline définies dans dvc.yaml
dvc repro

# Pousser les nouvelles données ou modifications vers le remote Google Drive
dvc push
```

---

## 4. Structure du projet

L'arborescence respecte la structure demandée :

```text
restauration-photos-ia/
├── data/
│   ├── raw/                    # Données brutes DIV2K HR (suivies par data/raw.dvc)
│   │   ├── DIV2K_train_HR/     # 800 images HR originales
│   │   └── DIV2K_valid_HR/     # 100 images HR validation
│   └── processed/              # Données générées (suivies par dvc.yaml)
│       ├── train_LR/           # 800 images dégradées (facteur x4)
│       └── valid_LR/           # 100 images dégradées (facteur x4)
├── splits/                     # Répartition stricte des données
│   ├── train.txt               # 700 images d'entraînement (issues de train_HR/train_LR)
│   ├── test.txt                # 100 images de test (issues de train_HR/train_LR)
│   └── val.txt                 # 100 images de validation (issues de valid_HR/valid_LR)
├── src/
│   ├── data/
│   │   ├── degrade.py          # Script de dégradation (flou, sous-échantillonnage, bruit, JPEG)
│   │   ├── split.py            # Génération des fichiers de split
│   │   └── dataset.py          # Datasets PyTorch (SRResNetDataset et BicubicUpscaleDataset)
│   ├── models/
│   │   ├── srresnet.py         # Architecture SRResNet x4
│   │   ├── train.py            # Entraînement et suivi MLflow
│   │   └── evaluate.py         # Évaluation des métriques et export ONNX
│   ├── utils/
│   │   └── metrics.py          # Calcul mathématique du PSNR et du SSIM
│   └── predict.py              # Inférence / restauration d'une image
├── dvc.yaml                    # Définition des étapes du pipeline DVC
├── params.yaml                 # Centralisation des hyperparamètres (dégradation, modèle, train)
├── requirements.txt            # Liste des dépendances du projet
└── README.md                   # Documentation du projet
```

---

## 5. Entraînement, Évaluation et Utilisation

### 5.1 Lancer l'entraînement

L'entraînement utilise les hyperparamètres définis dans `params.yaml` et enregistre les runs dans **MLflow** :

```bash
python src/models/train.py --config params.yaml
```

Les meilleurs poids sont automatiquement sauvegardés dans `checkpoints/srresnet_best.pt`.

### 5.2 Suivi des métriques avec MLflow

Pour visualiser les courbes de perte, l'évolution du PSNR/SSIM et les échantillons d'images :

```bash
mlflow ui
```

Accédez à l'interface via votre navigateur sur [http://localhost:5000](http://localhost:5000).

### 5.3 Évaluation sur le jeu de test

Pour calculer les scores finaux (PSNR et SSIM) sur les 100 images du split `test` :

```bash
python src/models/evaluate.py --checkpoint checkpoints/srresnet_best.pt --split test
```

### 5.4 Export ONNX

Pour exporter le modèle entraîné au format ONNX (utilisable pour le déploiement backend et l'inférence optimisée) :

```bash
python src/models/evaluate.py --checkpoint checkpoints/srresnet_best.pt --export-onnx checkpoints/srresnet.onnx
```

### 5.5 Inférence sur une image

Pour restaurer une image dégradée basse résolution :

```bash
python src/predict.py --input chemin/vers/image_lr.png --output chemin/vers/image_restauree.png --checkpoint checkpoints/srresnet_best.pt
```
