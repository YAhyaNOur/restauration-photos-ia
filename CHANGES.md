# Corrections du modèle SRGAN (Person 2)

Fichiers à copier par-dessus votre `src/` (extraire le ZIP à la racine du dépôt) :
`src/models/train_srgan.py`, `src/models/evaluate_srgan.py`, `src/models/discriminator.py`, `src/models/log_baseline_to_mlflow.py`.
`log_srgan_to_mlflow.py` et tous les fichiers de Person 1 sont inchangés. Les noms de fichiers, de checkpoints (`checkpoints/srgan_generator_best.pt`) et de sorties JSON sont conservés.

## Ce qui a été corrigé
1. **Checkpoint compatible avec la baseline / l'API** : `srgan_generator_best.pt` contient désormais le générateur seul sous `model_state_dict` (et `generator_state_dict` pour l'ancien format). `predict.py`, `evaluate.py` (`load_model`) et l'export ONNX le chargent tels quels. Vérifié.
2. **Meilleur modèle choisi sur la LPIPS de validation** (`srgan.select_metric`, défaut `val_lpips`) au lieu du PSNR.
3. **Comparaison équitable** : `evaluate_srgan.py` évalue bicubique, SRResNet ET SRGAN avec le même code (PSNR/SSIM par image, LPIPS, FID), donc LPIPS/FID existent aussi pour la baseline. Les anciens chiffres de `metrics/test_metrics.json` ne servent plus qu'en secours (signalé). Exemples avant/après dans `metrics/examples/`.
4. **FID standard** via `torchmetrics` (Inception FID) et `--crops-per-image 4` (défaut) pour le stabiliser. Plus de `sqrtm` maison ni de `pip install` à l'exécution (mettre `lpips` dans `requirements.txt`).
5. **MLflow plus léger** : seul le générateur (léger) est envoyé, UNE fois à la fin ; l'état complet (discriminateur + optimiseurs) reste local dans `checkpoints/srgan_last.pt`. `log_baseline_to_mlflow.py` ajoute `test_lpips` / `test_fid` de la baseline si `metrics/srresnet_test_metrics_full.json` existe.
6. **Stabilité de l'entraînement** : petite perte L1 (`pixel_loss_weight`, 0.01), échauffement L1 si pas de checkpoint baseline, un seul passage du générateur par lot, `drop_last`, VGG toujours en mode eval et sans gradient sur la cible, `--resume checkpoints/srgan_last.pt`.
7. Petits points : docstring du discriminateur (global, pas PatchGAN), commentaire VGG5,4 corrigé, métriques supplémentaires dans MLflow (`val_lpips`, `d_prob_real`, `d_prob_fake`, pertes détaillées), paramètre standard `learning_rate`, `weights_only=False` explicite.

## Utilisation
```
pip install -r requirements_model_fixes.txt
python -m src.models.train_srgan --config params.yaml            # (--smoke-test pour un test rapide)
python -m src.models.evaluate_srgan --config params.yaml          # tableau + metrics/comparison_report.json
python -m src.models.log_baseline_to_mlflow
python -m src.models.log_srgan_to_mlflow
```
Ordre : `evaluate_srgan.py` avant les deux scripts `log_*`. Premier lancement : téléchargement de VGG19, AlexNet (LPIPS) et Inception (FID), mis en cache ensuite.

## Testé / non testé
Testé sur CPU avec un mini-jeu de données synthétique (poids non entraînés, les valeurs n'ont donc aucun sens) : entraînement (échauffement, phase GAN, scheduler, reprise), chargement du checkpoint par `predict.py` et `evaluate.py`, évaluation complète avec FID, secours sans baseline, scripts `log_*` vers MLflow.
Non testé : un vrai entraînement sur DIV2K / GPU, et le téléchargement des poids ImageNet (bloqué dans mon environnement : j'ai vérifié le chemin de code avec ces poids désactivés). Résultats réels : à vérifier chez vous.
