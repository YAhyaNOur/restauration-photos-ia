"""Enregistre les résultats du baseline SRResNet (Person 1) dans MLflow
pour permettre la comparaison côte à côte avec les expériences SRGAN.

Ce script ne retraine rien : il lit les métriques existantes et les log.
"""

import json
import sys
from pathlib import Path

import yaml
import mlflow

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    config_path = "params.yaml"
    metrics_path = "metrics/test_metrics.json"

    # Lecture de la configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Lecture des métriques baseline
    with open(metrics_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    mlflow.set_experiment("SR_Comparison")

    with mlflow.start_run(run_name="srresnet_baseline_p1"):
        mlflow.log_params({
            "model_name": "SRResNet",
            "learning_rate": cfg["train"]["lr"],
            "batch_size": cfg["train"]["batch_size"],
            "epochs": cfg["train"]["epochs"],
            "scale": cfg["degrade"]["scale"],
            "num_blocks": cfg["model"]["num_blocks"],
            "num_features": cfg["model"]["num_features"],
            "seed": cfg["train"]["seed"],
            "person": "Person 1",
        })

        mlflow.log_metrics({
            "test_psnr": baseline["model"]["psnr"],
            "test_ssim": baseline["model"]["ssim"],
            "bicubic_psnr": baseline["bicubic"]["psnr"],
            "bicubic_ssim": baseline["bicubic"]["ssim"],
        })

        # Scores recalculés avec le même protocole que le SRGAN (produits par evaluate_srgan.py) : LPIPS et FID
        full_path = Path("metrics/srresnet_test_metrics_full.json")
        if full_path.exists():
            full = json.loads(full_path.read_text(encoding="utf-8"))
            mlflow.log_metrics(
                {
                    "test_psnr_same_protocol": full["psnr"],
                    "test_ssim_same_protocol": full["ssim"],
                    "test_lpips": full["lpips"],
                    "test_fid": full["fid"],
                }
            )
            mlflow.log_artifact(str(full_path), artifact_path="metrics")

        # Log du checkpoint comme artefact s'il existe
        checkpoint_path = Path("checkpoints/srresnet_best.pt")
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path), artifact_path="models")

        # Log des métriques JSON
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

        print("Baseline SRResNet enregistré dans MLflow (experiment: SR_Comparison)")
        print(f"  Test PSNR: {baseline['model']['psnr']:.4f} dB")
        print(f"  Test SSIM: {baseline['model']['ssim']:.4f}")


if __name__ == "__main__":
    main()
