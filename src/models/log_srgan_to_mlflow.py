"""Enregistre les résultats SRGAN dans l'expérience MLflow SR_Comparison
pour comparaison côte à côte avec le baseline SRResNet.

Exécuter après evaluate_srgan.py.
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
    srgan_metrics_path = "metrics/srgan_test_metrics.json"
    comparison_path = "metrics/comparison_report.json"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(srgan_metrics_path, "r", encoding="utf-8") as f:
        srgan = json.load(f)

    srgan_cfg = cfg["srgan"]

    mlflow.set_experiment("SR_Comparison")

    with mlflow.start_run(run_name="srgan_p2"):
        mlflow.log_params({
            "model_name": "SRGAN",
            "generator_lr": srgan_cfg["generator_lr"],
            "discriminator_lr": srgan_cfg["discriminator_lr"],
            "batch_size": srgan_cfg.get("batch_size", cfg["train"]["batch_size"]),
            "epochs": srgan_cfg.get("epochs", 50),
            "scale": cfg["degrade"]["scale"],
            "num_blocks": cfg["model"]["num_blocks"],
            "num_features": cfg["model"]["num_features"],
            "seed": cfg["train"]["seed"],
            "content_loss_weight": srgan_cfg.get("content_loss_weight", 1.0),
            "adversarial_loss_weight": srgan_cfg.get("adversarial_loss_weight", 0.001),
            "person": "Person 2",
        })

        mlflow.log_metrics({
            "test_psnr": srgan["psnr"],
            "test_ssim": srgan["ssim"],
            "test_lpips": srgan["lpips"],
            "test_fid": srgan["fid"],
        })

        # Log des artefacts
        checkpoint_path = Path("checkpoints/srgan_generator_best.pt")
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path), artifact_path="models")

        mlflow.log_artifact(srgan_metrics_path, artifact_path="metrics")

        if Path(comparison_path).exists():
            mlflow.log_artifact(comparison_path, artifact_path="metrics")

        print("SRGAN enregistré dans MLflow (experiment: SR_Comparison)")
        print(f"  Test PSNR: {srgan['psnr']:.4f} dB")
        print(f"  Test SSIM: {srgan['ssim']:.4f}")
        print(f"  Test LPIPS: {srgan['lpips']:.4f}")
        print(f"  Test FID: {srgan['fid']:.4f}")


if __name__ == "__main__":
    main()
