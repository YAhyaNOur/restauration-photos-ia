import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(pred, target):
    """Calcule le PSNR (Peak Signal-to-Noise Ratio) entre deux tenseurs dans [0, 1].

    Args:
        pred (torch.Tensor): Image ou lot d'images prédites.
        target (torch.Tensor): Image ou lot d'images de vérité terrain (HR).

    Returns:
        float: Valeur moyenne du PSNR en décibels (dB).
    """
    pred_clamped = torch.clamp(pred, 0.0, 1.0)
    target_clamped = torch.clamp(target, 0.0, 1.0)

    mse = torch.mean((pred_clamped - target_clamped) ** 2).item()
    if mse == 0:
        return 100.0

    psnr = 10.0 * np.log10(1.0 / mse)
    return float(psnr)


def _gaussian_kernel(window_size=11, sigma=1.5):
    """Génère un noyau Gaussien 2D normalisé pour le calcul local du SSIM."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    return kernel_2d.unsqueeze(0).unsqueeze(0)


def calculate_ssim(pred, target, window_size=11, sigma=1.5):
    """Calcule le SSIM (Structural Similarity Index) moyen selon Wang et al. (2004).

    Args:
        pred (torch.Tensor): Image (C, H, W) ou lot (B, C, H, W) dans [0, 1].
        target (torch.Tensor): Image ou lot de référence dans [0, 1].

    Returns:
        float: Score SSIM moyen dans [-1, 1] (idéalement proche de 1).
    """
    pred_clamped = torch.clamp(pred, 0.0, 1.0)
    target_clamped = torch.clamp(target, 0.0, 1.0)

    if pred_clamped.ndim == 3:
        pred_clamped = pred_clamped.unsqueeze(0)
        target_clamped = target_clamped.unsqueeze(0)

    channels = pred_clamped.shape[1]
    kernel = _gaussian_kernel(window_size, sigma).repeat(channels, 1, 1, 1).to(pred.device)
    padding = window_size // 2

    # Moyennes locales
    mu_x = F.conv2d(pred_clamped, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target_clamped, kernel, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    # Variances et covariance locales
    sigma_x_sq = F.conv2d(pred_clamped * pred_clamped, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target_clamped * target_clamped, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred_clamped * target_clamped, kernel, padding=padding, groups=channels) - mu_xy

    # Constantes de stabilité Wang et al. (C1 = (0.01 * L)^2, C2 = (0.03 * L)^2 avec L=1.0)
    c1 = 0.0001
    c2 = 0.0009

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )

    return float(ssim_map.mean().item())
