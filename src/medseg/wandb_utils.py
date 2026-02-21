from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import wandb


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)


def _gray_to_rgb(gray_01: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray_01, gray_01, gray_01], axis=-1)
    return (255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)


def _heatmap_rgb(x: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.clip(x, 0.0, 1.0))
    return (255.0 * rgba[..., :3]).astype(np.uint8)


def _overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.35):
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = (1.0 - alpha) * out[..., c][m] + alpha * color[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def wandb_log_mid_slices(
    prefix: str,
    image_zyx: np.ndarray,
    prob_mean_zyx: np.ndarray,
    unc_zyx: np.ndarray,
    gt_zyx: Optional[np.ndarray] = None,
    step: Optional[int] = None,
) -> None:
    """
    Logs:
    - input mid slice
    - prob overlay mid slice
    - uncertainty heatmap mid slice
    - optional GT overlay
    """
    z = int(image_zyx.shape[0] // 2)

    image_01 = _normalize_01(image_zyx[z])
    base_rgb = _gray_to_rgb(image_01)

    prob_01 = np.clip(prob_mean_zyx[z].astype(np.float32), 0.0, 1.0)
    unc_01 = _normalize_01(unc_zyx[z])

    prob_heat = _heatmap_rgb(prob_01, "magma")
    unc_heat = _heatmap_rgb(unc_01, "inferno")
    prob_overlay = _overlay_mask(base_rgb, prob_01 > 0.5, color=(255, 0, 0), alpha=0.30)

    log = {
        f"{prefix}/input_mid": wandb.Image(base_rgb),
        f"{prefix}/prob_overlay_mid": wandb.Image(prob_overlay),
        f"{prefix}/prob_heatmap_mid": wandb.Image(prob_heat),
        f"{prefix}/uncertainty_mid": wandb.Image(unc_heat),
    }

    if gt_zyx is not None:
        gt_overlay = _overlay_mask(base_rgb, gt_zyx[z] > 0, color=(0, 255, 0), alpha=0.30)
        log[f"{prefix}/gt_overlay_mid"] = wandb.Image(gt_overlay)

    wandb.log(log, step=step)


def wandb_log_figure(key: str, fig, step: Optional[int] = None) -> None:
    wandb.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def wandb_log_scalars(d: Dict, step: Optional[int] = None) -> None:
    wandb.log(d, step=step)
