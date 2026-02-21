from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    bin_acc: np.ndarray
    bin_conf: np.ndarray
    bin_count: np.ndarray


def _flatten_tensor(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy().reshape(-1)


def voxelwise_ece(
    prob: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 15,
    mask: Optional[torch.Tensor] = None,
) -> CalibrationResult:
    p = _flatten_tensor(prob)
    y = (_flatten_tensor(target) > 0.5).astype(np.int32)

    if mask is not None:
        m = _flatten_tensor(mask) > 0.5
        p = p[m]
        y = y[m]

    p = np.clip(p, 0.0, 1.0)
    pred = (p >= 0.5).astype(np.int32)
    conf = np.maximum(p, 1.0 - p)
    acc = (pred == y).astype(np.float32)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_acc = np.zeros(int(n_bins), dtype=np.float32)
    bin_conf = np.zeros(int(n_bins), dtype=np.float32)
    bin_count = np.zeros(int(n_bins), dtype=np.int64)

    for i in range(int(n_bins)):
        lo = bins[i]
        hi = bins[i + 1]
        if i == int(n_bins) - 1:
            sel = (conf >= lo) & (conf <= hi)
        else:
            sel = (conf >= lo) & (conf < hi)
        count = int(sel.sum())
        bin_count[i] = count
        if count > 0:
            bin_acc[i] = float(acc[sel].mean())
            bin_conf[i] = float(conf[sel].mean())

    total = max(1, int(bin_count.sum()))
    abs_gap = np.abs(bin_acc - bin_conf)
    ece = float(((bin_count / float(total)) * abs_gap).sum())
    mce = float(abs_gap[bin_count > 0].max()) if (bin_count > 0).any() else 0.0

    return CalibrationResult(
        ece=ece,
        mce=mce,
        bin_acc=bin_acc,
        bin_conf=bin_conf,
        bin_count=bin_count,
    )


def reliability_diagram_figure(cal: CalibrationResult, title: str = ""):
    """Return matplotlib Figure."""
    n_bins = len(cal.bin_acc)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = 1.0 / max(1, n_bins)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.bar(centers, cal.bin_acc, width=width * 0.9, alpha=0.7, color="#3f8efc", label="Accuracy")
    ax.plot(centers, cal.bin_conf, color="#f96b38", marker="o", linewidth=1.5, label="Confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def uncertainty_error_curve(
    unc: torch.Tensor,
    error: torch.Tensor,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return x,y arrays (unc_bin_centers, mean_error_per_bin)."""
    unc_np = _flatten_tensor(unc)
    err_np = _flatten_tensor(error)

    if unc_np.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    lo = float(np.min(unc_np))
    hi = float(np.max(unc_np))
    if hi <= lo:
        hi = lo + 1e-6

    bins = np.linspace(lo, hi, int(n_bins) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_err = np.full(int(n_bins), np.nan, dtype=np.float32)

    for i in range(int(n_bins)):
        if i == int(n_bins) - 1:
            sel = (unc_np >= bins[i]) & (unc_np <= bins[i + 1])
        else:
            sel = (unc_np >= bins[i]) & (unc_np < bins[i + 1])
        if np.any(sel):
            mean_err[i] = float(np.mean(err_np[sel]))

    return centers.astype(np.float32), mean_err
