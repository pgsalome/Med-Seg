from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn


@dataclass
class UncertaintyResult:
    prob_mean: torch.Tensor
    prob_samples: Optional[torch.Tensor]
    var: torch.Tensor
    entropy: torch.Tensor
    mutual_info: Optional[torch.Tensor]


def _autocast_enabled(x: torch.Tensor, amp: bool) -> bool:
    return bool(amp and x.is_cuda)


@contextmanager
def _autocast(x: torch.Tensor, amp: bool):
    enabled = _autocast_enabled(x, amp)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=enabled):
            yield
        return
    with torch.cuda.amp.autocast(enabled=enabled):
        yield


def _to_result_shape_prob(prob_bczyx: torch.Tensor) -> torch.Tensor:
    # (B,1,Z,Y,X) -> (1,Z,Y,X) for B=1
    if prob_bczyx.dim() == 5 and prob_bczyx.shape[0] == 1:
        return prob_bczyx[0]
    return prob_bczyx


def _to_result_shape_samples(samples_nbczyx: torch.Tensor) -> torch.Tensor:
    # (N,B,1,Z,Y,X) -> (N,1,Z,Y,X) for B=1
    if samples_nbczyx.dim() == 6 and samples_nbczyx.shape[1] == 1:
        return samples_nbczyx[:, 0]
    return samples_nbczyx


def predict_single(model: nn.Module, x: torch.Tensor, amp: bool = True) -> torch.Tensor:
    """Return sigmoid(logits)."""
    model.eval()
    with torch.no_grad():
        with _autocast(x, amp):
            logits = model(x)
        prob = torch.sigmoid(logits)
    return prob


def compute_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Binary entropy on probability map."""
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def compute_mutual_information(prob_samples: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """MI = H(mean) - mean(H(sample))."""
    if prob_samples.dim() == 6 and prob_samples.shape[1] == 1:
        prob_samples = prob_samples[:, 0]  # (N,1,Z,Y,X)
    p_mean = prob_samples.mean(dim=0)
    h_mean = compute_entropy(p_mean, eps=eps)
    h_each = compute_entropy(prob_samples, eps=eps)
    return h_mean - h_each.mean(dim=0)


def enable_dropout_inference(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC dropout."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def _aggregate_samples(samples_nbczyx: torch.Tensor, return_samples: bool) -> UncertaintyResult:
    prob_mean = samples_nbczyx.mean(dim=0)
    var = samples_nbczyx.var(dim=0, unbiased=False)
    entropy = compute_entropy(prob_mean)
    mutual_info = compute_mutual_information(samples_nbczyx)

    return UncertaintyResult(
        prob_mean=_to_result_shape_prob(prob_mean),
        prob_samples=_to_result_shape_samples(samples_nbczyx) if return_samples else None,
        var=_to_result_shape_prob(var),
        entropy=_to_result_shape_prob(entropy),
        mutual_info=_to_result_shape_prob(mutual_info),
    )


def predict_tta(
    model: nn.Module,
    x: torch.Tensor,
    tta_fns: Sequence[Callable],
    amp: bool = True,
    return_samples: bool = True,
) -> UncertaintyResult:
    """Run all TTA transforms once (N = len(tta_fns))."""
    model.eval()
    probs = []
    for fn in tta_fns:
        x_aug, inv_fn = fn(x)
        p_aug = predict_single(model, x_aug, amp=amp)
        p = inv_fn(p_aug)
        probs.append(p)
    prob_samples = torch.stack(probs, dim=0)  # (N,B,1,Z,Y,X)
    return _aggregate_samples(prob_samples, return_samples=return_samples)


def predict_mc_dropout(
    model: nn.Module,
    x: torch.Tensor,
    n: int = 20,
    amp: bool = True,
    return_samples: bool = True,
) -> UncertaintyResult:
    """Run N stochastic forward passes with dropout enabled."""
    model.eval()
    enable_dropout_inference(model)

    probs = []
    with torch.no_grad():
        for _ in range(int(n)):
            with _autocast(x, amp):
                logits = model(x)
            probs.append(torch.sigmoid(logits))
    prob_samples = torch.stack(probs, dim=0)  # (N,B,1,Z,Y,X)
    return _aggregate_samples(prob_samples, return_samples=return_samples)


def predict_ensemble(
    models: Sequence[nn.Module],
    x: torch.Tensor,
    amp: bool = True,
    return_samples: bool = True,
) -> UncertaintyResult:
    """Run each model once and compute disagreement."""
    probs = []
    with torch.no_grad():
        for model in models:
            model.eval()
            with _autocast(x, amp):
                logits = model(x)
            probs.append(torch.sigmoid(logits))
    prob_samples = torch.stack(probs, dim=0)  # (N,B,1,Z,Y,X)
    return _aggregate_samples(prob_samples, return_samples=return_samples)


def summarize_component_uncertainty(
    prob_mean: torch.Tensor,
    unc_map: torch.Tensor,
    threshold: float = 0.5,
    min_voxels: int = 10,
) -> List[Dict]:
    """
    Summarize uncertainty per connected component of the predicted mask.

    Supports:
    - prob_mean/unc_map shape (1, Z, Y, X)
    - prob_mean/unc_map shape (B, 1, Z, Y, X) (uses first batch item)
    """
    import scipy.ndimage as ndi

    if prob_mean.ndim == 5:
        pm = prob_mean[0, 0].detach().cpu().numpy()
        um = unc_map[0, 0].detach().cpu().numpy()
    elif prob_mean.ndim == 4:
        pm = prob_mean[0].detach().cpu().numpy()
        um = unc_map[0].detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected prob_mean shape: {tuple(prob_mean.shape)}")

    pred = pm >= float(threshold)
    labels, n_labels = ndi.label(pred.astype(np.uint8))

    out: List[Dict] = []
    for comp_id in range(1, n_labels + 1):
        vox = labels == comp_id
        vox_count = int(vox.sum())
        if vox_count < int(min_voxels):
            continue

        coords = np.argwhere(vox)
        zmin, ymin, xmin = coords.min(axis=0).tolist()
        zmax, ymax, xmax = (coords.max(axis=0) + 1).tolist()
        cz, cy, cx = coords.mean(axis=0).tolist()

        out.append(
            {
                "component_id": int(comp_id),
                "voxels": vox_count,
                "mean_prob": float(pm[vox].mean()),
                "max_prob": float(pm[vox].max()),
                "mean_unc": float(um[vox].mean()),
                "max_unc": float(um[vox].max()),
                "bbox_zyx": [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)],
                "centroid_zyx": [float(cz), float(cy), float(cx)],
            }
        )

    out.sort(key=lambda item: item["voxels"], reverse=True)
    return out
