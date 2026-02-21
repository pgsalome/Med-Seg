import numpy as np
import torch
from monai.metrics import DiceMetric


def make_dice_metric(include_background: bool = True, reduction: str = "mean") -> DiceMetric:
    return DiceMetric(include_background=include_background, reduction=reduction)


def binary_dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > threshold).float()
    tgt = (target > 0.5).float()

    dims = tuple(range(1, pred.ndim))
    intersection = (pred * tgt).sum(dim=dims)
    denominator = pred.sum(dim=dims) + tgt.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def binary_dice_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    intersection = float((pred & target).sum())
    denominator = float(pred.sum() + target.sum())
    return (2.0 * intersection + eps) / (denominator + eps)


def binary_surface_dice_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    spacing_zyx=None,
    tolerance_mm: float = 1.0,
    eps: float = 1e-8,
) -> float:
    import scipy.ndimage as ndi

    pred = (pred > 0).astype(bool)
    target = (target > 0).astype(bool)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch for surface dice: pred={pred.shape} target={target.shape}")

    if not target.any():
        return float("nan")
    if not pred.any():
        return 0.0

    structure = ndi.generate_binary_structure(pred.ndim, 1)
    pred_eroded = ndi.binary_erosion(pred, structure=structure, border_value=0)
    target_eroded = ndi.binary_erosion(target, structure=structure, border_value=0)
    pred_surface = pred & (~pred_eroded)
    target_surface = target & (~target_eroded)
    if not pred_surface.any():
        pred_surface = pred.copy()
    if not target_surface.any():
        target_surface = target.copy()

    if spacing_zyx is None:
        sampling = (1.0,) * pred.ndim
    else:
        sampling = tuple(float(v) for v in spacing_zyx)
        if len(sampling) != pred.ndim:
            sampling = (1.0,) * pred.ndim

    dist_to_target = ndi.distance_transform_edt(~target_surface, sampling=sampling)
    dist_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=sampling)

    pred_close = float((dist_to_target[pred_surface] <= float(tolerance_mm)).sum())
    target_close = float((dist_to_pred[target_surface] <= float(tolerance_mm)).sum())
    denom = float(pred_surface.sum() + target_surface.sum())
    return (pred_close + target_close + eps) / (denom + eps)


def lesion_recall_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    import scipy.ndimage as ndi

    pred = (pred > 0).astype(bool)
    target = (target > 0).astype(bool)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch for lesion recall: pred={pred.shape} target={target.shape}")

    if not target.any():
        return float("nan")

    structure = ndi.generate_binary_structure(target.ndim, 1)
    labeled, num_components = ndi.label(target, structure=structure)
    if num_components <= 0:
        return float("nan")

    detected = 0
    for comp_id in range(1, int(num_components) + 1):
        comp_mask = labeled == comp_id
        if np.any(pred & comp_mask):
            detected += 1
    return float(detected) / float(num_components)


def batch_surface_dice_sum_count(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing_zyx=None,
    tolerance_mm: float = 1.0,
):
    pred_np = (pred.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    target_np = (target.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    if pred_np.ndim == 5:
        pred_np = pred_np[:, 0]
    if target_np.ndim == 5:
        target_np = target_np[:, 0]

    score_sum = 0.0
    valid = 0
    for p, t in zip(pred_np, target_np):
        score = binary_surface_dice_numpy(
            p,
            t,
            spacing_zyx=spacing_zyx,
            tolerance_mm=float(tolerance_mm),
        )
        if np.isfinite(score):
            score_sum += float(score)
            valid += 1
    return float(score_sum), int(valid)


def batch_lesion_recall_sum_count(pred: torch.Tensor, target: torch.Tensor):
    pred_np = (pred.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    target_np = (target.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    if pred_np.ndim == 5:
        pred_np = pred_np[:, 0]
    if target_np.ndim == 5:
        target_np = target_np[:, 0]

    score_sum = 0.0
    valid = 0
    for p, t in zip(pred_np, target_np):
        score = lesion_recall_numpy(p, t)
        if np.isfinite(score):
            score_sum += float(score)
            valid += 1
    return float(score_sum), int(valid)
