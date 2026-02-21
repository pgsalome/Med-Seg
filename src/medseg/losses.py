import torch
import torch.nn.functional as F


def _as_float_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask > 0.5).float()


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    m = _as_float_mask(mask)
    return (loss * m).sum() / (m.sum() + float(eps))


def masked_soft_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    m = _as_float_mask(mask)
    p = probs.float() * m
    t = targets.float() * m
    dims = tuple(range(2, p.ndim))
    inter = (p * t).sum(dim=dims)
    denom = p.sum(dim=dims) + t.sum(dim=dims)
    dice = (2.0 * inter + float(eps)) / (denom + float(eps))
    return 1.0 - dice.mean()


def masked_dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    d = masked_soft_dice_loss(probs, targets, mask)
    b = masked_bce_with_logits(logits, targets, mask)
    return float(dice_weight) * d + float(bce_weight) * b


def masked_binary_dice_score(
    pred_binary: torch.Tensor,
    target_binary: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    m = _as_float_mask(mask)
    p = (pred_binary > 0.5).float() * m
    t = (target_binary > 0.5).float() * m
    dims = tuple(range(2, p.ndim))
    inter = (p * t).sum(dim=dims)
    denom = p.sum(dim=dims) + t.sum(dim=dims)
    dice = (2.0 * inter + float(eps)) / (denom + float(eps))
    return dice.mean()
