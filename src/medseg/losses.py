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


def masked_soft_dice_loss_per_sample(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns Dice loss per sample (shape: [B]), averaged over channels.
    """
    m = _as_float_mask(mask)
    p = probs.float() * m
    t = targets.float() * m
    dims = tuple(range(2, p.ndim))
    inter = (p * t).sum(dim=dims)
    denom = p.sum(dim=dims) + t.sum(dim=dims)
    dice = (2.0 * inter + float(eps)) / (denom + float(eps))
    return 1.0 - dice.mean(dim=1)


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


def masked_dice_bce_loss_volume_aware(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    min_fg_voxels: int = 50,
) -> torch.Tensor:
    """
    Volume-aware masked Dice+BCE loss.

    Dice is computed per sample and only applied to samples with at least
    `min_fg_voxels` foreground voxels after masking. BCE is always applied.
    """
    probs = torch.sigmoid(logits)
    bce = masked_bce_with_logits(logits, targets, mask)

    # Foreground voxel count per sample after masking.
    m = _as_float_mask(mask)
    tgt_masked = (targets.float() * m).detach()
    fg_voxels_per_sample = tgt_masked.sum(dim=tuple(range(1, tgt_masked.ndim)))
    valid = fg_voxels_per_sample >= float(max(0, int(min_fg_voxels)))

    dice_per_sample = masked_soft_dice_loss_per_sample(probs, targets, mask)
    if bool(valid.any().item()):
        dice_loss = dice_per_sample[valid].mean()
    else:
        dice_loss = logits.new_tensor(0.0)

    return float(dice_weight) * dice_loss + float(bce_weight) * bce


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
