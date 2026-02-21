import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss
from monai.transforms import Compose, EnsureTyped
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medseg.augment import build_train_transforms
from medseg.config import load_config
from medseg.dataset import MedSegCachePatchDataset, extract_patch
from medseg.dataset_volume import MedSegCacheVolumeDataset
from medseg.io_registry import load_registry
from medseg.losses import masked_dice_bce_loss
from medseg.metrics import (
    batch_lesion_recall_sum_count,
    batch_surface_dice_sum_count,
    make_dice_metric,
)
from medseg.models.brainiac_wrapper import BrainIACEncoder
from medseg.models.seg_head import EncoderToSegModel
from medseg.models.triad_backbones import TriadSwinEncoder, build_triad_plain_encoder
from medseg.registry_build import ensure_training_registry
from medseg.utils import set_seed


def _normalize_slice_to_u8(slice_2d: np.ndarray) -> np.ndarray:
    x = slice_2d.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (255.0 * x).astype(np.uint8)


def _overlay_mask(base_u8: np.ndarray, mask_2d: np.ndarray, color, alpha: float = 0.35) -> np.ndarray:
    rgb = np.stack([base_u8, base_u8, base_u8], axis=-1).astype(np.float32)
    m = mask_2d.astype(bool)
    if not m.any():
        return rgb.astype(np.uint8)
    for c in range(3):
        rgb[..., c][m] = (1.0 - alpha) * rgb[..., c][m] + alpha * float(color[c])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _slice_idx_from_mask(mask_zyx: np.ndarray) -> int:
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        return int(mask_zyx.shape[0] // 2)
    return int(np.round(coords[:, 0].mean()))


def _centroid_from_mask(mask_zyx: np.ndarray):
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        z, y, x = mask_zyx.shape
        return int(z // 2), int(y // 2), int(x // 2)
    cz, cy, cx = coords.mean(axis=0)
    return int(round(cz)), int(round(cy)), int(round(cx))


def _prepare_fixed_tracking_patch(val_ds: MedSegCachePatchDataset, device: torch.device):
    """
    Build one deterministic lesion-centered patch for visual tracking.
    This avoids random val patch sampling causing empty GT masks in W&B.
    """
    patch_size = tuple(int(v) for v in val_ds.patch)

    first_item = None
    for case in val_ds.cases:
        img_czyx, lab_czyx, _, _ = val_ds._get_cached(case)
        if first_item is None:
            first_item = (case, img_czyx, lab_czyx)

        center = _centroid_from_mask(lab_czyx[0])
        x_patch = extract_patch(img_czyx, center, patch_size)
        y_patch = extract_patch(lab_czyx.astype(np.float32), center, patch_size)

        if float(y_patch.sum()) > 0.0:
            x_t = torch.from_numpy(x_patch[None]).to(device)
            y_t = torch.from_numpy(y_patch[None]).to(device)
            return x_t, y_t, case.uid

    if first_item is None:
        return None, None, None

    case, img_czyx, lab_czyx = first_item
    center = _centroid_from_mask(lab_czyx[0])
    x_patch = extract_patch(img_czyx, center, patch_size)
    y_patch = extract_patch(lab_czyx.astype(np.float32), center, patch_size)
    x_t = torch.from_numpy(x_patch[None]).to(device)
    y_t = torch.from_numpy(y_patch[None]).to(device)
    return x_t, y_t, case.uid


def _build_tracking_overlays(
    x_bczyx: torch.Tensor, y_bczyx: torch.Tensor, pred_bczyx: torch.Tensor
):
    x_np = x_bczyx[0, 0].detach().float().cpu().numpy()
    y_np = (y_bczyx[0, 0].detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    p_np = (pred_bczyx[0, 0].detach().float().cpu().numpy() > 0.5).astype(np.uint8)

    z = _slice_idx_from_mask(y_np)
    if y_np[z].sum() == 0 and p_np[z].sum() == 0:
        z = _slice_idx_from_mask(p_np)

    base = _normalize_slice_to_u8(x_np[z])
    gt_mask = (y_np[z] > 0).astype(np.uint8) * 255
    pred_mask = (p_np[z] > 0).astype(np.uint8) * 255
    diff_mask = (y_np[z] != p_np[z]).astype(np.uint8) * 255
    diff_overlay = _overlay_mask(base, diff_mask > 0, color=(255, 0, 0), alpha=0.45)
    return gt_mask, pred_mask, diff_mask, diff_overlay, z


def freeze_encoder(model: torch.nn.Module, freeze: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = not freeze


def make_optimizer(cfg: dict, model: torch.nn.Module):
    base_lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    encoder_mult = float(cfg["transfer"]["encoder_lr_mult"])

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": base_lr * encoder_mult},
            {"params": head_params, "lr": base_lr},
        ],
        weight_decay=weight_decay,
    )


def make_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, total_epochs: int):
    sched_cfg = cfg.get("train", {}).get("scheduler", {})
    name = str(sched_cfg.get("name", "cosine")).strip().lower()
    total_epochs = max(1, int(total_epochs))

    if name in {"none", "off", "disabled"}:
        return None, "none"

    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    if name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_epochs,
            eta_min=min_lr,
        )
        return sched, "cosine"

    if name in {"poly", "polynomial"}:
        power = float(sched_cfg.get("power", 0.9))
        base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        lambdas = []
        for base_lr in base_lrs:
            min_factor = float(min_lr / max(base_lr, 1e-12))

            def _poly_lambda(epoch_idx, min_factor=min_factor, power=power):
                progress = min(max((float(epoch_idx) + 1.0) / float(total_epochs), 0.0), 1.0)
                factor = (1.0 - progress) ** power
                return max(factor, min_factor)

            lambdas.append(_poly_lambda)
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambdas)
        return sched, "poly"

    raise ValueError("train.scheduler.name must be one of: ['none', 'cosine', 'poly']")


def _safe_torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older torch versions without weights_only support.
        return torch.load(path, map_location=map_location)
    except Exception:
        # Fallback for checkpoints requiring legacy unpickling behavior.
        return torch.load(path, map_location=map_location, weights_only=False)


def _load_state_dict(path: str):
    ckpt = _safe_torch_load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def _make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@contextmanager
def _autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(device_type=device.type, enabled=enabled):
            yield
        return
    with torch.cuda.amp.autocast(enabled=enabled):
        yield


class FocalTverskyLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.base = TverskyLoss(
            include_background=True,
            sigmoid=True,
            alpha=float(alpha),
            beta=float(beta),
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.base(input, target)
        return torch.pow(base_loss, self.gamma)


def _parse_val_thresholds(cfg: dict):
    raw = cfg.get("metrics", {}).get("val_thresholds", [0.5])
    if isinstance(raw, (int, float, str)):
        raw = [raw]

    thresholds = []
    for value in raw:
        try:
            thr = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= thr <= 1.0:
            thresholds.append(thr)

    if not thresholds:
        return [0.5]
    return sorted(set(float(v) for v in thresholds))


def _as_logits_list(logits):
    if isinstance(logits, (list, tuple)):
        return list(logits)
    return [logits]


def _resolve_deep_supervision_weights(cfg: dict, n_outputs: int):
    n_outputs = max(1, int(n_outputs))
    ds_cfg = cfg.get("train", {}).get("deep_supervision", {})
    enabled = bool(ds_cfg.get("enabled", False))
    if (not enabled) or n_outputs <= 1:
        return [1.0] + [0.0] * (n_outputs - 1)

    raw_weights = ds_cfg.get("weights")
    if isinstance(raw_weights, (list, tuple)) and len(raw_weights) > 0:
        weights = [max(0.0, float(v)) for v in raw_weights[:n_outputs]]
        if len(weights) < n_outputs:
            weights.extend([0.0] * (n_outputs - len(weights)))
    else:
        weights = [1.0 / (2.0**i) for i in range(n_outputs)]

    total = float(sum(weights))
    if total <= 0.0:
        return [1.0] + [0.0] * (n_outputs - 1)
    return [float(w / total) for w in weights]


def build_loss_fn(cfg: dict):
    loss_cfg = cfg.get("train", {}).get("loss", {})
    if isinstance(loss_cfg, str):
        loss_name = str(loss_cfg).strip().lower()
        loss_cfg = {}
    else:
        loss_name = str(loss_cfg.get("name", "dicece")).strip().lower()

    if loss_name in {"dicece", "dice_ce", "dice+ce"}:
        return DiceCELoss(sigmoid=True, squared_pred=True), "dicece"

    if loss_name in {"focaltversky", "focal_tversky", "focal-tversky"}:
        focal_cfg = loss_cfg.get("focaltversky", {})
        alpha = float(focal_cfg.get("alpha", 0.3))
        beta = float(focal_cfg.get("beta", 0.7))
        gamma = float(focal_cfg.get("gamma", 0.75))
        return FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma), "focaltversky"

    if loss_name in {"dicefocal", "dice_focal", "dice-focal"}:
        focal_cfg = loss_cfg.get("dicefocal", {})
        lambda_dice = float(focal_cfg.get("lambda_dice", 1.0))
        lambda_focal = float(focal_cfg.get("lambda_focal", 1.0))
        gamma = float(focal_cfg.get("gamma", 2.0))
        return (
            DiceFocalLoss(
                sigmoid=True,
                squared_pred=True,
                lambda_dice=lambda_dice,
                lambda_focal=lambda_focal,
                gamma=gamma,
            ),
            "dicefocal",
        )

    raise ValueError(
        "train.loss.name must be one of: ['dicece', 'focaltversky', 'dicefocal']"
    )


def build_model(cfg: dict) -> torch.nn.Module:
    in_channels = int(cfg["model"]["in_channels"])
    out_channels = int(cfg["model"]["out_channels"])
    model_name = cfg["model"]["name"]
    ds_cfg = cfg.get("train", {}).get("deep_supervision", {})
    ds_enabled = bool(ds_cfg.get("enabled", False))
    ds_num_outputs = int(ds_cfg.get("num_outputs", 4))

    if model_name == "triad_plain_unet":
        encoder = build_triad_plain_encoder(in_channels=in_channels)
        ckpt_path = cfg["model"]["triad"]["plain_ckpt"]
        encoder.load_state_dict(_load_state_dict(ckpt_path), strict=True)
        return EncoderToSegModel(
            encoder,
            "triad_plain",
            out_channels=out_channels,
            deep_supervision=ds_enabled,
            num_deep_supervision_outputs=ds_num_outputs,
        )

    if model_name == "triad_swinb":
        encoder = TriadSwinEncoder(in_channels=in_channels)
        ckpt_path = cfg["model"]["triad"]["swinb_ckpt"]
        encoder.load_state_dict(_load_state_dict(ckpt_path), strict=True)
        return EncoderToSegModel(
            encoder,
            "triad_swin",
            out_channels=out_channels,
            deep_supervision=ds_enabled,
            num_deep_supervision_outputs=ds_num_outputs,
        )

    if model_name == "brainiac":
        brainiac_cfg = cfg["model"]["brainiac"]
        encoder = BrainIACEncoder(
            repo_root=brainiac_cfg["repo_root"],
            checkpoint_path=brainiac_cfg["checkpoint_path"],
        )
        return EncoderToSegModel(encoder, "brainiac", out_channels=out_channels)

    raise ValueError(f"Unknown model.name: {model_name}")


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    cfg = ensure_training_registry(cfg, cfg_path)
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=cfg["wandb"].get("run_name", "train"),
        tags=cfg["wandb"].get("tags", []),
        mode=cfg["wandb"].get("mode", "online"),
        config=cfg,
    )

    cases = load_registry(cfg)
    if len(cases) <= 10:
        raise RuntimeError("Too few cases after filtering. Need more than 10 cases.")

    split_idx = max(1, int(0.8 * len(cases)))
    train_cases = cases[:split_idx]
    val_cases = cases[split_idx:] if split_idx < len(cases) else cases[-1:]

    train_tfms = build_train_transforms(cfg)
    val_tfms = Compose([EnsureTyped(keys=["image", "label", "brain_mask"], track_meta=False)])
    val_track_tfms = Compose([EnsureTyped(keys=["image", "label", "weight"], track_meta=False)])

    train_ds = MedSegCachePatchDataset(train_cases, cfg, transforms=train_tfms)
    val_ds = MedSegCacheVolumeDataset(val_cases, cfg, transforms=val_tfms)
    # Keep deterministic patch overlays for tracking while using full-volume validation.
    val_track_ds = MedSegCachePatchDataset(val_cases, cfg, transforms=val_track_tfms)
    fixed_track_x, fixed_track_y, fixed_track_uid = _prepare_fixed_tracking_patch(val_track_ds, device)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )

    model = build_model(cfg).to(device)

    # Initialize lazy segmentation head once so optimizer sees all trainable params.
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            int(cfg["model"]["in_channels"]),
            *[int(v) for v in cfg["sampling"]["patch_size"]],
            device=device,
        )
        _ = model(dummy)

    loss_fn, loss_name = build_loss_fn(cfg)
    masked_loss_cfg = cfg.get("train", {}).get("masked_loss", {})
    use_masked_loss = bool(masked_loss_cfg.get("enabled", True))
    masked_dice_weight = float(masked_loss_cfg.get("dice_weight", 1.0))
    masked_bce_weight = float(masked_loss_cfg.get("bce_weight", 1.0))
    if use_masked_loss:
        loss_name = "masked_dice_bce"
    train_dice_metric = make_dice_metric(include_background=False, reduction="mean")
    val_thresholds = _parse_val_thresholds(cfg)
    val_dice_metrics = {
        float(thr): make_dice_metric(include_background=False, reduction="mean")
        for thr in val_thresholds
    }

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = _make_grad_scaler(device, enabled=use_amp)

    freeze_epochs = int(cfg["transfer"]["freeze_encoder_epochs"])
    freeze_encoder(model, True)
    optimizer = make_optimizer(cfg, model)
    scheduler, scheduler_name = make_scheduler(
        cfg,
        optimizer,
        total_epochs=int(cfg["train"]["epochs"]),
    )
    spacing_zyx = None
    target_spacing = cfg.get("preprocess", {}).get("target_spacing")
    if target_spacing is not None and len(target_spacing) == 3:
        spacing_zyx = (
            float(target_spacing[2]),
            float(target_spacing[1]),
            float(target_spacing[0]),
        )
    surface_tol_mm = float(
        cfg.get("metrics", {}).get("surface_dice_tolerance_mm", 1.0)
    )
    val_cfg = cfg.get("validation", {})
    val_roi_size = tuple(
        int(v) for v in val_cfg.get("roi_size", cfg["sampling"]["patch_size"])
    )
    val_sw_batch_size = int(val_cfg.get("sw_batch_size", 1))
    val_sw_overlap = float(val_cfg.get("sw_overlap", 0.25))
    val_sw_mode = str(val_cfg.get("sw_mode", "gaussian"))

    best = -1.0
    best_epoch = -1
    best_train_dice = 0.0
    bad_epochs = 0
    patience = int(cfg["train"]["early_stop_patience"])

    checkpoint_dir = cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()

        if epoch == freeze_epochs:
            freeze_encoder(model, False)
            optimizer = make_optimizer(cfg, model)
            scheduler, scheduler_name = make_scheduler(
                cfg,
                optimizer,
                total_epochs=max(1, int(cfg["train"]["epochs"]) - epoch),
            )

        train_loss = 0.0
        num_batches = 0
        train_dice_metric.reset()
        train_surface_sum = 0.0
        train_surface_n = 0
        train_lesion_sum = 0.0
        train_lesion_n = 0
        train_num_outputs = 1
        printed_train_batch_info = False
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            w = batch.get("weight")
            if w is None:
                w = torch.ones_like(y)
            else:
                w = w.to(device)
            if not printed_train_batch_info:
                print(f"[train][epoch {epoch}] img.shape={tuple(x.shape)} label.shape={tuple(y.shape)}")
                print(f"[train][epoch {epoch}] label.unique()={torch.unique(y).detach().cpu().tolist()}")
                printed_train_batch_info = True

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, enabled=use_amp):
                logits_out = model(x)
                logits_list = _as_logits_list(logits_out)
                train_num_outputs = len(logits_list)
                ds_weights = _resolve_deep_supervision_weights(cfg, len(logits_list))
                loss = 0.0
                for ds_idx, (logits_ds, ds_weight) in enumerate(zip(logits_list, ds_weights)):
                    if float(ds_weight) <= 0.0:
                        continue
                    y_ds = y
                    w_ds = w
                    if logits_ds.shape[-3:] != y.shape[-3:]:
                        y_ds = F.interpolate(y, size=logits_ds.shape[-3:], mode="nearest")
                        w_ds = F.interpolate(w, size=logits_ds.shape[-3:], mode="nearest")
                    if use_masked_loss:
                        ds_loss = masked_dice_bce_loss(
                            logits_ds,
                            y_ds,
                            w_ds,
                            dice_weight=masked_dice_weight,
                            bce_weight=masked_bce_weight,
                        )
                    else:
                        ds_loss = loss_fn(logits_ds, y_ds)
                    loss = loss + float(ds_weight) * ds_loss
                logits = logits_list[0]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["train"]["grad_clip"])
            )
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                train_pred = (torch.sigmoid(logits.detach()) > 0.5).float()
                train_weight = (w > 0.5).float()
                train_pred_masked = train_pred * train_weight
                y_masked = y * train_weight
                train_dice_metric(y_pred=train_pred_masked, y=y_masked)
                surf_sum, surf_n = batch_surface_dice_sum_count(
                    train_pred_masked,
                    y_masked,
                    spacing_zyx=spacing_zyx,
                    tolerance_mm=surface_tol_mm,
                )
                lrec_sum, lrec_n = batch_lesion_recall_sum_count(train_pred_masked, y_masked)
                train_surface_sum += float(surf_sum)
                train_surface_n += int(surf_n)
                train_lesion_sum += float(lrec_sum)
                train_lesion_n += int(lrec_n)

            train_loss += float(loss.item())
            num_batches += 1

        model.eval()
        for metric in val_dice_metrics.values():
            metric.reset()
        track_batch = None
        val_pred_fg_vox_by_thr = {float(thr): 0 for thr in val_thresholds}
        val_gt_fg_vox = 0
        val_total_vox = 0
        val_surface_sum = 0.0
        val_surface_n = 0
        val_lesion_sum = 0.0
        val_lesion_n = 0
        printed_val_batch_info = False
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                brain_mask = batch.get("brain_mask")
                if brain_mask is None:
                    brain_mask = torch.ones_like(y)
                else:
                    brain_mask = brain_mask.to(device)
                brain_weight = (brain_mask > 0.5).float()
                if not printed_val_batch_info:
                    print(f"[val][epoch {epoch}] img.shape={tuple(x.shape)} label.shape={tuple(y.shape)}")
                    print(f"[val][epoch {epoch}] label.unique()={torch.unique(y).detach().cpu().tolist()}")
                    printed_val_batch_info = True
                logits = sliding_window_inference(
                    x,
                    roi_size=val_roi_size,
                    sw_batch_size=val_sw_batch_size,
                    predictor=model,
                    overlap=val_sw_overlap,
                    mode=val_sw_mode,
                )
                probs = torch.sigmoid(logits)
                pred_for_reporting = (probs > 0.5).float()
                pred_for_reporting_masked = pred_for_reporting * brain_weight
                y_masked = y * brain_weight
                for thr in val_thresholds:
                    pred_thr = (probs > float(thr)).float()
                    pred_thr_masked = pred_thr * brain_weight
                    val_dice_metrics[float(thr)](y_pred=pred_thr_masked, y=y_masked)
                    val_pred_fg_vox_by_thr[float(thr)] += int(pred_thr_masked.sum().item())
                surf_sum, surf_n = batch_surface_dice_sum_count(
                    pred_for_reporting_masked,
                    y_masked,
                    spacing_zyx=spacing_zyx,
                    tolerance_mm=surface_tol_mm,
                )
                lrec_sum, lrec_n = batch_lesion_recall_sum_count(pred_for_reporting_masked, y_masked)
                val_surface_sum += float(surf_sum)
                val_surface_n += int(surf_n)
                val_lesion_sum += float(lrec_sum)
                val_lesion_n += int(lrec_n)
                val_gt_fg_vox += int(y_masked.sum().item())
                val_total_vox += int(brain_weight.sum().item())
                if track_batch is None:
                    track_batch = (
                        x.detach().cpu(),
                        y.detach().cpu(),
                        pred_for_reporting.detach().cpu(),
                    )

        if fixed_track_x is not None and fixed_track_y is not None:
            with torch.no_grad():
                fixed_pred = (torch.sigmoid(model(fixed_track_x)) > 0.5).float()
            track_batch = (
                fixed_track_x.detach().cpu(),
                fixed_track_y.detach().cpu(),
                fixed_pred.detach().cpu(),
            )

        train_dice = float(train_dice_metric.aggregate().item()) if num_batches > 0 else 0.0
        val_dice_by_thr = {
            float(thr): float(val_dice_metrics[float(thr)].aggregate().item())
            for thr in val_thresholds
        }
        best_val_thr = max(val_dice_by_thr, key=val_dice_by_thr.get)
        val_dice = float(val_dice_by_thr[best_val_thr])
        val_pred_fg_vox = int(val_pred_fg_vox_by_thr[best_val_thr])
        train_surface_dice = train_surface_sum / max(1, train_surface_n)
        train_lesion_recall = train_lesion_sum / max(1, train_lesion_n)
        val_surface_dice = val_surface_sum / max(1, val_surface_n)
        val_lesion_recall = val_lesion_sum / max(1, val_lesion_n)
        mean_train_loss = train_loss / max(1, num_batches)
        pred_fg_ratio = float(val_pred_fg_vox / max(1, val_total_vox))
        gt_fg_ratio = float(val_gt_fg_vox / max(1, val_total_vox))
        pred_to_gt_fg = float(val_pred_fg_vox / max(1, val_gt_fg_vox))

        log_payload = {
            "epoch": epoch,
            "train/loss": mean_train_loss,
            "train/dice": train_dice,
            "train/surface_dice": train_surface_dice,
            "train/lesion_recall": train_lesion_recall,
            "val/dice": val_dice,
            "val/best_threshold": float(best_val_thr),
            "val/mode": "full_volume_sliding_window",
            "val/surface_dice": val_surface_dice,
            "val/lesion_recall": val_lesion_recall,
            "val/pred_fg_ratio": pred_fg_ratio,
            "val/gt_fg_ratio": gt_fg_ratio,
            "val/pred_to_gt_fg": pred_to_gt_fg,
            "train/loss_name": loss_name,
            "train/deep_supervision_outputs": int(train_num_outputs),
            "train/scheduler": scheduler_name,
            "train/lr_encoder": float(optimizer.param_groups[0]["lr"]),
            "train/lr_head": float(optimizer.param_groups[-1]["lr"]),
        }
        for thr, score in val_dice_by_thr.items():
            log_payload[f"val/dice@{thr:.2f}"] = float(score)
        if track_batch is not None:
            gt_mask, pred_mask, diff_mask, diff_overlay, z_mid = _build_tracking_overlays(*track_batch)
            uid_suffix = f" uid={fixed_track_uid}" if fixed_track_uid else ""
            log_payload["track/val_mask_gt"] = wandb.Image(
                gt_mask, caption=f"Epoch {epoch} GT mask z={z_mid}{uid_suffix}"
            )
            log_payload["track/val_mask_pred"] = wandb.Image(
                pred_mask, caption=f"Epoch {epoch} Predicted mask z={z_mid}{uid_suffix}"
            )
            log_payload["track/val_mask_diff"] = wandb.Image(
                diff_mask, caption=f"Epoch {epoch} |GT-Pred| mask diff z={z_mid}{uid_suffix}"
            )
            log_payload["track/val_mask_diff_on_ct"] = wandb.Image(
                diff_overlay, caption=f"Epoch {epoch} mask diff over CT z={z_mid}{uid_suffix}"
            )
        wandb.log(log_payload)
        print(
            f"Epoch {epoch:04d} "
            f"train_loss={mean_train_loss:.4f} "
            f"train_dice={train_dice:.4f} "
            f"train_surface={train_surface_dice:.4f} "
            f"train_lesion_recall={train_lesion_recall:.4f} "
            f"val_dice={val_dice:.4f} "
            f"val_best_thr={best_val_thr:.2f} "
            f"val_surface={val_surface_dice:.4f} "
            f"val_lesion_recall={val_lesion_recall:.4f} "
            f"lr_head={float(optimizer.param_groups[-1]['lr']):.2e}"
        )
        print(
            f"[scalar][epoch {epoch}] "
            f"train_loss.item()={mean_train_loss:.6f} "
            f"train_dice.item()={train_dice:.6f} "
            f"val_dice.item()={val_dice:.6f}"
        )

        model_tag = cfg["model"]["name"]
        metric_path = os.path.join(checkpoint_dir, f"best_metric_{model_tag}.json")
        weight_path = os.path.join(checkpoint_dir, f"best_{model_tag}.pth")

        if val_dice > best:
            best = val_dice
            best_epoch = int(epoch)
            best_train_dice = float(train_dice)
            bad_epochs = 0
            torch.save(model.state_dict(), weight_path)
            with open(metric_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "best_val_dice": float(best),
                        "best_epoch": int(best_epoch),
                        "train_dice_at_best": float(best_train_dice),
                        "best_val_threshold": float(best_val_thr),
                    },
                    file,
                )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

        if scheduler is not None:
            scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()
    main(args.cfg)
