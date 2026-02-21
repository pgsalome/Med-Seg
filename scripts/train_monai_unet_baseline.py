import json
import os
import sys
from contextlib import contextmanager

import torch
import wandb
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureTyped
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medseg.augment import build_train_transforms
from medseg.config import load_config
from medseg.dataset import MedSegCachePatchDataset
from medseg.dataset_volume import MedSegCacheVolumeDataset
from medseg.io_registry import load_registry
from medseg.losses import masked_dice_bce_loss
from medseg.metrics import make_dice_metric
from medseg.registry_build import ensure_training_registry
from medseg.utils import set_seed


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


def build_monai_unet(cfg: dict) -> torch.nn.Module:
    in_channels = int(cfg["model"]["in_channels"])
    out_channels = int(cfg["model"]["out_channels"])
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=0.0,
    )


def main(cfg_path: str, epochs: int, early_stop_patience: int = 0):
    if epochs < 1:
        raise ValueError("--epochs must be >= 1")

    cfg = load_config(cfg_path)
    cfg = ensure_training_registry(cfg, cfg_path)
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    run_name = f"{cfg['wandb'].get('run_name', 'run')}-monai-unet-scratch"
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=run_name,
        tags=cfg["wandb"].get("tags", []) + ["baseline", "monai_unet", "scratch"],
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

    train_ds = MedSegCachePatchDataset(train_cases, cfg, transforms=train_tfms)
    val_ds = MedSegCacheVolumeDataset(val_cases, cfg, transforms=val_tfms)

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

    model = build_monai_unet(cfg).to(device)

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True)
    masked_loss_cfg = cfg.get("train", {}).get("masked_loss", {})
    use_masked_loss = bool(masked_loss_cfg.get("enabled", True))
    masked_dice_weight = float(masked_loss_cfg.get("dice_weight", 1.0))
    masked_bce_weight = float(masked_loss_cfg.get("bce_weight", 1.0))
    train_dice_metric = make_dice_metric(include_background=True, reduction="mean")
    val_thresholds = _parse_val_thresholds(cfg)
    val_dice_metrics = {
        float(thr): make_dice_metric(include_background=True, reduction="mean")
        for thr in val_thresholds
    }

    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = _make_grad_scaler(device, enabled=use_amp)
    val_cfg = cfg.get("validation", {})
    val_roi_size = tuple(
        int(v) for v in val_cfg.get("roi_size", cfg["sampling"]["patch_size"])
    )
    val_sw_batch_size = int(val_cfg.get("sw_batch_size", 1))
    val_sw_overlap = float(val_cfg.get("sw_overlap", 0.25))
    val_sw_mode = str(val_cfg.get("sw_mode", "gaussian"))

    best_val = -1.0
    bad_epochs = 0
    checkpoint_dir = cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metric_path = os.path.join(checkpoint_dir, "best_metric_monai_unet_baseline.json")
    weight_path = os.path.join(checkpoint_dir, "best_monai_unet_baseline.pth")

    for epoch in range(int(epochs)):
        model.train()
        train_loss = 0.0
        train_steps = 0
        train_dice_metric.reset()
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
                logits = model(x)
                if use_masked_loss:
                    loss = masked_dice_bce_loss(
                        logits,
                        y,
                        w,
                        dice_weight=masked_dice_weight,
                        bce_weight=masked_bce_weight,
                    )
                else:
                    loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                train_pred = (torch.sigmoid(logits.detach()) > 0.5).float()
                train_weight = (w > 0.5).float()
                train_dice_metric(y_pred=train_pred * train_weight, y=y * train_weight)

            train_loss += float(loss.item())
            train_steps += 1

        model.eval()
        for metric in val_dice_metrics.values():
            metric.reset()
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
                for thr in val_thresholds:
                    pred_thr = (probs > float(thr)).float()
                    val_dice_metrics[float(thr)](y_pred=pred_thr * brain_weight, y=y * brain_weight)

        train_dice = float(train_dice_metric.aggregate().item()) if train_steps > 0 else 0.0
        train_loss_mean = train_loss / max(1, train_steps)
        val_dice_by_thr = {
            float(thr): float(val_dice_metrics[float(thr)].aggregate().item())
            for thr in val_thresholds
        }
        best_val_thr = max(val_dice_by_thr, key=val_dice_by_thr.get)
        val_dice = float(val_dice_by_thr[best_val_thr])

        log_payload = {
            "epoch": epoch,
            "train/loss": train_loss_mean,
            "train/dice": train_dice,
            "val/dice": val_dice,
            "val/best_threshold": float(best_val_thr),
            "val/mode": "full_volume_sliding_window",
            "train/model": "monai_unet_baseline_scratch",
        }
        for thr, score in val_dice_by_thr.items():
            log_payload[f"val/dice@{thr:.2f}"] = float(score)
        wandb.log(log_payload)
        print(
            f"[baseline-unet] epoch {epoch:04d} "
            f"train_loss={train_loss_mean:.4f} "
            f"train_dice={train_dice:.4f} "
            f"val_dice={val_dice:.4f} "
            f"val_best_thr={best_val_thr:.2f}"
        )
        print(
            f"[baseline-unet][scalar][epoch {epoch}] "
            f"train_loss.item()={train_loss_mean:.6f} "
            f"train_dice.item()={train_dice:.6f} "
            f"val_dice.item()={val_dice:.6f}"
        )

        if val_dice > best_val:
            best_val = val_dice
            bad_epochs = 0
            torch.save(model.state_dict(), weight_path)
            with open(metric_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "model": "monai_unet_baseline_scratch",
                        "best_val_dice": float(best_val),
                        "best_epoch": int(epoch),
                        "best_val_threshold": float(best_val_thr),
                        "epochs_requested": int(epochs),
                    },
                    file,
                    indent=2,
                )
        else:
            bad_epochs += 1
            if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
                print("[baseline-unet] early stopping.")
                break

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a simple MONAI 3D UNet baseline from scratch."
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Disable early stopping with 0 (default).",
    )
    args = parser.parse_args()
    main(args.cfg, epochs=int(args.epochs), early_stop_patience=int(args.early_stop_patience))
