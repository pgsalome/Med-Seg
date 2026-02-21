import copy
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureTyped
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medseg.augment import build_train_transforms
from medseg.config import load_config
from medseg.dataset import MedSegCachePatchDataset
from medseg.dataset_volume import MedSegCacheVolumeDataset
from medseg.io_registry import CaseItem, load_registry
from medseg.losses import masked_dice_bce_loss
from medseg.metrics import (
    batch_lesion_recall_sum_count,
    batch_surface_dice_sum_count,
    make_dice_metric,
)
from medseg.registry_build import ensure_training_registry
from medseg.utils import set_seed
try:
    from scripts.train import (
        _autocast,
        _build_tracking_overlays,
        _make_grad_scaler,
        _parse_val_thresholds,
        _prepare_fixed_tracking_patch,
        _as_logits_list,
        _resolve_deep_supervision_weights,
        build_loss_fn,
        build_model,
        freeze_encoder,
        make_optimizer,
        make_scheduler,
    )
except ModuleNotFoundError:
    from train import (
        _autocast,
        _build_tracking_overlays,
        _make_grad_scaler,
        _parse_val_thresholds,
        _prepare_fixed_tracking_patch,
        _as_logits_list,
        _resolve_deep_supervision_weights,
        build_loss_fn,
        build_model,
        freeze_encoder,
        make_optimizer,
        make_scheduler,
    )


def patient_folds(cases: List[CaseItem], num_folds: int, seed: int):
    grouped: Dict[str, List[CaseItem]] = {}
    for case in cases:
        grouped.setdefault(case.patient_id, []).append(case)

    patient_ids = list(grouped.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    folds: List[List[str]] = [[] for _ in range(num_folds)]
    for idx, patient_id in enumerate(patient_ids):
        folds[idx % num_folds].append(patient_id)

    return folds, grouped


def expand_cases(patient_ids: List[str], grouped: Dict[str, List[CaseItem]]) -> List[CaseItem]:
    expanded: List[CaseItem] = []
    for patient_id in patient_ids:
        expanded.extend(grouped[patient_id])
    return expanded


def _mean_std(values: List[float]):
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def run_fold(cfg: dict, fold_idx: int, train_cases: List[CaseItem], val_cases: List[CaseItem]) -> dict:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run_name = cfg["wandb"].get("run_name", "train-cv")
    run_tags = cfg["wandb"].get("tags", []) + [f"cv-fold-{fold_idx}"]

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=f"{run_name}-fold{fold_idx}",
        tags=run_tags,
        mode=cfg["wandb"].get("mode", "online"),
        config=cfg,
    )

    train_tfms = build_train_transforms(cfg)
    val_tfms = Compose([EnsureTyped(keys=["image", "label", "brain_mask"], track_meta=False)])
    val_track_tfms = Compose([EnsureTyped(keys=["image", "label", "weight"], track_meta=False)])

    train_ds = MedSegCachePatchDataset(train_cases, cfg, transforms=train_tfms)
    val_ds = MedSegCacheVolumeDataset(val_cases, cfg, transforms=val_tfms)
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

    checkpoint_root = cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    checkpoint_dir = os.path.join(checkpoint_root, f"fold_{fold_idx}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best = -1.0
    best_epoch = -1
    best_train_dice = 0.0
    bad_epochs = 0
    patience = int(cfg["train"]["early_stop_patience"])
    model_tag = cfg["model"]["name"]
    metric_path = os.path.join(checkpoint_dir, f"best_metric_{model_tag}.json")
    weight_path = os.path.join(checkpoint_dir, f"best_{model_tag}.pth")
    fold_history: List[dict] = []

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
        train_steps = 0
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
                print(
                    f"[fold {fold_idx}][train][epoch {epoch}] "
                    f"img.shape={tuple(x.shape)} label.shape={tuple(y.shape)}"
                )
                print(
                    f"[fold {fold_idx}][train][epoch {epoch}] "
                    f"label.unique()={torch.unique(y).detach().cpu().tolist()}"
                )
                printed_train_batch_info = True

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, enabled=use_amp):
                logits_out = model(x)
                logits_list = _as_logits_list(logits_out)
                train_num_outputs = len(logits_list)
                ds_weights = _resolve_deep_supervision_weights(cfg, len(logits_list))
                loss = 0.0
                for logits_ds, ds_weight in zip(logits_list, ds_weights):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
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
            train_steps += 1

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
                    print(
                        f"[fold {fold_idx}][val][epoch {epoch}] "
                        f"img.shape={tuple(x.shape)} label.shape={tuple(y.shape)}"
                    )
                    print(
                        f"[fold {fold_idx}][val][epoch {epoch}] "
                        f"label.unique()={torch.unique(y).detach().cpu().tolist()}"
                    )
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

        train_dice = (
            float(train_dice_metric.aggregate().item()) if train_steps > 0 else 0.0
        )
        val_dice_by_thr = {
            float(thr): float(val_dice_metrics[float(thr)].aggregate().item())
            for thr in val_thresholds
        }
        best_val_thr = max(val_dice_by_thr, key=val_dice_by_thr.get)
        val_pred_fg_vox = int(val_pred_fg_vox_by_thr[best_val_thr])
        pred_fg_ratio = float(val_pred_fg_vox / max(1, val_total_vox))
        gt_fg_ratio = float(val_gt_fg_vox / max(1, val_total_vox))
        pred_to_gt_fg = float(val_pred_fg_vox / max(1, val_gt_fg_vox))
        train_surface_dice = train_surface_sum / max(1, train_surface_n)
        train_lesion_recall = train_lesion_sum / max(1, train_lesion_n)
        val_dice = float(val_dice_by_thr[best_val_thr])
        val_surface_dice = val_surface_sum / max(1, val_surface_n)
        val_lesion_recall = val_lesion_sum / max(1, val_lesion_n)
        mean_train_loss = train_loss / max(1, train_steps)
        log_payload = {
            "fold": fold_idx,
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
            log_payload[f"track/fold_{fold_idx}_val_mask_gt"] = wandb.Image(
                gt_mask, caption=f"Fold {fold_idx} epoch {epoch} GT mask z={z_mid}{uid_suffix}"
            )
            log_payload[f"track/fold_{fold_idx}_val_mask_pred"] = wandb.Image(
                pred_mask, caption=f"Fold {fold_idx} epoch {epoch} Predicted mask z={z_mid}{uid_suffix}"
            )
            log_payload[f"track/fold_{fold_idx}_val_mask_diff"] = wandb.Image(
                diff_mask, caption=f"Fold {fold_idx} epoch {epoch} |GT-Pred| z={z_mid}{uid_suffix}"
            )
            log_payload[f"track/fold_{fold_idx}_val_mask_diff_on_ct"] = wandb.Image(
                diff_overlay, caption=f"Fold {fold_idx} epoch {epoch} diff on CT z={z_mid}{uid_suffix}"
            )
        wandb.log(log_payload)
        print(
            f"[fold {fold_idx}] epoch {epoch:04d} "
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
            f"[fold {fold_idx}][scalar][epoch {epoch}] "
            f"train_loss.item()={mean_train_loss:.6f} "
            f"train_dice.item()={train_dice:.6f} "
            f"val_dice.item()={val_dice:.6f}"
        )
        fold_history.append(
            {
                "fold": fold_idx,
                "epoch": int(epoch),
                "train_loss": float(mean_train_loss),
                "train_dice": float(train_dice),
                "train_surface_dice": float(train_surface_dice),
                "train_lesion_recall": float(train_lesion_recall),
                "val_dice": float(val_dice),
                "val_surface_dice": float(val_surface_dice),
                "val_lesion_recall": float(val_lesion_recall),
            }
        )

        if val_dice > best:
            best = val_dice
            best_epoch = int(epoch)
            best_train_dice = float(train_dice)
            bad_epochs = 0
            torch.save(model.state_dict(), weight_path)
            with open(metric_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "fold": fold_idx,
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
                print(f"[fold {fold_idx}] early stopping.")
                break

        if scheduler is not None:
            scheduler.step()

    wandb.finish()
    fold_history_path = os.path.join(checkpoint_dir, "fold_history.json")
    with open(fold_history_path, "w", encoding="utf-8") as file:
        json.dump(fold_history, file, indent=2)
    return {
        "fold": fold_idx,
        "best_val_dice": float(best),
        "best_epoch": int(best_epoch),
        "train_dice_at_best": float(best_train_dice),
        "epochs_ran": len(fold_history),
        "history_path": fold_history_path,
        "history": fold_history,
    }


def main(cfg_path: str, folds: int = 5):
    cfg = load_config(cfg_path)
    cfg = ensure_training_registry(cfg, cfg_path)
    set_seed(int(cfg["seed"]))

    cases = load_registry(cfg)
    if len(cases) <= folds:
        raise RuntimeError("Too few cases for requested CV folds after filtering.")

    fold_ids, grouped = patient_folds(cases, num_folds=folds, seed=int(cfg["seed"]))
    all_ids = set(grouped.keys())

    results = []
    fold_histories: Dict[int, List[dict]] = {}
    for fold_idx, val_ids in enumerate(fold_ids):
        val_set = set(val_ids)
        train_set = list(all_ids - val_set)
        train_cases = expand_cases(train_set, grouped)
        val_cases = expand_cases(val_ids, grouped)
        if not train_cases or not val_cases:
            continue

        fold_cfg = copy.deepcopy(cfg)
        fold_result = run_fold(fold_cfg, fold_idx, train_cases, val_cases)
        fold_histories[fold_idx] = fold_result["history"]
        results.append(
            {
                "fold": fold_idx,
                "train_patients": len(train_set),
                "val_patients": len(val_ids),
                "best_val_dice": float(fold_result["best_val_dice"]),
                "best_epoch": int(fold_result["best_epoch"]),
                "train_dice_at_best": float(fold_result["train_dice_at_best"]),
                "epochs_ran": int(fold_result["epochs_ran"]),
                "history_path": fold_result["history_path"],
            }
        )

    epoch_aggregate: List[dict] = []
    if fold_histories:
        max_epoch = max(
            (h["epoch"] for history in fold_histories.values() for h in history),
            default=-1,
        )
        for epoch in range(max_epoch + 1):
            epoch_rows = [
                h
                for history in fold_histories.values()
                for h in history
                if h["epoch"] == epoch
            ]
            if not epoch_rows:
                continue
            train_loss_mean, train_loss_std = _mean_std([r["train_loss"] for r in epoch_rows])
            train_dice_mean, train_dice_std = _mean_std([r["train_dice"] for r in epoch_rows])
            train_surface_mean, train_surface_std = _mean_std(
                [r["train_surface_dice"] for r in epoch_rows]
            )
            train_lesion_recall_mean, train_lesion_recall_std = _mean_std(
                [r["train_lesion_recall"] for r in epoch_rows]
            )
            val_dice_mean, val_dice_std = _mean_std([r["val_dice"] for r in epoch_rows])
            val_surface_mean, val_surface_std = _mean_std(
                [r["val_surface_dice"] for r in epoch_rows]
            )
            val_lesion_recall_mean, val_lesion_recall_std = _mean_std(
                [r["val_lesion_recall"] for r in epoch_rows]
            )
            epoch_aggregate.append(
                {
                    "epoch": int(epoch),
                    "num_folds": len(epoch_rows),
                    "train_loss_mean": train_loss_mean,
                    "train_loss_std": train_loss_std,
                    "train_dice_mean": train_dice_mean,
                    "train_dice_std": train_dice_std,
                    "train_surface_dice_mean": train_surface_mean,
                    "train_surface_dice_std": train_surface_std,
                    "train_lesion_recall_mean": train_lesion_recall_mean,
                    "train_lesion_recall_std": train_lesion_recall_std,
                    "val_dice_mean": val_dice_mean,
                    "val_dice_std": val_dice_std,
                    "val_surface_dice_mean": val_surface_mean,
                    "val_surface_dice_std": val_surface_std,
                    "val_lesion_recall_mean": val_lesion_recall_mean,
                    "val_lesion_recall_std": val_lesion_recall_std,
                }
            )

    out_dir = cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "cv_summary.json")
    fold_csv_path = os.path.join(out_dir, "cv_fold_results.csv")
    epoch_csv_path = os.path.join(out_dir, "cv_epoch_aggregate.csv")
    mean_best, std_best = _mean_std([r["best_val_dice"] for r in results])
    mean_train_dice_at_best, std_train_dice_at_best = _mean_std(
        [r["train_dice_at_best"] for r in results]
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "folds": folds,
                "results": results,
                "epoch_aggregate": epoch_aggregate,
                "mean_best_val_dice": mean_best,
                "std_best_val_dice": std_best,
                "mean_train_dice_at_best": mean_train_dice_at_best,
                "std_train_dice_at_best": std_train_dice_at_best,
            },
            file,
            indent=2,
        )

    with open(fold_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fold",
                "train_patients",
                "val_patients",
                "best_val_dice",
                "best_epoch",
                "train_dice_at_best",
                "epochs_ran",
                "history_path",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    with open(epoch_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "num_folds",
                "train_loss_mean",
                "train_loss_std",
                "train_dice_mean",
                "train_dice_std",
                "train_surface_dice_mean",
                "train_surface_dice_std",
                "train_lesion_recall_mean",
                "train_lesion_recall_std",
                "val_dice_mean",
                "val_dice_std",
                "val_surface_dice_mean",
                "val_surface_dice_std",
                "val_lesion_recall_mean",
                "val_lesion_recall_std",
            ],
        )
        writer.writeheader()
        writer.writerows(epoch_aggregate)

    print("CV fold results:")
    for row in results:
        print(
            f"  fold {row['fold']}: "
            f"best_val_dice={row['best_val_dice']:.4f} "
            f"(epoch={row['best_epoch']}), "
            f"train_dice_at_best={row['train_dice_at_best']:.4f}, "
            f"val_patients={row['val_patients']}, "
            f"epochs_ran={row['epochs_ran']}"
        )
    print(
        "CV aggregate: "
        f"mean_best_val_dice={mean_best:.4f} +- {std_best:.4f}, "
        f"mean_train_dice_at_best={mean_train_dice_at_best:.4f} +- {std_train_dice_at_best:.4f}"
    )
    print(f"Saved CV summary: {summary_path}")
    print(f"Saved fold results CSV: {fold_csv_path}")
    print(f"Saved epoch aggregate CSV: {epoch_csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.cfg, folds=args.folds)
