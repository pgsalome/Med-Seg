import json
import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medseg.calibration import (
    CalibrationResult,
    reliability_diagram_figure,
    uncertainty_error_curve,
    voxelwise_ece,
)
from medseg.config import load_config
from medseg.dataset import case_cache_key, load_npz, save_npz
from medseg.io_registry import load_registry
from medseg.preprocess import load_and_preprocess_with_steps
from medseg.registry_build import ensure_training_registry
from medseg.tta import build_tta_fns, load_tta_preset
from medseg.uncertainty import compute_entropy, predict_ensemble, predict_mc_dropout, predict_single, predict_tta
from medseg.wandb_utils import wandb_log_figure, wandb_log_scalars
try:
    from scripts.train import _load_state_dict, build_model
except ModuleNotFoundError:
    from train import _load_state_dict, build_model


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(ROOT, path)


def _init_model_for_eval(cfg: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            int(cfg["model"]["in_channels"]),
            *[int(v) for v in cfg["sampling"]["patch_size"]],
            device=device,
        )
        _ = model(dummy)
    state = _load_state_dict(ckpt_path)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        if hasattr(model, "encoder"):
            model.encoder.load_state_dict(state, strict=True)
        else:
            raise
    model.eval()
    return model


def _load_or_cache_case(case, cfg: dict):
    cache_dir = cfg["cache"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = case_cache_key(case, cfg)
    npz_path = os.path.join(cache_dir, f"{cache_key}.npz")
    if os.path.exists(npz_path) and not cfg["cache"]["overwrite"]:
        image, label, spacing, _brain = load_npz(npz_path)
        return image, label, spacing

    image, label, spacing, _, brain = load_and_preprocess_with_steps(case.image_path, case.mask_path, cfg)
    save_npz(npz_path, image, label, spacing, brain)
    return image, label, spacing


def _run_prediction(
    method: str,
    x: torch.Tensor,
    model: Optional[torch.nn.Module],
    models: Optional[List[torch.nn.Module]],
    tta_fns,
    mc_n: int,
    amp: bool,
):
    if method == "none":
        p = predict_single(model, x, amp=amp)
        p = p[0] if p.shape[0] == 1 else p
        ent = compute_entropy(p)
        return {"prob": p, "unc": ent}
    if method == "tta":
        out = predict_tta(model=model, x=x, tta_fns=tta_fns, amp=amp, return_samples=False)
        return {"prob": out.prob_mean, "unc": out.entropy}
    if method == "mc_dropout":
        out = predict_mc_dropout(model=model, x=x, n=mc_n, amp=amp, return_samples=False)
        return {"prob": out.prob_mean, "unc": out.entropy}
    if method == "ensemble":
        out = predict_ensemble(models=models, x=x, amp=amp, return_samples=False)
        return {"prob": out.prob_mean, "unc": out.entropy}
    raise ValueError(method)


def _aggregate_calibration(cal_items: List[CalibrationResult]) -> CalibrationResult:
    n_bins = len(cal_items[0].bin_count)
    count = np.zeros(n_bins, dtype=np.int64)
    acc_sum = np.zeros(n_bins, dtype=np.float64)
    conf_sum = np.zeros(n_bins, dtype=np.float64)
    for cal in cal_items:
        count += cal.bin_count
        acc_sum += cal.bin_acc * cal.bin_count
        conf_sum += cal.bin_conf * cal.bin_count

    bin_acc = np.zeros(n_bins, dtype=np.float32)
    bin_conf = np.zeros(n_bins, dtype=np.float32)
    nz = count > 0
    bin_acc[nz] = (acc_sum[nz] / count[nz]).astype(np.float32)
    bin_conf[nz] = (conf_sum[nz] / count[nz]).astype(np.float32)

    total = max(1, int(count.sum()))
    gap = np.abs(bin_acc - bin_conf)
    ece = float(((count / float(total)) * gap).sum())
    mce = float(gap[nz].max()) if nz.any() else 0.0
    return CalibrationResult(ece=ece, mce=mce, bin_acc=bin_acc, bin_conf=bin_conf, bin_count=count)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--method", choices=["none", "tta", "mc_dropout", "ensemble"], default="none")
    parser.add_argument("--tta_preset", default=None)
    parser.add_argument("--mc_n", type=int, default=None)
    parser.add_argument("--ensemble_ckpts", nargs="+", default=None)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--out_dir", default="outputs/calibration")
    parser.add_argument("--max_curve_voxels", type=int, default=20000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = ensure_training_registry(cfg, args.cfg)

    method = args.method
    tta_preset = args.tta_preset or cfg.get("uncertainty", {}).get("tta_preset", "presets/brain_tta.yaml")
    mc_n = int(args.mc_n if args.mc_n is not None else cfg.get("uncertainty", {}).get("mc_n", 20))
    amp = bool(args.amp or cfg["train"].get("amp", True))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = None
    models = None
    if method in {"none", "tta", "mc_dropout"}:
        if not args.ckpt:
            raise ValueError("--ckpt is required unless method=ensemble.")
        model = _init_model_for_eval(cfg, _resolve_path(args.ckpt), device)
    if method == "ensemble":
        if not args.ensemble_ckpts:
            raise ValueError("--ensemble_ckpts is required for method=ensemble.")
        models = [_init_model_for_eval(cfg, _resolve_path(path), device) for path in args.ensemble_ckpts]

    tta_fns = None
    if method == "tta":
        tta_specs = load_tta_preset(_resolve_path(tta_preset))
        tta_fns = build_tta_fns(tta_specs)

    if args.wandb:
        run_name = cfg["wandb"].get("run_name", "calibrate")
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            name=f"{run_name}-calibration-{method}",
            tags=cfg["wandb"].get("tags", []) + ["calibration", method],
            mode=cfg["wandb"].get("mode", "online"),
            config=cfg,
        )

    cases = load_registry(cfg)
    print(f"Cases: {len(cases)}")

    out_dir = _resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cal_items: List[CalibrationResult] = []
    case_ece: List[float] = []
    case_mce: List[float] = []
    unc_samples = []
    err_samples = []

    for step, case in enumerate(tqdm(cases)):
        image, label, _ = _load_or_cache_case(case, cfg)
        x = torch.from_numpy(image[None]).to(device)
        target = torch.from_numpy(label).float()

        pred = _run_prediction(
            method=method,
            x=x,
            model=model,
            models=models,
            tta_fns=tta_fns,
            mc_n=mc_n,
            amp=amp,
        )
        prob = pred["prob"].detach().cpu()  # (1,Z,Y,X)
        unc = pred["unc"].detach().cpu()  # (1,Z,Y,X)

        cal = voxelwise_ece(prob=prob, target=target, n_bins=args.n_bins, mask=None)
        cal_items.append(cal)
        case_ece.append(cal.ece)
        case_mce.append(cal.mce)

        if args.wandb:
            wandb_log_scalars(
                {"calibration/ece_case": cal.ece, "calibration/mce_case": cal.mce},
                step=step,
            )

        if method != "none":
            pred_bin = (prob > 0.5).float()
            error = (pred_bin != target).float()
            unc_flat = unc.view(-1).numpy()
            err_flat = error.view(-1).numpy()
            if len(unc_flat) > args.max_curve_voxels:
                sel = np.random.choice(len(unc_flat), size=args.max_curve_voxels, replace=False)
                unc_flat = unc_flat[sel]
                err_flat = err_flat[sel]
            unc_samples.append(unc_flat.astype(np.float32))
            err_samples.append(err_flat.astype(np.float32))

    agg = _aggregate_calibration(cal_items)
    rel_fig = reliability_diagram_figure(agg, title=f"{method} reliability")
    if args.wandb:
        wandb_log_figure("calibration/reliability", rel_fig)
    else:
        plt.close(rel_fig)

    summary = {
        "method": method,
        "ece_global": float(agg.ece),
        "mce_global": float(agg.mce),
        "ece_case_mean": float(np.mean(case_ece)) if case_ece else 0.0,
        "mce_case_mean": float(np.mean(case_mce)) if case_mce else 0.0,
        "n_cases": len(cases),
    }

    if method != "none" and unc_samples:
        unc_cat = torch.from_numpy(np.concatenate(unc_samples, axis=0))
        err_cat = torch.from_numpy(np.concatenate(err_samples, axis=0))
        xs, ys = uncertainty_error_curve(unc=unc_cat, error=err_cat, n_bins=20)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Mean Error")
        ax.set_title("Error vs Uncertainty")
        fig.tight_layout()
        if args.wandb:
            wandb_log_figure("calibration/error_vs_uncertainty", fig)
        else:
            plt.close(fig)

        valid = np.isfinite(ys)
        if np.any(valid):
            corr = np.corrcoef(xs[valid], ys[valid])[0, 1]
            summary["unc_error_corr"] = float(corr)

    if args.wandb:
        wandb_log_scalars(summary)

    with open(os.path.join(out_dir, "calibration_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
