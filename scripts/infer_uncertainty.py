import json
import os
import sys
from typing import List, Optional

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

from medseg.config import load_config
from medseg.dataset import case_cache_key, load_npz, save_npz
from medseg.io_registry import load_registry
from medseg.preprocess import load_and_preprocess_with_steps
from medseg.registry_build import ensure_training_registry
from medseg.tta import build_tta_fns, load_tta_preset
from medseg.uncertainty import (
    UncertaintyResult,
    predict_ensemble,
    predict_mc_dropout,
    predict_tta,
    summarize_component_uncertainty,
)
from medseg.wandb_utils import wandb_log_mid_slices, wandb_log_scalars
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


def _init_model_for_infer(cfg: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
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
        # Allow direct backbone checkpoints by loading the encoder only.
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
        return image, label, spacing, cache_key

    image, label, spacing, _, brain = load_and_preprocess_with_steps(case.image_path, case.mask_path, cfg)
    save_npz(npz_path, image, label, spacing, brain)
    return image, label, spacing, cache_key


def _select_cases(cases, uid: Optional[str], uids_file: Optional[str]):
    if uid:
        wanted = {uid}
    elif uids_file:
        with open(uids_file, "r", encoding="utf-8") as file:
            wanted = {line.strip() for line in file if line.strip()}
    else:
        return cases
    return [case for case in cases if case.uid in wanted]


def _predict(
    method: str,
    x: torch.Tensor,
    model: Optional[torch.nn.Module],
    models: Optional[List[torch.nn.Module]],
    tta_fns,
    mc_n: int,
    amp: bool,
) -> UncertaintyResult:
    if method == "tta":
        return predict_tta(model=model, x=x, tta_fns=tta_fns, amp=amp, return_samples=True)
    if method == "mc_dropout":
        return predict_mc_dropout(model=model, x=x, n=mc_n, amp=amp, return_samples=True)
    if method == "ensemble":
        return predict_ensemble(models=models, x=x, amp=amp, return_samples=True)
    raise ValueError(f"Unsupported method: {method}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--method", choices=["tta", "mc_dropout", "ensemble"], default=None)
    parser.add_argument("--tta_preset", default=None)
    parser.add_argument("--mc_n", type=int, default=None)
    parser.add_argument("--ensemble_ckpts", nargs="+", default=None)
    parser.add_argument("--uids_file", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--out_dir", default="outputs_uncertainty")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = ensure_training_registry(cfg, args.cfg)

    method = args.method or cfg.get("uncertainty", {}).get("method", "tta")
    tta_preset = args.tta_preset or cfg.get("uncertainty", {}).get("tta_preset", "presets/brain_tta.yaml")
    mc_n = int(args.mc_n if args.mc_n is not None else cfg.get("uncertainty", {}).get("mc_n", 20))
    amp = bool(args.amp or cfg["train"].get("amp", True))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = None
    models = None
    if method in {"tta", "mc_dropout"}:
        if not args.ckpt:
            raise ValueError("--ckpt is required for method tta or mc_dropout.")
        ckpt_path = _resolve_path(args.ckpt)
        model = _init_model_for_infer(cfg, ckpt_path, device)
    elif method == "ensemble":
        if not args.ensemble_ckpts:
            raise ValueError("--ensemble_ckpts is required for method ensemble.")
        models = []
        for ckpt in args.ensemble_ckpts:
            models.append(_init_model_for_infer(cfg, _resolve_path(ckpt), device))

    tta_fns = None
    if method == "tta":
        tta_specs = load_tta_preset(_resolve_path(tta_preset))
        tta_fns = build_tta_fns(tta_specs)

    if args.wandb:
        run_name = cfg["wandb"].get("run_name", "infer-unc")
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            name=f"{run_name}-infer-{method}",
            tags=cfg["wandb"].get("tags", []) + ["uncertainty", method],
            mode=cfg["wandb"].get("mode", "online"),
            config=cfg,
        )

    cases = _select_cases(load_registry(cfg), uid=args.uid, uids_file=args.uids_file)
    print(f"Cases selected: {len(cases)}")

    out_dir = _resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for step, case in enumerate(tqdm(cases)):
        image, label, spacing, _cache_key = _load_or_cache_case(case, cfg)
        x = torch.from_numpy(image[None]).to(device)

        result = _predict(
            method=method,
            x=x,
            model=model,
            models=models,
            tta_fns=tta_fns,
            mc_n=mc_n,
            amp=amp,
        )

        case_dir = os.path.join(out_dir, case.uid.replace("/", "_"))
        os.makedirs(case_dir, exist_ok=True)

        prob_mean = result.prob_mean.detach().cpu().numpy()
        unc_entropy = result.entropy.detach().cpu().numpy()
        unc_var = result.var.detach().cpu().numpy()

        np.save(os.path.join(case_dir, "prob_mean.npy"), prob_mean)
        np.save(os.path.join(case_dir, "unc_entropy.npy"), unc_entropy)
        np.save(os.path.join(case_dir, "unc_var.npy"), unc_var)
        if result.mutual_info is not None:
            np.save(os.path.join(case_dir, "unc_mi.npy"), result.mutual_info.detach().cpu().numpy())
        if result.prob_samples is not None:
            np.save(os.path.join(case_dir, "prob_samples.npy"), result.prob_samples.detach().cpu().numpy())

        comps = summarize_component_uncertainty(
            prob_mean=result.prob_mean,
            unc_map=result.entropy,
            threshold=0.5,
            min_voxels=10,
        )
        with open(os.path.join(case_dir, "components.json"), "w", encoding="utf-8") as file:
            json.dump(comps, file, indent=2)

        scalar_log = {
            "unc/prob_mean_mean": float(prob_mean.mean()),
            "unc/entropy_mean": float(unc_entropy.mean()),
            "unc/var_mean": float(unc_var.mean()),
            "unc/num_cases_done": float(step + 1),
            "unc/num_components": float(len(comps)),
        }
        pred_mask = prob_mean[0] >= 0.5
        scalar_log["unc/max_unc_entropy"] = float(unc_entropy[0].max())
        scalar_log["unc/mean_unc_entropy_pred"] = (
            float(unc_entropy[0][pred_mask].mean()) if pred_mask.any() else 0.0
        )
        if result.mutual_info is not None:
            scalar_log["unc/mi_mean"] = float(result.mutual_info.detach().cpu().mean().item())

        if args.wandb:
            wandb_log_mid_slices(
                prefix=f"unc/{case.uid}",
                image_zyx=image[0],
                prob_mean_zyx=prob_mean[0],
                unc_zyx=unc_entropy[0],
                gt_zyx=label[0] if label is not None else None,
                step=step,
            )
            wandb_log_scalars(scalar_log, step=step)

            if comps:
                rows = []
                for comp in comps[:10]:
                    rows.append(
                        [
                            comp["component_id"],
                            comp["voxels"],
                            comp["mean_prob"],
                            comp["mean_unc"],
                            comp["max_unc"],
                        ]
                    )
                wandb.log(
                    {
                        "unc/components": wandb.Table(
                            data=rows,
                            columns=["id", "voxels", "mean_prob", "mean_unc", "max_unc"],
                        )
                    },
                    step=step,
                )

        summary.append(
            {
                "uid": case.uid,
                "patient_id": case.patient_id,
                "spacing": list(spacing),
                "entropy_mean": float(unc_entropy.mean()),
                "var_mean": float(unc_var.mean()),
                "num_components": int(len(comps)),
                "mi_mean": float(result.mutual_info.detach().cpu().mean().item())
                if result.mutual_info is not None
                else None,
            }
        )

    with open(os.path.join(out_dir, "uncertainty_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
