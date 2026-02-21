import argparse
import copy
import json
import os
import sys
import tempfile

import optuna
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medseg.config import load_config
try:
    from scripts.train_cv import main as train_cv_main
except ModuleNotFoundError:
    from train_cv import main as train_cv_main


def objective(trial: optuna.Trial, base_cfg: dict, folds: int):
    cfg = copy.deepcopy(base_cfg)

    trial_budget_epochs = trial.suggest_int("trial_epochs", 80, 120)
    min_bg = 0.10
    cfg.setdefault("metrics", {})
    cfg["metrics"]["val_thresholds"] = [0.3, 0.4, 0.5, 0.6]
    cfg.setdefault("validation", {})
    cfg["validation"]["sw_overlap"] = 0.5
    # Tightened search space based on recent sweeps:
    # better runs favored higher fg sampling, lower bg,
    # moderate-high lr, and lower weight decay.

    cfg["train"]["lr"] = trial.suggest_float("lr", 8e-5, 3.5e-4, log=True)
    cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 3e-6, 1e-4, log=True)
    cfg["transfer"]["encoder_lr_mult"] = trial.suggest_float(
        "encoder_lr_mult", 0.08, 0.25, log=True
    )
    cfg["transfer"]["freeze_encoder_epochs"] = trial.suggest_int(
        "freeze_encoder_epochs", 5, 10
    )

    p_fg = trial.suggest_float("p_fg", 0.70, 0.82)
    p_hardneg_frac = trial.suggest_float("p_hardneg_frac", 0.2, 0.8)
    p_hardneg_min = 0.06
    p_hardneg_max = max(p_hardneg_min, (1.0 - min_bg) - p_fg)
    p_hardneg = p_hardneg_min + p_hardneg_frac * (p_hardneg_max - p_hardneg_min)
    p_bg = 1.0 - p_fg - p_hardneg
    if p_bg < min_bg:
        p_bg = min_bg
        p_hardneg = max(0.0, 1.0 - p_fg - p_bg)

    cfg["sampling"]["p_fg"] = float(p_fg)
    cfg["sampling"]["p_hardneg"] = float(p_hardneg)
    cfg["sampling"]["p_bg"] = float(p_bg)
    trial.set_user_attr("effective_p_hardneg", float(p_hardneg))
    trial.set_user_attr("effective_p_bg", float(p_bg))

    patch_z = trial.suggest_categorical("patch_z", [64, 80, 96, 112])
    patch_xy = trial.suggest_categorical("patch_xy", [128, 144, 160])
    cfg["sampling"]["patch_size"][0] = patch_z
    cfg["sampling"]["patch_size"][1] = patch_xy
    cfg["sampling"]["patch_size"][2] = patch_xy

    loss_cfg = cfg["train"].get("loss")
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}
        cfg["train"]["loss"] = loss_cfg
    cfg["train"]["loss"]["name"] = "dicefocal"

    cfg["augment"]["intensity"]["gamma_prob"] = trial.suggest_float("gamma_prob", 0.25, 0.50)
    cfg["augment"]["intensity"]["noise_prob"] = trial.suggest_float("noise_prob", 0.10, 0.30)
    cfg["augment"]["intensity"]["bias_field_prob"] = trial.suggest_float(
        "bias_field_prob", 0.25, 0.55
    )
    cfg["augment"]["intensity"]["coarse_dropout_prob"] = trial.suggest_float(
        "coarse_dropout_prob", 0.20, 0.40
    )
    cfg["augment"]["domain"]["thick_slice_sim_prob"] = trial.suggest_float(
        "thick_slice_sim_prob", 0.15, 0.35
    )

    cfg["train"]["epochs"] = trial_budget_epochs
    cfg["train"]["early_stop_patience"] = 12

    cfg["wandb"]["run_name"] = f"optuna-trial-{trial.number}"
    cfg["wandb"]["tags"] = cfg["wandb"].get("tags", []) + ["optuna"]

    checkpoint_dir = cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    summary_path = os.path.join(checkpoint_dir, "cv_summary.json")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_cfg_path = tmp.name

    try:
        train_cv_main(tmp_cfg_path, folds=int(folds))
    finally:
        if os.path.exists(tmp_cfg_path):
            os.unlink(tmp_cfg_path)

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        best = float(summary.get("mean_best_val_dice", 0.0))
        trial.set_user_attr("cv_std_best_val_dice", float(summary.get("std_best_val_dice", 0.0)))
        return best
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    base_cfg = load_config(args.cfg)
    seed = int(base_cfg.get("seed", 1337))
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, base_cfg, folds=int(args.folds)),
        n_trials=args.trials,
        catch=(RuntimeError, ValueError),
    )

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    if study.best_trial is not None:
        print(
            "Best effective sampling:",
            {
                "p_fg": study.best_trial.params.get("p_fg"),
                "p_hardneg": study.best_trial.user_attrs.get("effective_p_hardneg"),
                "p_bg": study.best_trial.user_attrs.get("effective_p_bg"),
            },
        )


if __name__ == "__main__":
    main()
