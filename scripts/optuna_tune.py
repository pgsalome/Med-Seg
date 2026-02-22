import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import tempfile

import optuna
import torch
import yaml
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medseg.config import load_config
from medseg.registry_build import ensure_training_registry
try:
    from scripts.train_cv import main as train_cv_main
except ModuleNotFoundError:
    from train_cv import main as train_cv_main


def objective(trial: optuna.Trial, base_cfg: dict, folds: int):
    cfg = copy.deepcopy(base_cfg)
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_gpu not in (None, ""):
        cfg["device"] = "cuda"
        trial.set_user_attr("cuda_visible_devices", str(visible_gpu))

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

    patch_z = trial.suggest_categorical("patch_z", [80, 96, 112])
    patch_xy = trial.suggest_categorical("patch_xy", [144, 160, 176])
    cfg["sampling"]["patch_size"][0] = patch_z
    cfg["sampling"]["patch_size"][1] = patch_xy
    cfg["sampling"]["patch_size"][2] = patch_xy

    loss_cfg = cfg["train"].get("loss")
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}
        cfg["train"]["loss"] = loss_cfg
    loss_name = trial.suggest_categorical("loss", ["dicece", "dicefocal"])
    cfg["train"]["loss"]["name"] = loss_name

    cfg.setdefault("train", {})
    cfg["train"].setdefault("oversampling", {})
    cfg["train"]["oversampling"]["epoch_multiplier"] = trial.suggest_int("oversample_mult", 3, 6)
    cfg["train"]["oversampling"]["cap"] = trial.suggest_float("oversample_cap", 5.0, 12.0)

    cfg["train"].setdefault("deep_supervision", {})
    cfg["train"]["deep_supervision"]["enabled"] = trial.suggest_categorical(
        "deep_supervision", [True, False]
    )

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
    cfg["augment"]["spatial"]["rotate_range_deg"] = trial.suggest_int("rotate_deg", 8, 20)
    cfg["augment"]["spatial"]["elastic_prob"] = trial.suggest_float("elastic_prob", 0.08, 0.25)
    cfg["augment"]["spatial"]["affine_prob"] = trial.suggest_float("affine_prob", 0.25, 0.50)

    cfg["train"]["epochs"] = trial_budget_epochs
    cfg["train"]["early_stop_patience"] = 15

    gpu_suffix = f"-gpu{visible_gpu}" if visible_gpu not in (None, "") else ""
    cfg["wandb"]["run_name"] = f"optuna-trial-{trial.number}{gpu_suffix}"
    cfg["wandb"]["tags"] = cfg["wandb"].get("tags", []) + ["optuna"]

    base_checkpoint_dir = cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    trial_checkpoint_dir = os.path.join(
        base_checkpoint_dir,
        "optuna_trials",
        f"trial_{trial.number:05d}",
    )
    cfg.setdefault("output", {})
    cfg["output"]["checkpoint_dir"] = trial_checkpoint_dir
    summary_path = os.path.join(trial_checkpoint_dir, "cv_summary.json")
    trial.set_user_attr("checkpoint_dir", trial_checkpoint_dir)

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


def _resolve_storage_url(base_cfg: dict) -> str:
    optuna_cfg = base_cfg.get("optuna", {})
    storage = optuna_cfg.get("storage")
    if storage not in (None, ""):
        return str(storage)

    checkpoint_dir = base_cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    db_path = os.path.abspath(os.path.join(checkpoint_dir, "optuna_study.db"))
    return f"sqlite:///{db_path}"


def _resolve_gpu_ids(base_cfg: dict):
    optuna_cfg = base_cfg.get("optuna", {})
    requested = int(optuna_cfg.get("num_gpus", 1))

    available = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if available <= 0:
        return []

    raw_ids = optuna_cfg.get("gpu_ids")
    if isinstance(raw_ids, str) and raw_ids.strip():
        gpu_ids = [int(part.strip()) for part in raw_ids.split(",") if part.strip()]
    elif isinstance(raw_ids, (list, tuple)) and len(raw_ids) > 0:
        gpu_ids = [int(v) for v in raw_ids]
    else:
        gpu_ids = list(range(available))

    gpu_ids = [gid for gid in gpu_ids if 0 <= gid < available]
    if not gpu_ids:
        return []
    return gpu_ids[: max(1, min(requested, len(gpu_ids)))]


def _worker_optimize(
    worker_idx: int,
    gpu_id,
    storage_url: str,
    study_name: str,
    base_cfg: dict,
    folds: int,
    total_trials: int,
):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    max_trials_cb = MaxTrialsCallback(
        int(total_trials),
        states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
    )
    study.optimize(
        lambda trial: objective(trial, base_cfg, folds=int(folds)),
        n_trials=None,
        catch=(RuntimeError, ValueError),
        callbacks=[max_trials_cb],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=None, help="Pin this worker to one GPU index.")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (overrides cfg.optuna.storage).")
    parser.add_argument("--study-name", default=None, help="Optuna study name (overrides cfg.optuna.study_name).")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override cfg.optuna.num_gpus for auto multi-worker mode.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.cfg)
    base_cfg.setdefault("optuna", {})
    if args.storage not in (None, ""):
        base_cfg["optuna"]["storage"] = str(args.storage)
    if args.study_name not in (None, ""):
        base_cfg["optuna"]["study_name"] = str(args.study_name)
    if args.num_gpus is not None:
        base_cfg["optuna"]["num_gpus"] = int(args.num_gpus)

    base_cfg = ensure_training_registry(base_cfg, args.cfg)
    base_cfg.setdefault("registry_local", {})
    if base_cfg["registry_local"].get("auto_build", False):
        base_cfg["registry_local"]["rebuild_each_run"] = False

    seed = int(base_cfg.get("seed", 1337))
    sampler = optuna.samplers.TPESampler(seed=seed)
    storage_url = _resolve_storage_url(base_cfg)
    study_name = str(
        base_cfg.get("optuna", {}).get(
            "study_name",
            f"medseg_optuna_{os.path.splitext(os.path.basename(args.cfg))[0]}",
        )
    )
    optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    max_trials_cb = MaxTrialsCallback(
        int(args.trials),
        states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
    )

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
        print(f"[optuna] single worker pinned to GPU {int(args.gpu)}")
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        study.optimize(
            lambda trial: objective(trial, base_cfg, folds=int(args.folds)),
            n_trials=None,
            catch=(RuntimeError, ValueError),
            callbacks=[max_trials_cb],
        )
    else:
        gpu_ids = _resolve_gpu_ids(base_cfg)
        if gpu_ids and len(gpu_ids) > 1:
            print(f"[optuna] launching {len(gpu_ids)} workers on GPUs: {gpu_ids}")
            ctx = mp.get_context("spawn")
            procs = []
            for idx, gpu_id in enumerate(gpu_ids):
                p = ctx.Process(
                    target=_worker_optimize,
                    args=(
                        int(idx),
                        int(gpu_id),
                        storage_url,
                        study_name,
                        base_cfg,
                        int(args.folds),
                        int(args.trials),
                    ),
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            bad = [p.pid for p in procs if p.exitcode != 0]
            if bad:
                raise RuntimeError(f"Optuna worker(s) failed: {bad}")
        else:
            if gpu_ids:
                print(f"[optuna] single worker on GPU {gpu_ids[0]}")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_ids[0]))
            else:
                print("[optuna] no CUDA GPUs configured/available; running single-worker.")
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            study.optimize(
                lambda trial: objective(trial, base_cfg, folds=int(args.folds)),
                n_trials=None,
                catch=(RuntimeError, ValueError),
                callbacks=[max_trials_cb],
            )

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    completed = [
        t
        for t in study.trials
        if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    ]
    print(f"[optuna] completed/pruned/failed trials: {len(completed)}")
    if len(completed) < int(args.trials):
        print(f"[optuna][warning] requested {args.trials} trials, got {len(completed)}.")

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    if study.best_trial is not None:
        print(
            "Best effective sampling:",
            {
                "p_fg": study.best_trial.params.get("p_fg"),
                "p_hardneg": study.best_trial.user_attrs.get("effective_p_hardneg"),
                "p_bg": study.best_trial.user_attrs.get("effective_p_bg"),
                "cuda_visible_devices": study.best_trial.user_attrs.get("cuda_visible_devices"),
            },
        )

    print(
        "[optuna] study:",
        {
            "name": study_name,
            "storage": storage_url,
        },
    )


if __name__ == "__main__":
    main()
