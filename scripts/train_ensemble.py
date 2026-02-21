import os
import shutil
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from scripts.train import main as train_main
except ModuleNotFoundError:
    from train import main as train_main


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    from medseg.config import load_config

    base_cfg = load_config(args.cfg)
    base_seed = int(base_cfg.get("seed", 1337))

    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    else:
        seeds = [base_seed + i for i in range(int(args.k))]

    out_root = os.path.join("checkpoints", "ensemble")
    os.makedirs(out_root, exist_ok=True)

    model_name = base_cfg["model"]["name"]
    run_name_base = base_cfg.get("wandb", {}).get("run_name", "ensemble-train")

    for i, seed in enumerate(seeds):
        cfg = load_config(args.cfg)
        cfg["seed"] = int(seed)
        cfg.setdefault("wandb", {})
        cfg["wandb"]["run_name"] = f"{run_name_base}-seed-{seed}"
        cfg["wandb"]["tags"] = cfg["wandb"].get("tags", []) + ["ensemble", f"seed-{seed}"]

        run_ckpt_dir = os.path.join(out_root, "runs", f"seed_{seed}")
        cfg.setdefault("output", {})
        cfg["output"]["checkpoint_dir"] = run_ckpt_dir

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp_cfg = tmp.name

        print(f"[ensemble] ({i+1}/{len(seeds)}) training seed={seed}")
        train_main(tmp_cfg)

        best_ckpt = os.path.join(run_ckpt_dir, f"best_{model_name}.pth")
        best_metric = os.path.join(run_ckpt_dir, f"best_metric_{model_name}.json")
        final_ckpt = os.path.join(out_root, f"seed_{seed}.pth")
        final_ckpt_compat = os.path.join(out_root, f"{model_name}_seed_{seed}.pth")
        final_metric = os.path.join(out_root, f"seed_{seed}.json")

        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"Expected checkpoint not found: {best_ckpt}")

        shutil.copy2(best_ckpt, final_ckpt)
        shutil.copy2(best_ckpt, final_ckpt_compat)
        if os.path.exists(best_metric):
            shutil.copy2(best_metric, final_metric)

        print(f"[ensemble] saved: {final_ckpt}")
        if os.path.exists(tmp_cfg):
            os.unlink(tmp_cfg)


if __name__ == "__main__":
    main()
