import json
import os
import sys
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Optional

import numpy as np
import wandb
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from medseg.config import load_config
from medseg.dataset import case_cache_key, save_npz
from medseg.io_registry import load_registry
from medseg.preprocess import load_and_preprocess_with_steps
from medseg.qa_viz import save_mid_and_centroid
from medseg.registry_build import ensure_training_registry


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "NA"


def _build_meta_fields(
    uid: str,
    patient_id: str,
    site: str,
    image_path: str,
    mask_path: str,
    cache_key: str,
    image: np.ndarray,
    label: np.ndarray,
    spacing,
) -> dict:
    return {
        "uid": uid,
        "patient_id": patient_id,
        "site": site,
        "image_path": image_path,
        "mask_path": mask_path,
        "cache_key": cache_key,
        "spacing": list(spacing),
        "final_shape_czyx": list(image.shape),
        "lesion_voxels": int(label.sum()),
        "img_mean": float(image.mean()),
        "img_std": float(image.std()),
        "img_min": float(image.min()),
        "img_max": float(image.max()),
    }


def _save_final_qa_from_cache(
    image: np.ndarray,
    label: np.ndarray,
    spacing,
    qa_dir: str,
    uid_tag: str,
    patient_id: str,
    bg_value: float,
) -> str:
    if image.ndim != 4 or label.ndim != 4:
        raise ValueError(
            f"Expected cached arrays with shape CZYX (4D), got image={image.shape} label={label.shape}"
        )

    spacing_zyx = None
    if spacing is not None and len(spacing) == 3:
        sx, sy, sz = [float(v) for v in spacing]
        spacing_zyx = (sz, sy, sx)

    prefix = f"{uid_tag}__04_norm"
    save_mid_and_centroid(
        image[0],
        label[0],
        qa_dir,
        prefix=prefix,
        title=f"{patient_id} 04_norm",
        spacing_zyx=spacing_zyx,
        background_value=float(bg_value),
        exclude_background_from_step=4,
    )
    return os.path.join(qa_dir, f"{prefix}_centroid_overlay.png")


def _process_case_payload(payload: dict) -> dict:
    cfg = payload["cfg"]
    uid = payload["uid"]
    patient_id = payload["patient_id"]
    site = payload["site"]
    image_path = payload["image_path"]
    mask_path = payload["mask_path"]
    cache_dir = payload["cache_dir"]
    bg_value = float(payload["bg_value"])

    case_key_obj = SimpleNamespace(uid=uid, image_path=image_path, mask_path=mask_path)
    cache_key = case_cache_key(case_key_obj, cfg)
    npz_path = os.path.join(cache_dir, f"{cache_key}.npz")
    patient_tag = _safe_name(patient_id)
    uid_tag = _safe_name(uid)
    qa_dir = os.path.join(cache_dir, "qa", patient_tag)
    os.makedirs(qa_dir, exist_ok=True)
    meta_path = os.path.join(qa_dir, f"{uid_tag}__meta.json")
    final_norm_png = os.path.join(qa_dir, f"{uid_tag}__04_norm_centroid_overlay.png")

    result = {
        "uid": uid,
        "patient_id": patient_id,
        "site": site,
        "status": "ok",
        "mode": "none",
        "meta": None,
        "qa_paths": {},
    }

    try:
        if os.path.exists(npz_path) and not cfg["cache"]["overwrite"]:
            has_brain_mask = False
            try:
                with np.load(npz_path, allow_pickle=False) as cached:
                    has_brain_mask = "brain_mask" in cached.files
            except Exception:
                has_brain_mask = False

            if not has_brain_mask:
                result["warning"] = (
                    f"Cached NPZ for {uid} is missing brain_mask; rebuilding cache entry."
                )
            elif os.path.exists(meta_path) and os.path.exists(final_norm_png):
                result["status"] = "skipped"
                result["mode"] = "already_cached"
                return result

            if has_brain_mask:
                with np.load(npz_path, allow_pickle=False) as data:
                    image = data["image"]
                    label = data["label"]
                    spacing = tuple(data["spacing"].tolist())

                final_norm_png = _save_final_qa_from_cache(
                    image=image,
                    label=label,
                    spacing=spacing,
                    qa_dir=qa_dir,
                    uid_tag=uid_tag,
                    patient_id=patient_id,
                    bg_value=bg_value,
                )
                meta = _build_meta_fields(
                    uid=uid,
                    patient_id=patient_id,
                    site=site,
                    image_path=image_path,
                    mask_path=mask_path,
                    cache_key=cache_key,
                    image=image,
                    label=label,
                    spacing=spacing,
                )
                with open(meta_path, "w", encoding="utf-8") as file:
                    json.dump(meta, file, indent=2)

                result["mode"] = "from_npz"
                result["meta"] = meta
                if os.path.exists(final_norm_png):
                    result["qa_paths"]["04_norm_qa"] = final_norm_png
                return result

        image, label, spacing, steps, brain_mask = load_and_preprocess_with_steps(
            image_path,
            mask_path,
            cfg,
        )
        save_npz(npz_path, image, label, spacing, brain_mask)

        qa_paths = {}
        for step in ["00_raw", "01_resampled", "02_n4", "03_crop", "04_norm"]:
            step_payload = steps[step]
            vol = step_payload["img"]
            msk = step_payload["msk"]
            spacing_zyx = step_payload.get("spacing_zyx")
            prefix = f"{uid_tag}__{step}"
            save_mid_and_centroid(
                vol,
                msk,
                qa_dir,
                prefix=prefix,
                title=f"{patient_id} {step}",
                spacing_zyx=spacing_zyx,
                background_value=bg_value,
                exclude_background_from_step=4,
            )
            qa_paths[f"{step}_qa"] = os.path.join(qa_dir, f"{prefix}_centroid_overlay.png")

        meta = _build_meta_fields(
            uid=uid,
            patient_id=patient_id,
            site=site,
            image_path=image_path,
            mask_path=mask_path,
            cache_key=cache_key,
            image=image,
            label=label,
            spacing=spacing,
        )
        with open(meta_path, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)

        result["mode"] = "full_preprocess"
        result["meta"] = meta
        result["qa_paths"] = qa_paths
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


def _log_case_result(result: dict) -> None:
    if result.get("status") != "ok":
        return
    meta = result.get("meta") or {}
    log_data = {
        "case/patient_id": result.get("patient_id"),
        "case/site": result.get("site"),
        "case/lesion_voxels": meta.get("lesion_voxels", 0),
        "case/img_mean": meta.get("img_mean", 0.0),
        "case/img_std": meta.get("img_std", 0.0),
        "case/img_min": meta.get("img_min", 0.0),
        "case/img_max": meta.get("img_max", 0.0),
    }
    for name, path in (result.get("qa_paths") or {}).items():
        if os.path.exists(path):
            log_data[f"qa/{name}"] = wandb.Image(path)
    wandb.log(log_data)


def main(cfg_path: str, workers: Optional[int] = None):
    cfg = load_config(cfg_path)
    cfg = ensure_training_registry(cfg, cfg_path)

    cache_dir = cfg["cache"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    if workers is None:
        workers = int(cfg.get("cache", {}).get("prepare_workers", 1))
    workers = max(1, int(workers))

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=cfg["wandb"].get("run_name", "cache-prep"),
        tags=cfg["wandb"].get("tags", []),
        mode=cfg["wandb"].get("mode", "online"),
        config=cfg,
    )

    cases = load_registry(cfg)
    print(f"Cases: {len(cases)} | workers: {workers}")
    preprocess_cfg = cfg.get("preprocess", {})
    bg_value = preprocess_cfg.get(
        "background_value",
        preprocess_cfg.get("crop", {}).get("background_value", -5.0),
    )

    payloads = [
        {
            "cfg": cfg,
            "cache_dir": cache_dir,
            "bg_value": float(bg_value),
            "uid": case.uid,
            "patient_id": case.patient_id,
            "site": case.site,
            "image_path": case.image_path,
            "mask_path": case.mask_path,
        }
        for case in cases
    ]

    processed = 0
    skipped = 0
    errors = 0

    if workers == 1:
        for payload in tqdm(payloads):
            result = _process_case_payload(payload)
            if "warning" in result:
                print(f"[prepare_cache][warning] {result['warning']}")
            if result.get("status") == "error":
                errors += 1
                print(f"[prepare_cache][error] {result.get('uid')}: {result.get('error')}")
                continue
            if result.get("status") == "skipped":
                skipped += 1
                continue
            processed += 1
            _log_case_result(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_case_payload, payload) for payload in payloads]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if "warning" in result:
                    print(f"[prepare_cache][warning] {result['warning']}")
                if result.get("status") == "error":
                    errors += 1
                    print(f"[prepare_cache][error] {result.get('uid')}: {result.get('error')}")
                    continue
                if result.get("status") == "skipped":
                    skipped += 1
                    continue
                processed += 1
                _log_case_result(result)

    wandb.finish()
    print(
        f"[prepare_cache] done. processed={processed} skipped={skipped} errors={errors} "
        f"(workers={workers})"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for case preprocessing (default: cache.prepare_workers or 1).",
    )
    args = parser.parse_args()
    main(args.cfg, workers=args.workers)
