from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .config import load_config
from .io_registry import normalize_registry_file


def _repo_root() -> str:
    # src/medseg/registry_build.py -> repo root two levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _pick_first_ci(row: Dict[str, Any], keys: List[str]):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        key_l = key.lower()
        for existing_key, value in row.items():
            if isinstance(existing_key, str) and existing_key.lower() == key_l and value not in (None, ""):
                return value
    return None


def _coerce_label_id(value: Any, default: Any = 1):
    if value in (None, ""):
        value = default

    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            return default
        if raw.lstrip("-").isdigit():
            try:
                return int(raw)
            except ValueError:
                return raw
        return raw

    return str(value)


def _resolve_path(path_value: str, bases: List[str], require_exists: bool) -> str:
    if os.path.isabs(path_value):
        return path_value

    candidates = [os.path.normpath(os.path.join(base, path_value)) for base in bases]
    if require_exists:
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return candidates[0]


def standardize_registry_rows(
    rows: List[Dict[str, Any]],
    source_cfg: dict,
    input_image_key: Optional[str] = None,
    input_mask_key: Optional[str] = None,
    label_id_field: Optional[str] = None,
    label_id_default: Any = 1,
    label_id_key: str = "label_id",
) -> List[Dict[str, Any]]:
    image_key = input_image_key or source_cfg["data"]["image_key"]
    mask_key = input_mask_key or source_cfg["data"]["mask_key"]
    site_key = source_cfg["data"].get("site_key", "site_group")
    test_key = source_cfg["data"].get("test_flag_field", "test")
    resolved_label_id_field = (
        label_id_field
        or source_cfg["data"].get("mask_id_key")
        or source_cfg["data"].get("label_id_field")
    )

    out: List[Dict[str, Any]] = []
    missing_label_id_rows = 0
    missing_label_id_patients: List[str] = []
    for row in rows:
        image_path = _pick_first_ci(
            row,
            [image_key],
        )
        mask_path = _pick_first_ci(
            row,
            [mask_key],
        )

        patient_id = _pick_first_ci(
            row,
            ["patient_id", "PatientID", "subject_id", "subject", "id", "case_id"],
        )
        item: Dict[str, Any] = {
            "patient_id": str(patient_id) if patient_id not in (None, "") else "NA",
            "IMG": image_path,
            "mask": mask_path,
        }

        if resolved_label_id_field:
            row_label_id = _pick_first_ci(row, [resolved_label_id_field])
        else:
            row_label_id = _pick_first_ci(
                row,
                ["label_id", "mask_id", "lesion_id", "FU1_lesion_id"],
            )
        if row_label_id in (None, ""):
            missing_label_id_rows += 1
            if len(missing_label_id_patients) < 10:
                missing_label_id_patients.append(item["patient_id"])
        item[label_id_key] = _coerce_label_id(row_label_id, default=label_id_default)
        if label_id_key != "label_id":
            item["label_id"] = item[label_id_key]

        site = _pick_first_ci(row, [site_key, "site_group", "site", "center", "institution"])
        if site is not None:
            item["site_group"] = site

        test_value = _pick_first_ci(row, [test_key, "test", "is_test", "split"])
        if test_value is not None:
            item["test"] = test_value

        out.append(item)

    if missing_label_id_rows > 0:
        source_name = resolved_label_id_field or "label_id/mask_id aliases"
        sample = ", ".join(missing_label_id_patients)
        print(
            "[registry][warning] "
            f"{missing_label_id_rows}/{len(rows)} rows missing '{source_name}'. "
            f"Defaulted {label_id_key}={label_id_default}. "
            f"Sample patient_id: {sample}"
        )
    return out


def build_standard_registry_from_source_cfg(
    source_cfg_path: str,
    output_path: str,
    input_image_key: Optional[str] = None,
    input_mask_key: Optional[str] = None,
    label_id_field: Optional[str] = None,
    label_id_default: Any = 1,
    label_id_key: str = "label_id",
) -> Dict[str, Any]:
    repo_root = _repo_root()
    cwd = os.getcwd()
    source_cfg_path = _resolve_path(
        source_cfg_path,
        bases=[cwd, repo_root],
        require_exists=True,
    )

    source_cfg = load_config(source_cfg_path)
    source_cfg_dir = os.path.dirname(os.path.abspath(source_cfg_path))
    source_registry_path = _resolve_path(
        source_cfg["data"]["registry_json"],
        bases=[source_cfg_dir, cwd, repo_root],
        require_exists=True,
    )

    source_cfg_local = deepcopy(source_cfg)
    source_cfg_local["data"] = dict(source_cfg["data"])
    source_cfg_local["data"]["registry_json"] = source_registry_path

    rows = normalize_registry_file(source_registry_path, source_cfg_local)
    standard_rows = standardize_registry_rows(
        rows=rows,
        source_cfg=source_cfg_local,
        input_image_key=input_image_key,
        input_mask_key=input_mask_key,
        label_id_field=label_id_field,
        label_id_default=label_id_default,
        label_id_key=label_id_key,
    )

    output_path = _resolve_path(output_path, bases=[repo_root, cwd], require_exists=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(standard_rows, file, indent=2)

    return {
        "source_cfg_path": source_cfg_path,
        "source_registry_path": source_registry_path,
        "output_path": output_path,
        "rows": len(standard_rows),
    }


def ensure_training_registry(cfg: dict, cfg_path: str) -> dict:
    registry_local_cfg = cfg.get("registry_local", {})
    if not registry_local_cfg.get("auto_build", False):
        return cfg

    source_cfg_path = registry_local_cfg.get(
        "source_cfg",
        "configs/experiment_triad_plain_preregistry.yaml",
    )
    output_path = registry_local_cfg.get(
        "output_path",
        cfg["data"].get("registry_json", "registries/registry.local.json"),
    )
    rebuild_each_run = bool(registry_local_cfg.get("rebuild_each_run", True))
    label_id_key = str(registry_local_cfg.get("label_id_key", "label_id"))
    label_id_field = registry_local_cfg.get("label_id_field")
    if label_id_field in (None, ""):
        label_id_field = registry_local_cfg.get("mask_id_key")

    repo_root = _repo_root()
    cwd = os.getcwd()
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    output_abs = _resolve_path(output_path, bases=[repo_root, cwd, cfg_dir], require_exists=False)

    if rebuild_each_run or not os.path.exists(output_abs):
        stats = build_standard_registry_from_source_cfg(
            source_cfg_path=source_cfg_path,
            output_path=output_abs,
            input_image_key=registry_local_cfg.get("input_image_key"),
            input_mask_key=registry_local_cfg.get("input_mask_key"),
            label_id_field=label_id_field,
            label_id_default=registry_local_cfg.get("label_id_default", 1),
            label_id_key=label_id_key,
        )
        print(
            "[registry] built training registry: "
            f"{stats['rows']} rows -> {stats['output_path']} "
            f"(source: {stats['source_registry_path']})"
        )
    else:
        print(f"[registry] using existing training registry: {output_abs}")

    out = deepcopy(cfg)
    out["data"] = dict(cfg["data"])
    out["data"]["registry_json"] = output_abs
    out["data"]["image_key"] = "IMG"
    out["data"]["mask_key"] = "mask"

    id_keys = list(out["data"].get("id_keys", []))
    if "patient_id" not in id_keys:
        id_keys.insert(0, "patient_id")
    if label_id_key not in id_keys:
        id_keys.append(label_id_key)
    out["data"]["id_keys"] = id_keys
    return out
