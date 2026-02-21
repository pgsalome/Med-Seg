import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CaseItem:
    uid: str
    patient_id: str
    site: str
    image_path: str
    mask_path: Optional[str]
    raw: Dict[str, Any]


def _get_ci(row: Dict[str, Any], key: str):
    if key in row:
        return row[key]
    key_l = key.lower()
    for existing_key, value in row.items():
        if isinstance(existing_key, str) and existing_key.lower() == key_l:
            return value
    return None


def _pick_first(row: Dict[str, Any], keys: List[str]):
    for key in keys:
        value = _get_ci(row, key)
        if value not in (None, ""):
            return value
    return None


def _to_number(value):
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "false"}:
            return float(1 if raw == "true" else 0)
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _coerce_id(value: Any, default: Any = 1):
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


def _matches_keep_value(value: Any, keep_value: Any) -> bool:
    if value == keep_value:
        return True

    n_value = _to_number(value)
    n_keep = _to_number(keep_value)
    if n_value is not None and n_keep is not None:
        return n_value == n_keep

    if isinstance(value, str):
        split = value.strip().lower()
        if split in {"train", "training"}:
            return _matches_keep_value(0, keep_value)
        if split in {"val", "valid", "validation", "test"}:
            return _matches_keep_value(1, keep_value)

    return str(value).strip().lower() == str(keep_value).strip().lower()


def _resolve_path(path_value: Any, registry_dir: str) -> Optional[str]:
    if path_value in (None, ""):
        return None
    if not isinstance(path_value, str):
        return None

    path = os.path.expanduser(path_value)
    if os.path.exists(path):
        return path

    alt = os.path.normpath(os.path.join(registry_dir, path))
    if os.path.exists(alt):
        return alt
    return path


def _read_registry_payload(path: str):
    with open(path, "r", encoding="utf-8") as file:
        raw = file.read()
    if not raw.strip():
        raise ValueError(f"Registry file is empty: {path}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in registry file: {path}") from exc


def _extract_rows(payload: Any) -> Tuple[List[Dict[str, Any]], str]:
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
        if len(rows) != len(payload):
            raise ValueError("Registry list contains non-dict items.")
        return rows, "list"

    if isinstance(payload, dict):
        for key in ("rows", "cases", "data", "patients", "items", "registry"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = [row for row in value if isinstance(row, dict)]
                if len(rows) != len(value):
                    raise ValueError(f"Registry field '{key}' contains non-dict items.")
                return rows, f"dict[{key}]"

        if payload and all(isinstance(value, dict) for value in payload.values()):
            rows: List[Dict[str, Any]] = []
            for patient_id, row in payload.items():
                item = dict(row)
                item.setdefault("patient_id", str(patient_id))
                rows.append(item)
            return rows, "dict_map"

        return [payload], "single_dict"

    raise ValueError("Registry payload must be a list or dict.")


def _normalize_row(row: Dict[str, Any], cfg: dict) -> Dict[str, Any]:
    img_key = cfg["data"]["image_key"]
    msk_key = cfg["data"]["mask_key"]
    site_key = cfg["data"]["site_key"]
    test_key = cfg["data"]["test_flag_field"]

    out = dict(row)

    for nested_key in ("paths", "files"):
        nested = _get_ci(out, nested_key)
        if not isinstance(nested, dict):
            continue
        for key in (
            img_key,
            msk_key,
            "IMG",
            "image",
            "image_path",
            "img",
            "mask",
            "mask_path",
            "label",
            "seg",
            "segmentation",
            "FU1_CT",
            "FU1_CT1",
            "RN",
        ):
            if _get_ci(out, key) in (None, ""):
                value = _get_ci(nested, key)
                if value not in (None, ""):
                    out[key] = value

    patient_id = _pick_first(
        out, ["patient_id", "PatientID", "patient", "subject_id", "subject", "case_id", "id"]
    )
    if patient_id is not None:
        out["patient_id"] = str(patient_id)

    site = _pick_first(out, [site_key, "site_group", "site", "center", "institution", "hospital"])
    if site is not None:
        out[site_key] = site

    image_path = _pick_first(
        out,
        [
            img_key,
            "IMG",
            "image",
            "image_path",
            "img",
            "scan",
            "volume",
            "t1",
            "t1ce",
            "t1post",
            "FU1_CT",
            "FU1_CT1",
        ],
    )
    if image_path is not None:
        out[img_key] = image_path
        out["IMG"] = image_path

    mask_path = _pick_first(
        out,
        [msk_key, "mask", "mask_path", "label", "label_path", "seg", "segmentation", "RN"],
    )
    if mask_path is not None:
        out[msk_key] = mask_path
        out["mask"] = mask_path
        out["RN"] = mask_path

    label_id = _pick_first(out, ["label_id", "mask_id", "MASK_ID", "lesion_id", "FU1_lesion_id"])
    out["label_id"] = _coerce_id(label_id, default=1)
    out.setdefault("mask_id", out["label_id"])

    test_value = _pick_first(out, [test_key, "test", "is_test", "split"])
    if test_value is not None:
        out[test_key] = test_value

    for id_key in cfg["data"]["id_keys"]:
        if id_key in out and out[id_key] not in (None, ""):
            continue

        low = id_key.lower()
        if "patient" in low:
            if "patient_id" in out:
                out[id_key] = out["patient_id"]
            continue
        if "lesion_id" in low:
            lesion_id = _pick_first(out, [id_key, "lesion_id", "FU1_lesion_id"])
            if lesion_id is not None:
                out[id_key] = lesion_id
            continue
        if "lesion_seen_date" in low or low.endswith("date"):
            lesion_date = _pick_first(out, [id_key, "lesion_seen_date", "FU1_lesion_seen_date", "date"])
            if lesion_date is not None:
                out[id_key] = lesion_date

    return out


def normalize_registry_file(registry_path: str, cfg: dict) -> List[Dict[str, Any]]:
    payload = _read_registry_payload(registry_path)
    rows, source = _extract_rows(payload)
    normalized_rows = [_normalize_row(row, cfg) for row in rows]

    changed = sum(1 for raw, norm in zip(rows, normalized_rows) if raw != norm)
    if source != "list" or changed > 0:
        print(f"[registry] normalized input (source={source}, rows={len(rows)}, adjusted_rows={changed}).")

    return normalized_rows


def load_registry(cfg: dict) -> List[CaseItem]:
    registry_path = cfg["data"]["registry_json"]
    registry_dir = os.path.dirname(os.path.abspath(registry_path))
    rows = normalize_registry_file(registry_path, cfg)

    img_key = cfg["data"]["image_key"]
    msk_key = cfg["data"]["mask_key"]
    site_key = cfg["data"]["site_key"]
    id_keys = cfg["data"]["id_keys"]

    out: List[CaseItem] = []
    for row in rows:
        if cfg["data"].get("use_test_flag", False):
            flag_field = cfg["data"]["test_flag_field"]
            flag_value = row.get(flag_field)
            if flag_value is not None and not _matches_keep_value(
                flag_value, cfg["data"]["keep_test_value"]
            ):
                continue

        image_path = _resolve_path(
            _pick_first(row, [img_key, "IMG", "image", "image_path", "FU1_CT", "FU1_CT1"]),
            registry_dir,
        )
        mask_path = _resolve_path(
            _pick_first(row, [msk_key, "mask", "mask_path", "RN", "label"]),
            registry_dir,
        )

        if not image_path or not os.path.exists(image_path):
            continue
        if cfg["data"].get("require_mask", True):
            if not mask_path or not os.path.exists(mask_path):
                continue

        uid = "__".join(str(row.get(key, "NA")) for key in id_keys)
        out.append(
            CaseItem(
                uid=uid,
                patient_id=str(row.get("patient_id", "NA")),
                site=str(row.get(site_key, "NA")),
                image_path=image_path,
                mask_path=mask_path,
                raw=row,
            )
        )
    return out
