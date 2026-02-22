import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .dataset import case_cache_key, load_npz, save_npz
from .preprocess import load_and_preprocess_with_steps


def _binary_dice_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    p = (pred > 0).astype(np.uint8)
    t = (target > 0).astype(np.uint8)
    inter = float((p & t).sum())
    denom = float(p.sum() + t.sum())
    return float((2.0 * inter + eps) / (denom + eps))


def parse_size_bins(cfg: dict) -> List[Dict[str, int]]:
    raw_bins = cfg.get("metrics", {}).get("size_bins")
    if not raw_bins:
        raw_bins = [
            {"name": "tiny", "min_vox": 0, "max_vox": 50},
            {"name": "small", "min_vox": 50, "max_vox": 500},
            {"name": "medium", "min_vox": 500, "max_vox": 10000},
            {"name": "large", "min_vox": 10000, "max_vox": 1_000_000_000},
        ]

    parsed: List[Dict[str, int]] = []
    for idx, item in enumerate(raw_bins):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"bin_{idx}"))
        min_vox = int(item.get("min_vox", 0))
        max_vox = int(item.get("max_vox", 1_000_000_000))
        if max_vox <= min_vox:
            continue
        parsed.append({"name": name, "min_vox": min_vox, "max_vox": max_vox})

    if not parsed:
        raise ValueError("metrics.size_bins is empty/invalid after parsing.")
    return parsed


class SizeStratifiedDiceAccumulator:
    def __init__(self, bins: Sequence[Dict[str, int]]):
        self.bins = list(bins)
        self._sum: Dict[str, float] = {b["name"]: 0.0 for b in self.bins}
        self._count: Dict[str, int] = {b["name"]: 0 for b in self.bins}

    def _pick_bin_name(self, fg_voxels: int) -> Optional[str]:
        v = int(fg_voxels)
        for b in self.bins:
            if int(b["min_vox"]) <= v < int(b["max_vox"]):
                return str(b["name"])
        return None

    def update(self, pred_binary: np.ndarray, target_binary: np.ndarray, mask: Optional[np.ndarray] = None):
        pred = (pred_binary > 0).astype(np.uint8)
        target = (target_binary > 0).astype(np.uint8)

        if mask is not None:
            m = (mask > 0).astype(np.uint8)
            pred = pred * m
            target = target * m

        fg = int(target.sum())
        if fg <= 0:
            return

        bin_name = self._pick_bin_name(fg)
        if bin_name is None:
            return

        score = _binary_dice_numpy(pred, target)
        if np.isfinite(score):
            self._sum[bin_name] += float(score)
            self._count[bin_name] += 1

    def aggregate(self, prefix: str = "val/size") -> Dict[str, float]:
        out: Dict[str, float] = {}
        total_count = 0
        weighted_sum = 0.0
        for b in self.bins:
            name = str(b["name"])
            c = int(self._count.get(name, 0))
            s = float(self._sum.get(name, 0.0))
            d = float(s / c) if c > 0 else float("nan")
            out[f"{prefix}/{name}/count"] = c
            out[f"{prefix}/{name}/dice"] = d
            if c > 0 and np.isfinite(d):
                total_count += c
                weighted_sum += d * c

        out[f"{prefix}/overall_count"] = int(total_count)
        out[f"{prefix}/overall_dice"] = (
            float(weighted_sum / total_count) if total_count > 0 else float("nan")
        )
        return out


def _lesion_voxels_from_case(case, cfg: dict, cache_dir: str) -> int:
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = case_cache_key(case, cfg)
    npz_path = os.path.join(cache_dir, f"{cache_key}.npz")

    if os.path.exists(npz_path) and not cfg.get("cache", {}).get("overwrite", False):
        _img, label, _spacing, _brain = load_npz(npz_path)
        return int(label.sum())

    image, label, spacing, _steps, brain = load_and_preprocess_with_steps(
        case.image_path,
        case.mask_path,
        cfg,
    )
    save_npz(npz_path, image, label, spacing, brain)
    return int(label.sum())


def compute_case_sampling_weights(
    cases,
    cfg: dict,
    cache_dir: str,
    method: str = "inverse_sqrt",
    floor: float = 1.0,
    cap: float = 8.0,
):
    if len(cases) == 0:
        return []

    vols = np.asarray(
        [max(1.0, float(_lesion_voxels_from_case(case, cfg, cache_dir))) for case in cases],
        dtype=np.float32,
    )

    method_l = str(method).strip().lower()
    if method_l == "uniform":
        raw = np.ones_like(vols)
    elif method_l == "inverse_log":
        raw = 1.0 / np.log1p(vols)
    elif method_l == "inverse_sqrt":
        raw = 1.0 / np.sqrt(vols)
    else:
        raise ValueError("oversampling.method must be one of: ['inverse_sqrt', 'inverse_log', 'uniform']")

    raw = raw / max(float(np.min(raw)), 1e-8)
    weights = np.clip(raw, float(floor), float(cap)).astype(np.float32)
    return weights.tolist()
