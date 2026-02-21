import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_registry import CaseItem
from .preprocess import load_and_preprocess_with_steps


def _cfg_hash(cfg: dict) -> str:
    sub = {
        "preprocess": cfg["preprocess"],
        "data": {
            "image_key": cfg["data"]["image_key"],
            "mask_key": cfg["data"]["mask_key"],
        },
    }
    payload = json.dumps(sub, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def case_cache_key(case: CaseItem, cfg: dict) -> str:
    cfg_hash = _cfg_hash(cfg)
    base = f"{case.uid}|{case.image_path}|{case.mask_path}|{cfg_hash}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def save_npz(
    path: str,
    image: np.ndarray,
    label: np.ndarray,
    spacing,
    brain_mask: np.ndarray = None,
) -> None:
    payload = {
        "image": image,
        "label": label,
        "spacing": np.array(spacing, dtype=np.float32),
    }
    if brain_mask is not None:
        payload["brain_mask"] = brain_mask.astype(np.uint8)
    np.savez_compressed(path, **payload)


def load_npz(path: str):
    with np.load(path, allow_pickle=False) as data:
        brain = data["brain_mask"] if "brain_mask" in data.files else None
        image = data["image"]
        label = data["label"]
        spacing = tuple(data["spacing"].tolist())
    return image, label, spacing, brain


def extract_patch(
    arr_czyx: np.ndarray,
    center_zyx: Tuple[int, int, int],
    patch_zyx: Tuple[int, int, int],
) -> np.ndarray:
    channels, z_dim, y_dim, x_dim = arr_czyx.shape
    pz, py, px = patch_zyx
    cz, cy, cx = center_zyx

    z0 = max(0, cz - pz // 2)
    z1 = min(z_dim, z0 + pz)
    y0 = max(0, cy - py // 2)
    y1 = min(y_dim, y0 + py)
    x0 = max(0, cx - px // 2)
    x1 = min(x_dim, x0 + px)

    patch = np.zeros((channels, pz, py, px), dtype=arr_czyx.dtype)
    patch[:, : z1 - z0, : y1 - y0, : x1 - x0] = arr_czyx[:, z0:z1, y0:y1, x0:x1]
    return patch


def rand_center_from_mask(mask_zyx: np.ndarray):
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        return None
    z, y, x = coords[np.random.randint(0, coords.shape[0])]
    return int(z), int(y), int(x)


def brain_mask_from_zscored(img_zyx: np.ndarray) -> np.ndarray:
    threshold = np.percentile(img_zyx, 30)
    return (img_zyx > threshold).astype(np.uint8)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _slice_idx_centroid(mask_zyx: np.ndarray) -> int:
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        return int(mask_zyx.shape[0] // 2)
    z = int(np.round(coords[:, 0].mean()))
    return int(np.clip(z, 0, mask_zyx.shape[0] - 1))


def _to_uint8(img_2d: np.ndarray) -> np.ndarray:
    x = img_2d.astype(np.float32)
    finite = np.isfinite(x)
    if finite.any():
        vals = x[finite]
    else:
        vals = np.array([0.0], dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals) + 1e-6)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (255.0 * x).astype(np.uint8)


class MedSegCachePatchDataset(Dataset):
    def __init__(self, cases: List[CaseItem], cfg: dict, transforms=None):
        self.cases = cases
        self.cfg = cfg
        self.transforms = transforms

        self.cache_dir = cfg["cache"]["cache_dir"]
        os.makedirs(self.cache_dir, exist_ok=True)

        self.patch = tuple(int(x) for x in cfg["sampling"]["patch_size"])
        self.p_fg = float(cfg["sampling"]["p_fg"])
        self.p_hn = float(cfg["sampling"]["p_hardneg"])
        self.p_bg = float(cfg["sampling"]["p_bg"])
        self.hn_pct = float(cfg["sampling"]["hardneg"]["bright_percentile"])
        self.hn_min_dist = int(cfg["sampling"]["hardneg"]["min_dist_to_lesion_vox"])
        self.min_brain_frac = float(cfg["sampling"].get("min_brain_frac", 0.60))
        self.max_brain_resample_tries = int(cfg["sampling"].get("max_brain_resample_tries", 16))
        comp_cfg = cfg["sampling"].get("component_sampling", {})
        self.comp_sampling_enabled = bool(comp_cfg.get("enabled", True))
        self.small_component_max_vox = int(comp_cfg.get("small_component_max_vox", 300))
        self.small_component_oversample_prob = float(
            comp_cfg.get("small_component_oversample_prob", 0.75)
        )
        self._component_cache: Dict[str, Tuple[List[np.ndarray], np.ndarray]] = {}

        debug_cfg = cfg.get("debug", {})
        self.debug_patch_save_max = int(debug_cfg.get("save_patch_samples_max", 5))
        self.debug_patch_dir = os.path.join(self.cache_dir, "debug_patches")
        os.makedirs(self.debug_patch_dir, exist_ok=True)

    def __len__(self):
        return len(self.cases)

    def _get_cached(self, case: CaseItem):
        cache_key = case_cache_key(case, self.cfg)
        npz_path = os.path.join(self.cache_dir, f"{cache_key}.npz")

        if os.path.exists(npz_path) and not self.cfg["cache"]["overwrite"]:
            image, label, spacing, brain_mask = load_npz(npz_path)
            if brain_mask is not None:
                return image, label, spacing, brain_mask

        image, label, spacing, _, brain_mask = load_and_preprocess_with_steps(
            case.image_path,
            case.mask_path,
            self.cfg,
        )
        save_npz(npz_path, image, label, spacing, brain_mask)
        return image, label, spacing, brain_mask

    def _hardneg_center(
        self,
        img_zyx: np.ndarray,
        lesion_zyx: np.ndarray,
        brain_zyx: np.ndarray,
    ) -> Tuple[int, int, int]:
        if (brain_zyx > 0).any():
            bright_thr = np.percentile(img_zyx[brain_zyx > 0], self.hn_pct)
        else:
            bright_thr = np.percentile(img_zyx, self.hn_pct)

        candidates = (img_zyx >= bright_thr) & (brain_zyx > 0) & (~(lesion_zyx > 0))
        coords = np.argwhere(candidates)
        if coords.size == 0:
            coords = np.argwhere(brain_zyx > 0)

        for _ in range(50):
            z, y, x = coords[np.random.randint(0, coords.shape[0])]
            z0 = max(0, z - self.hn_min_dist)
            z1 = min(lesion_zyx.shape[0], z + self.hn_min_dist + 1)
            y0 = max(0, y - self.hn_min_dist)
            y1 = min(lesion_zyx.shape[1], y + self.hn_min_dist + 1)
            x0 = max(0, x - self.hn_min_dist)
            x1 = min(lesion_zyx.shape[2], x + self.hn_min_dist + 1)
            if lesion_zyx[z0:z1, y0:y1, x0:x1].any():
                continue
            return int(z), int(y), int(x)

        z, y, x = coords[0]
        return int(z), int(y), int(x)

    def _sample_center(
        self,
        case_uid: str,
        img_zyx: np.ndarray,
        lesion_zyx: np.ndarray,
        brain_zyx: np.ndarray,
    ) -> Tuple[int, int, int]:
        sample_mode = np.random.rand()
        if sample_mode < self.p_fg:
            if self.comp_sampling_enabled:
                center = self._component_aware_fg_center(case_uid, lesion_zyx)
            else:
                center = rand_center_from_mask(lesion_zyx)
            if center is None:
                coords = np.argwhere(brain_zyx > 0)
                z, y, x = coords[np.random.randint(0, coords.shape[0])]
                center = (int(z), int(y), int(x))
            return center

        if sample_mode < self.p_fg + self.p_hn:
            return self._hardneg_center(img_zyx, lesion_zyx, brain_zyx)

        coords = np.argwhere(brain_zyx > 0)
        z, y, x = coords[np.random.randint(0, coords.shape[0])]
        return int(z), int(y), int(x)

    def _component_aware_fg_center(
        self,
        case_uid: str,
        lesion_zyx: np.ndarray,
    ):
        components, sizes = self._lesion_components(case_uid, lesion_zyx)
        if not components:
            return None

        small_ids = np.flatnonzero(sizes <= self.small_component_max_vox)
        choose_small = (
            small_ids.size > 0 and np.random.rand() < self.small_component_oversample_prob
        )
        if choose_small:
            comp_idx = int(small_ids[np.random.randint(0, small_ids.size)])
        else:
            comp_idx = int(np.random.randint(0, len(components)))

        coords = components[comp_idx]
        z, y, x = coords[np.random.randint(0, coords.shape[0])]
        return int(z), int(y), int(x)

    def _lesion_components(
        self,
        case_uid: str,
        lesion_zyx: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        cached = self._component_cache.get(case_uid)
        if cached is not None:
            return cached

        import scipy.ndimage as ndi

        lesion_bin = lesion_zyx > 0
        labeled, num = ndi.label(lesion_bin)
        components: List[np.ndarray] = []
        sizes = []
        if int(num) > 0:
            for comp_id in range(1, int(num) + 1):
                coords = np.argwhere(labeled == comp_id)
                if coords.size == 0:
                    continue
                components.append(coords.astype(np.int32))
                sizes.append(int(coords.shape[0]))

        size_arr = np.asarray(sizes, dtype=np.int32)
        self._component_cache[case_uid] = (components, size_arr)
        return components, size_arr

    def _maybe_save_patch_debug(
        self, image_patch_czyx: np.ndarray, label_patch_czyx: np.ndarray, uid: str
    ) -> None:
        if self.debug_patch_save_max <= 0:
            return
        lock_path = os.path.join(self.debug_patch_dir, ".save_lock")
        lock_fd = None
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return
        except OSError:
            return

        try:
            existing = [
                name for name in os.listdir(self.debug_patch_dir) if name.endswith("__overlay.png")
            ]
            if len(existing) >= self.debug_patch_save_max:
                return

            idx = len(existing)
            base = f"{idx:03d}__{_safe_name(uid)}"
            img_zyx = image_patch_czyx[0].astype(np.float32)
            msk_zyx = (label_patch_czyx[0] > 0.5).astype(np.uint8)
            z = _slice_idx_centroid(msk_zyx)
            img_u8 = _to_uint8(img_zyx[z])
            msk_u8 = (msk_zyx[z] * 255).astype(np.uint8)

            # Save full 3D patches.
            np.save(
                os.path.join(self.debug_patch_dir, f"{base}__image_patch.npy"),
                image_patch_czyx.astype(np.float32),
            )
            np.save(
                os.path.join(self.debug_patch_dir, f"{base}__mask_patch.npy"),
                msk_zyx[None, ...].astype(np.uint8),
            )

            # Save 2D visual slices (image / mask / overlay) for quick QA.
            overlay = np.stack([img_u8, img_u8, img_u8], axis=-1)
            mask2d = msk_u8 > 0
            if mask2d.any():
                alpha = 0.45
                overlay[..., 0][mask2d] = (
                    (1.0 - alpha) * overlay[..., 0][mask2d] + alpha * 255.0
                ).astype(np.uint8)
                overlay[..., 1][mask2d] = (
                    (1.0 - alpha) * overlay[..., 1][mask2d]
                ).astype(np.uint8)
                overlay[..., 2][mask2d] = (
                    (1.0 - alpha) * overlay[..., 2][mask2d]
                ).astype(np.uint8)

            try:
                import matplotlib.image as mpimg

                mpimg.imsave(
                    os.path.join(self.debug_patch_dir, f"{base}__image.png"),
                    img_u8,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                mpimg.imsave(
                    os.path.join(self.debug_patch_dir, f"{base}__mask.png"),
                    msk_u8,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                mpimg.imsave(
                    os.path.join(self.debug_patch_dir, f"{base}__overlay.png"),
                    overlay,
                )
            except Exception:
                # Non-fatal: keep .npy patch dumps even if PNG export fails.
                return
        finally:
            try:
                if lock_fd is not None:
                    os.close(lock_fd)
            finally:
                try:
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                except OSError:
                    pass

    def __getitem__(self, idx):
        case = self.cases[idx]
        img_czyx, lab_czyx, _, brain_cached = self._get_cached(case)

        img_zyx = img_czyx[0]
        lesion_zyx = lab_czyx[0]
        if brain_cached is None:
            brain_zyx = brain_mask_from_zscored(img_zyx)
        else:
            brain_zyx = (brain_cached > 0).astype(np.uint8)
        brain_czyx = brain_zyx[None, ...].astype(np.float32)

        center = self._sample_center(case.uid, img_zyx, lesion_zyx, brain_zyx)
        tries = max(1, self.max_brain_resample_tries)
        if self.min_brain_frac > 0.0:
            for _ in range(tries):
                brain_patch = extract_patch(brain_czyx, center, self.patch)[0]
                brain_frac = float(brain_patch.mean())
                if brain_frac >= self.min_brain_frac:
                    break
                center = self._sample_center(case.uid, img_zyx, lesion_zyx, brain_zyx)

        image_patch = extract_patch(img_czyx, center, self.patch)
        label_patch = extract_patch(lab_czyx.astype(np.float32), center, self.patch)
        weight_patch = extract_patch(brain_czyx.astype(np.float32), center, self.patch)
        self._maybe_save_patch_debug(image_patch, label_patch, case.uid)

        sample = {
            "image": torch.from_numpy(image_patch),
            "label": torch.from_numpy(label_patch),
            "weight": torch.from_numpy(weight_patch.astype(np.float32)),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
