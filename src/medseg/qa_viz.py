import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _to_uint8(x: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    x = x.astype(np.float32)
    if valid_mask is not None and valid_mask.shape == x.shape and valid_mask.any():
        vals = x[valid_mask]
    else:
        vals = x.reshape(-1)

    vmin, vmax = np.percentile(vals, [1, 99])
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-8)
    out = (255 * x).astype(np.uint8)

    if valid_mask is not None and valid_mask.shape == x.shape:
        out[~valid_mask] = 0
    return out


def _slice_idx_mid(vol_zyx: np.ndarray) -> int:
    return vol_zyx.shape[0] // 2


def _slice_idx_centroid(mask_zyx: np.ndarray) -> int:
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        return _slice_idx_mid(mask_zyx)

    z = int(np.round(coords[:, 0].mean()))
    z = int(np.clip(z, 0, mask_zyx.shape[0] - 1))
    if (mask_zyx[z] > 0).any():
        return z

    # If centroid rounding falls into an empty inter-slice gap, choose
    # the nearest non-empty lesion slice for stable visualization.
    per_z = (mask_zyx > 0).reshape(mask_zyx.shape[0], -1).sum(axis=1)
    nz = np.where(per_z > 0)[0]
    if nz.size == 0:
        return z

    dist = np.abs(nz - z)
    nearest = nz[dist == dist.min()]
    if nearest.size == 1:
        return int(nearest[0])
    return int(nearest[np.argmax(per_z[nearest])])


def _parse_step_index(prefix: str) -> int:
    tail = str(prefix).split("__")[-1]
    match = re.match(r"^(\d+)_", tail)
    if not match:
        return -1
    return int(match.group(1))


def _valid_non_bg_mask(arr: np.ndarray, bg_value: Optional[float]) -> np.ndarray:
    valid = np.isfinite(arr)
    if bg_value is None:
        return valid
    return valid & (~np.isclose(arr, float(bg_value), rtol=0.0, atol=1e-6))


def _inplane_aspect_from_spacing_zyx(spacing_zyx: Optional[tuple]) -> float:
    if spacing_zyx is None:
        return 1.0
    if len(spacing_zyx) != 3:
        return 1.0
    sy = float(spacing_zyx[1])
    sx = float(spacing_zyx[2])
    if (not np.isfinite(sy)) or (not np.isfinite(sx)) or sy <= 0.0 or sx <= 0.0:
        return 1.0
    return sy / sx


def _draw_mask_outline(ax, mask2d: np.ndarray, color: str = "red", linewidth: float = 1.0) -> None:
    if mask2d.ndim != 2:
        return
    if not np.any(mask2d):
        return
    # Keep contour coordinates in the same array index frame as imshow().
    # Using origin="upper" here flips y for contour data and misaligns mask outlines.
    ax.contour(
        mask2d.astype(np.uint8),
        levels=[0.5],
        colors=[color],
        linewidths=[float(linewidth)],
    )


def save_slice_png(
    vol_zyx: np.ndarray,
    out_path: str,
    title: str,
    z_index: int,
    mask_zyx: Optional[np.ndarray] = None,
    alpha: float = 0.35,
    spacing_zyx: Optional[tuple] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    z_index = int(np.clip(z_index, 0, vol_zyx.shape[0] - 1))
    img = _to_uint8(vol_zyx[z_index])
    aspect = _inplane_aspect_from_spacing_zyx(spacing_zyx)

    plt.figure(figsize=(5, 5), dpi=150)
    plt.axis("off")
    plt.title(title, fontsize=8)
    plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.gca().set_aspect(aspect)

    if mask_zyx is not None and z_index < mask_zyx.shape[0]:
        mask = mask_zyx[z_index] > 0
        if mask.shape == img.shape and mask.any():
            _draw_mask_outline(plt.gca(), mask, color="red", linewidth=1.0)

    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def _safe_hist_range(full_vals: np.ndarray, mask_vals: Optional[np.ndarray]) -> tuple:
    if mask_vals is not None and mask_vals.size > 0:
        all_vals = np.concatenate([full_vals, mask_vals], axis=0)
    else:
        all_vals = full_vals

    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(np.min(all_vals))
        hi = float(np.max(all_vals) + 1e-6)
    return float(lo), float(hi)


def _plot_normalized_hist(
    ax,
    full_vals: np.ndarray,
    mask_vals: Optional[np.ndarray],
    lo: float,
    hi: float,
    bins: int = 80,
) -> None:
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_full, _ = np.histogram(full_vals, bins=edges, density=True)
    ax.plot(centers, hist_full, color="tab:blue", lw=1.6, label=f"Full image (n={full_vals.size})")

    if mask_vals is not None and mask_vals.size > 0:
        hist_mask, _ = np.histogram(mask_vals, bins=edges, density=True)
        ax.plot(centers, hist_mask, color="tab:red", lw=1.6, label=f"Mask voxels (n={mask_vals.size})")
    else:
        ax.text(
            0.5,
            0.88,
            "Mask histogram unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Intensity", fontsize=8)
    ax.set_ylabel("Density (normalized)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(fontsize=7, loc="upper right")


def save_image_mask_separate_with_hist_png(
    vol_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    out_path: str,
    title: str,
    exclude_background: bool = False,
    bg_value: Optional[float] = None,
    spacing_zyx: Optional[tuple] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    z_img = int(np.clip(_slice_idx_mid(vol_zyx), 0, vol_zyx.shape[0] - 1))
    z_msk = int(np.clip(_slice_idx_centroid(mask_zyx), 0, mask_zyx.shape[0] - 1))
    aspect = _inplane_aspect_from_spacing_zyx(spacing_zyx)

    valid_3d = _valid_non_bg_mask(vol_zyx, bg_value) if exclude_background else np.isfinite(vol_zyx)
    valid_2d = valid_3d[z_img] if z_img < valid_3d.shape[0] else None
    img = _to_uint8(vol_zyx[z_img], valid_mask=valid_2d)
    mask = (mask_zyx[z_msk] > 0).astype(np.uint8) * 255

    full_vals = vol_zyx[valid_3d].astype(np.float32).ravel()
    if full_vals.size == 0:
        full_vals = np.array([0.0], dtype=np.float32)

    mask_vals = None
    if vol_zyx.shape == mask_zyx.shape:
        _mask_vals = vol_zyx[(mask_zyx > 0) & valid_3d].astype(np.float32).ravel()
        if _mask_vals.size > 0:
            mask_vals = _mask_vals

    lo, hi = _safe_hist_range(full_vals, mask_vals)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    fig.suptitle(title, fontsize=9)

    axes[0].imshow(img, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"Image only (z={z_img})", fontsize=8)
    axes[0].axis("off")
    axes[0].set_aspect(aspect)

    axes[1].imshow(mask, cmap="gray", interpolation="nearest")
    axes[1].set_title(f"Mask only (z={z_msk})", fontsize=8)
    axes[1].axis("off")
    axes[1].set_aspect(aspect)

    _plot_normalized_hist(axes[2], full_vals, mask_vals, lo, hi)
    axes[2].set_title("Normalized intensity distribution", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overlay_with_hist_png(
    vol_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    out_path: str,
    title: str,
    alpha: float = 0.35,
    exclude_background: bool = False,
    bg_value: Optional[float] = None,
    spacing_zyx: Optional[tuple] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    z = int(np.clip(_slice_idx_centroid(mask_zyx), 0, vol_zyx.shape[0] - 1))
    aspect = _inplane_aspect_from_spacing_zyx(spacing_zyx)
    valid_3d = _valid_non_bg_mask(vol_zyx, bg_value) if exclude_background else np.isfinite(vol_zyx)
    valid_2d = valid_3d[z] if z < valid_3d.shape[0] else None
    img = _to_uint8(vol_zyx[z], valid_mask=valid_2d)
    mask2d = mask_zyx[z] > 0

    full_vals = vol_zyx[valid_3d].astype(np.float32).ravel()
    if full_vals.size == 0:
        full_vals = np.array([0.0], dtype=np.float32)

    mask_vals = vol_zyx[(mask_zyx > 0) & valid_3d].astype(np.float32).ravel()
    if mask_vals.size == 0:
        mask_vals = None

    lo, hi = _safe_hist_range(full_vals, mask_vals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    fig.suptitle(title, fontsize=9)

    axes[0].imshow(img, cmap="gray", interpolation="nearest")
    if mask2d.shape == img.shape and mask2d.any():
        _draw_mask_outline(axes[0], mask2d, color="red", linewidth=1.0)
    axes[0].set_title(f"Image + mask overlay (z={z})", fontsize=8)
    axes[0].axis("off")
    axes[0].set_aspect(aspect)

    _plot_normalized_hist(axes[1], full_vals, mask_vals, lo, hi)
    axes[1].set_title("Normalized intensity distribution", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_mid_and_centroid(
    vol_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    out_dir: str,
    prefix: str,
    title: str,
    spacing_zyx: Optional[tuple] = None,
    background_value: Optional[float] = None,
    exclude_background_from_step: int = 4,
) -> None:
    # Keep only one QA image per step.
    for legacy_suffix in ("_mid.png", "_centroid.png", "_intensity_hist.png"):
        legacy_path = os.path.join(out_dir, f"{prefix}{legacy_suffix}")
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    overlay_path = os.path.join(out_dir, f"{prefix}_centroid_overlay.png")
    is_raw_step = "__00_raw" in prefix
    step_idx = _parse_step_index(prefix)
    exclude_bg = bool(step_idx >= int(exclude_background_from_step))
    if (vol_zyx.shape == mask_zyx.shape) and (not is_raw_step):
        save_overlay_with_hist_png(
            vol_zyx,
            mask_zyx,
            overlay_path,
            title=f"{title}",
            spacing_zyx=spacing_zyx,
            exclude_background=exclude_bg,
            bg_value=background_value,
        )
    else:
        save_image_mask_separate_with_hist_png(
            vol_zyx,
            mask_zyx,
            overlay_path,
            title=f"{title} image/mask separate (space mismatch)",
            spacing_zyx=spacing_zyx,
            exclude_background=exclude_bg,
            bg_value=background_value,
        )
