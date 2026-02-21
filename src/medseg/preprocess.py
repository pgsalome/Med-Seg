import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk


def sitk_resample_to_spacing(img: sitk.Image, out_spacing, is_label: bool) -> sitk.Image:
    in_spacing = img.GetSpacing()
    in_size = img.GetSize()
    out_spacing = tuple(float(x) for x in out_spacing)
    out_size = [
        int(np.round(in_size[i] * (in_spacing[i] / out_spacing[i]))) for i in range(3)
    ]

    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(out_spacing)
    res.SetSize(out_size)
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetTransform(sitk.Transform())
    res.SetDefaultPixelValue(0)
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return res.Execute(img)


def sitk_resample_to_reference(img: sitk.Image, ref: sitk.Image, is_label: bool) -> sitk.Image:
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(ref)
    res.SetTransform(sitk.Transform())
    res.SetDefaultPixelValue(0)
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return res.Execute(img)


def n4_bias_correct(img: sitk.Image, shrink_factor: int, num_iters) -> sitk.Image:
    image = sitk.Cast(img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(x) for x in num_iters])

    if shrink_factor > 1:
        image_small = sitk.Shrink(image, [shrink_factor] * 3)
        mask_small = sitk.Shrink(mask, [shrink_factor] * 3)
        _ = corrector.Execute(image_small, mask_small)
        log_bias_small = corrector.GetLogBiasFieldAsImage(image_small)
        log_bias = sitk.Resample(
            log_bias_small,
            image,
            sitk.Transform(),
            sitk.sitkBSpline,
            0.0,
            sitk.sitkFloat32,
        )
        corrected = image / sitk.Exp(log_bias)
        return sitk.Cast(corrected, img.GetPixelID())

    corrected = corrector.Execute(image, mask)
    return sitk.Cast(corrected, img.GetPixelID())


def orient_image(img: sitk.Image, orient: Optional[str]) -> sitk.Image:
    if not orient:
        return img

    code = str(orient).upper()
    try:
        return sitk.DICOMOrient(img, code)
    except Exception:
        # Compatibility with older SimpleITK versions.
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation(code)
        return orienter.Execute(img)


def make_simple_brain_mask(img_zyx: np.ndarray) -> np.ndarray:
    import scipy.ndimage as ndi

    x = img_zyx.astype(np.float32)
    lo, hi = np.percentile(x, [2, 98])
    x = np.clip(x, lo, hi)
    threshold = np.percentile(x, 20)

    mask = (x > threshold).astype(np.uint8)
    labels, num = ndi.label(mask)
    if num == 0:
        return np.ones_like(mask, dtype=np.uint8)

    sizes = ndi.sum(mask, labels, index=list(range(1, num + 1)))
    keep = int(np.argmax(sizes)) + 1
    out = (labels == keep).astype(np.uint8)
    out = ndi.binary_closing(out, iterations=2).astype(np.uint8)
    out = ndi.binary_fill_holes(out).astype(np.uint8)
    return out


def make_hdbet_brain_mask(
    image: sitk.Image,
    hdbet_cmd: str = "hd-bet",
    mode: Optional[str] = None,
    device: Optional[str] = None,
    tta: Optional[int] = None,
    postprocess: Optional[int] = None,
    save_mask: int = 1,
) -> Optional[np.ndarray]:
    cmd_candidates = []
    if hdbet_cmd:
        cmd_candidates.append(str(hdbet_cmd))

    # Robust fallback: resolve from current python env (e.g., .venv/bin/hd-bet).
    if not shutil.which(hdbet_cmd):
        venv_cmd = os.path.join(os.path.dirname(sys.executable), "hd-bet")
        if os.path.exists(venv_cmd):
            cmd_candidates.append(venv_cmd)

    # De-duplicate while preserving order.
    seen = set()
    unique_cmds = []
    for cmd in cmd_candidates:
        if cmd in seen:
            continue
        seen.add(cmd)
        unique_cmds.append(cmd)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.nii.gz")
        out_base = os.path.join(tmpdir, "output")
        out_path = f"{out_base}.nii.gz"
        sitk.WriteImage(image, input_path)

        for bin_cmd in unique_cmds:
            cmd = [bin_cmd, "-i", input_path, "-o", out_path]
            if mode:
                cmd += ["-mode", str(mode)]
            if device is not None:
                cmd += ["-device", str(device)]
            if tta is not None:
                cmd += ["-tta", str(int(tta))]
            if postprocess is not None:
                cmd += ["-pp", str(int(postprocess))]
            cmd += ["-s", str(int(save_mask))]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                continue

            candidates = [
                f"{out_base}_mask.nii.gz",
                f"{out_path}_mask.nii.gz",
                f"{out_base}_bet_mask.nii.gz",
                os.path.join(tmpdir, "output_mask.nii.gz"),
            ]
            candidates.extend(sorted(glob.glob(os.path.join(tmpdir, "*mask*.nii*"))))

            for mask_path in candidates:
                if not os.path.exists(mask_path):
                    continue
                mask_img = sitk.ReadImage(mask_path)
                return (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)

    return None


def crop_to_brain_bbox(
    img_zyx: np.ndarray,
    lesion_zyx: np.ndarray,
    brain_zyx: np.ndarray,
    margin: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(brain_zyx > 0)
    if coords.size == 0:
        return img_zyx, lesion_zyx, brain_zyx

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1

    zmin = max(zmin - margin, 0)
    ymin = max(ymin - margin, 0)
    xmin = max(xmin - margin, 0)
    zmax = min(zmax + margin, img_zyx.shape[0])
    ymax = min(ymax + margin, img_zyx.shape[1])
    xmax = min(xmax + margin, img_zyx.shape[2])

    crop = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
    return img_zyx[crop], lesion_zyx[crop], brain_zyx[crop]


def clip_and_normalize(
    img_zyx: np.ndarray,
    brain_mask_zyx: np.ndarray,
    p_lo: float,
    p_hi: float,
    mode: str = "zscore",
) -> np.ndarray:
    x = img_zyx.astype(np.float32)

    if (brain_mask_zyx > 0).any():
        vox = x[brain_mask_zyx > 0]
    else:
        vox = x.reshape(-1)

    lo, hi = np.percentile(vox, [p_lo, p_hi])
    x = np.clip(x, lo, hi)

    if (brain_mask_zyx > 0).any():
        values = x[brain_mask_zyx > 0]
    else:
        values = x.reshape(-1)

    mode_l = str(mode).strip().lower()
    if mode_l == "minmax":
        vmin = float(values.min())
        vmax = float(values.max())
        return (x - vmin) / (vmax - vmin + 1e-8)

    mean = float(values.mean())
    std = float(values.std()) + 1e-8
    return (x - mean) / std


def _same_space_report(
    image: sitk.Image,
    mask: sitk.Image,
    spacing_tol: float = 1e-3,
    origin_tol: float = 1e-2,
    direction_tol: float = 1e-3,
) -> Dict[str, Any]:
    img_size = tuple(int(v) for v in image.GetSize())
    msk_size = tuple(int(v) for v in mask.GetSize())
    img_spacing = tuple(float(v) for v in image.GetSpacing())
    msk_spacing = tuple(float(v) for v in mask.GetSpacing())
    img_origin = tuple(float(v) for v in image.GetOrigin())
    msk_origin = tuple(float(v) for v in mask.GetOrigin())
    img_dir = tuple(float(v) for v in image.GetDirection())
    msk_dir = tuple(float(v) for v in mask.GetDirection())

    size_equal = img_size == msk_size
    spacing_equal = np.allclose(img_spacing, msk_spacing, rtol=0.0, atol=float(spacing_tol))
    origin_equal = np.allclose(img_origin, msk_origin, rtol=0.0, atol=float(origin_tol))
    direction_equal = np.allclose(img_dir, msk_dir, rtol=0.0, atol=float(direction_tol))

    return {
        "same_space": bool(size_equal and spacing_equal and origin_equal and direction_equal),
        "size_equal": bool(size_equal),
        "spacing_equal": bool(spacing_equal),
        "origin_equal": bool(origin_equal),
        "direction_equal": bool(direction_equal),
        "img_size": img_size,
        "msk_size": msk_size,
        "img_spacing": img_spacing,
        "msk_spacing": msk_spacing,
        "img_origin": img_origin,
        "msk_origin": msk_origin,
        "img_direction": img_dir,
        "msk_direction": msk_dir,
    }


def load_and_preprocess_with_steps(image_path: str, mask_path: str, cfg: dict):
    steps: Dict[str, Any] = {}

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    # Early geometry sanity check to catch wrong registry keys (e.g., non-registered image key).
    check_cfg = cfg.get("preprocess", {}).get("space_check", {})
    enabled = bool(check_cfg.get("enabled", True))
    if enabled:
        report = _same_space_report(
            image=image,
            mask=mask,
            spacing_tol=float(check_cfg.get("spacing_tol", 1e-3)),
            origin_tol=float(check_cfg.get("origin_tol", 1e-2)),
            direction_tol=float(check_cfg.get("direction_tol", 1e-3)),
        )
        if not report["same_space"]:
            warnings.warn(
                (
                    "[preprocess][space-check] Image and mask are not in the same raw space; "
                    "mask will be resampled to the image reference grid during preprocessing.\n"
                    f"image: {image_path}\n"
                    f"mask:  {mask_path}\n"
                    f"size_equal={report['size_equal']} spacing_equal={report['spacing_equal']} "
                    f"origin_equal={report['origin_equal']} direction_equal={report['direction_equal']}\n"
                    f"img_size={report['img_size']} msk_size={report['msk_size']}\n"
                    f"img_spacing={report['img_spacing']} msk_spacing={report['msk_spacing']}"
                ),
                RuntimeWarning,
            )

    raw_spacing_zyx = tuple(float(v) for v in reversed(image.GetSpacing()))
    image_raw = sitk.GetArrayFromImage(image)
    lesion_raw = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8)
    steps["00_raw"] = {"img": image_raw, "msk": lesion_raw, "spacing_zyx": raw_spacing_zyx}

    orient = cfg["preprocess"].get("orient")
    image = orient_image(image, orient)
    mask = orient_image(mask, orient)

    spacing = tuple(cfg["preprocess"]["target_spacing"])
    image_resampled = sitk_resample_to_spacing(image, spacing, is_label=False)
    # Always align mask to image grid (size/origin/direction/spacing) to preserve spatial correspondence.
    mask_resampled = sitk_resample_to_reference(mask, image_resampled, is_label=True)

    image_rs = sitk.GetArrayFromImage(image_resampled)
    lesion_rs = (sitk.GetArrayFromImage(mask_resampled) > 0).astype(np.uint8)
    rs_spacing_zyx = tuple(float(v) for v in reversed(image_resampled.GetSpacing()))
    steps["01_resampled"] = {"img": image_rs, "msk": lesion_rs, "spacing_zyx": rs_spacing_zyx}

    n4_cfg = cfg["preprocess"]["n4_bias_correction"]
    image_n4 = n4_bias_correct(
        image_resampled,
        int(n4_cfg["shrink_factor"]),
        n4_cfg["num_iters"],
    )
    image_n4_np = sitk.GetArrayFromImage(image_n4)
    steps["02_n4"] = {"img": image_n4_np, "msk": lesion_rs, "spacing_zyx": rs_spacing_zyx}

    brain_mask_cfg = cfg["preprocess"]["crop"].get("brain_mask", {})
    brain_mask_method = str(brain_mask_cfg.get("method", "simple")).lower()
    if brain_mask_method == "hdbet":
        brain_mask = make_hdbet_brain_mask(
            image_n4,
            hdbet_cmd=str(brain_mask_cfg.get("hdbet_cmd", "hd-bet")),
            mode=brain_mask_cfg.get("hdbet_mode", None),
            device=brain_mask_cfg.get("hdbet_device", None),
            tta=brain_mask_cfg.get("hdbet_tta", None),
            postprocess=brain_mask_cfg.get("hdbet_postprocess", None),
            save_mask=int(brain_mask_cfg.get("hdbet_save_mask", 1)),
        )
        required = bool(brain_mask_cfg.get("required", False))
        fallback_to_simple = bool(brain_mask_cfg.get("fallback_to_simple", True))
        if brain_mask is None or brain_mask.shape != image_n4_np.shape:
            msg = (
                f"HD-BET failed or returned invalid mask for '{image_path}'. "
                "Check hd-bet install/path or set preprocess.crop.brain_mask.fallback_to_simple=true."
            )
            if required and not fallback_to_simple:
                raise RuntimeError(msg)
            if fallback_to_simple:
                brain_mask = make_simple_brain_mask(image_n4_np)
            else:
                raise RuntimeError(msg)
    else:
        brain_mask = make_simple_brain_mask(image_n4_np)

    if cfg["preprocess"]["crop"]["enabled"]:
        image_crop, lesion_crop, brain_crop = crop_to_brain_bbox(
            image_n4_np,
            lesion_rs,
            brain_mask,
            int(cfg["preprocess"]["crop"]["margin_vox"]),
        )
    else:
        image_crop, lesion_crop, brain_crop = image_n4_np, lesion_rs, brain_mask

    # Optional skull stripping: keep only voxels inside the brain mask.
    apply_brain_mask = bool(cfg["preprocess"]["crop"].get("apply_brain_mask", False))
    brain_fg = brain_crop > 0
    bg_value = float(
        cfg["preprocess"].get(
            "background_value",
            cfg["preprocess"]["crop"].get("background_value", -5.0),
        )
    )
    if apply_brain_mask:
        # Keep only brain signal and assign explicit background value outside brain.
        image_crop = np.where(brain_fg, image_crop, bg_value).astype(np.float32)
        # Ensure labels are valid only inside the brain mask.
        lesion_crop = (lesion_crop > 0).astype(np.uint8)
        lesion_crop = lesion_crop * brain_fg.astype(np.uint8)

    steps["03_crop"] = {"img": image_crop, "msk": lesion_crop, "spacing_zyx": rs_spacing_zyx}

    intensity_cfg = cfg["preprocess"]["intensity"]
    p_lo, p_hi = intensity_cfg["clip_percentiles"]
    norm_mode = intensity_cfg.get("normalization_mode")
    if norm_mode in (None, ""):
        # Backward compatibility with older config keys.
        norm_mode = intensity_cfg.get("normalize")
    if norm_mode in (None, ""):
        norm_mode = "zscore" if bool(intensity_cfg.get("zscore", True)) else "minmax"
    norm_mode = str(norm_mode).strip().lower()
    if norm_mode not in {"zscore", "minmax"}:
        raise ValueError(
            "preprocess.intensity.normalization_mode must be one of: ['zscore', 'minmax']"
        )
    image_norm = clip_and_normalize(
        image_crop,
        brain_crop,
        float(p_lo),
        float(p_hi),
        mode=norm_mode,
    )
    if apply_brain_mask:
        image_norm = np.where(brain_fg, image_norm, bg_value).astype(np.float32)
    steps["04_norm"] = {"img": image_norm, "msk": lesion_crop, "spacing_zyx": rs_spacing_zyx}

    image_final = image_norm[None, ...].astype(np.float32)
    label_final = lesion_crop[None, ...].astype(np.uint8)
    brain_final = (brain_crop > 0).astype(np.uint8)

    return image_final, label_final, spacing, steps, brain_final
