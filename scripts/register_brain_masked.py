import json
import os
import shutil
import subprocess as sp
import sys
import tempfile
from pathlib import Path
from typing import Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medseg.preprocess import make_hdbet_brain_mask

DEFAULT_NIFTYREG_ROOT = Path(
    "/home/pgsalome/projects/git/registration_tutorial/packages/niftyreg/bin"
)
_TOTALSEG_UNSUPPORTED_SKIN_KEYS: Set[Tuple[str, str, str]] = set()
_TOTALSEG_SKIN_FALLBACK_NOTIFIED: Set[Tuple[str, str, str]] = set()


def _read_image(path: str) -> sitk.Image:
    p = Path(path)
    if p.is_dir():
        series = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(p))
        if not series:
            raise RuntimeError(f"No DICOM series files found in directory: {p}")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series)
        return reader.Execute()
    return sitk.ReadImage(str(p))


def _write_image(img: sitk.Image, out_path: Path, as_float: bool) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.Cast(img, sitk.sitkFloat32) if as_float else img
    sitk.WriteImage(out, str(out_path))
    return out_path


def _simple_brain_mask(image: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32))
    valid = np.isfinite(arr)
    if not valid.any():
        mask = np.ones_like(arr, dtype=np.uint8)
    else:
        vals = arr[valid]
        lo, hi = np.percentile(vals, [2, 98])
        arr_clip = np.clip(arr, lo, hi)
        thr = np.percentile(arr_clip[valid], 20)
        m = (arr_clip > thr).astype(np.uint8)

        labels, n = ndi.label(m)
        if n == 0:
            mask = np.ones_like(arr, dtype=np.uint8)
        else:
            sizes = ndi.sum(m, labels, index=np.arange(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            mask = (labels == keep).astype(np.uint8)
            mask = ndi.binary_closing(mask, iterations=2).astype(np.uint8)
            mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_img.CopyInformation(image)
    return sitk.Cast(mask_img, sitk.sitkUInt8)


def _largest_component(mask_zyx: np.ndarray) -> np.ndarray:
    labels, num = ndi.label(mask_zyx)
    if num == 0:
        return np.zeros_like(mask_zyx, dtype=bool)
    sizes = ndi.sum(mask_zyx, labels, index=np.arange(1, num + 1))
    keep = int(np.argmax(sizes)) + 1
    return labels == keep


def _ball_structure(radius_vox: int) -> np.ndarray:
    r = int(max(0, radius_vox))
    if r <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy + zz * zz) <= (r * r)


def _clean_binary_mask(
    mask_img: sitk.Image,
    closing_radius_vox: int = 2,
    dilate_radius_vox: int = 1,
) -> sitk.Image:
    m = sitk.GetArrayFromImage(mask_img) > 0
    if not m.any():
        out = np.zeros_like(m, dtype=np.uint8)
        out_img = sitk.GetImageFromArray(out)
        out_img.CopyInformation(mask_img)
        return sitk.Cast(out_img, sitk.sitkUInt8)

    # 1) Keep only largest component.
    m = _largest_component(m)
    # 2) Close cracks/gaps.
    if int(closing_radius_vox) > 0:
        m = ndi.binary_closing(m, structure=_ball_structure(int(closing_radius_vox)))
    # 3) Fill internal holes.
    m = ndi.binary_fill_holes(m)
    # Re-enforce single component after closing/filling.
    m = _largest_component(m)
    # 4) Optional dilation.
    if int(dilate_radius_vox) > 0:
        m = ndi.binary_dilation(m, structure=_ball_structure(int(dilate_radius_vox)))
    # Final consistency pass.
    m = _largest_component(m)
    m = ndi.binary_fill_holes(m)

    out = m.astype(np.uint8)
    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(mask_img)
    return sitk.Cast(out_img, sitk.sitkUInt8)


def _resample_binary_mask_to_ref(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    if (
        mask_img.GetSize() != ref_img.GetSize()
        or mask_img.GetSpacing() != ref_img.GetSpacing()
        or mask_img.GetOrigin() != ref_img.GetOrigin()
        or mask_img.GetDirection() != ref_img.GetDirection()
    ):
        mask_img = sitk.Resample(
            mask_img,
            ref_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    arr = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref_img)
    return sitk.Cast(out, sitk.sitkUInt8)


def _compute_robustfov_z_range(
    image: sitk.Image,
    robustfov_cmd: str = "robustfov",
    robustfov_brain_size_mm: Optional[float] = 170.0,
) -> Optional[Tuple[int, int]]:
    try:
        with tempfile.TemporaryDirectory(prefix="medseg_robustfov_") as td:
            td_path = Path(td)
            inp = td_path / "input.nii.gz"
            roi = td_path / "roi.nii.gz"

            sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), str(inp))
            cmd = [robustfov_cmd, "-i", str(inp), "-r", str(roi)]
            if robustfov_brain_size_mm is not None:
                cmd += ["-b", str(float(robustfov_brain_size_mm))]

            proc = sp.run(cmd, text=True, capture_output=True)
            if proc.returncode != 0 or (not roi.exists()):
                return None

            roi_img = sitk.ReadImage(str(roi))
            try:
                idx0 = image.TransformPhysicalPointToContinuousIndex(roi_img.GetOrigin())
            except Exception:
                return None

            z0 = int(round(float(idx0[2])))
            z1 = z0 + int(roi_img.GetSize()[2])
            z0 = max(0, z0)
            z1 = min(int(image.GetSize()[2]), z1)
            if z1 <= z0:
                return None
            return z0, z1
    except Exception:
        # Any robustfov/tool IO issue should allow caller fallback logic.
        return None


def _margin_vox_zyx(img: sitk.Image, margin_mm: float) -> Tuple[int, int, int]:
    sx, sy, sz = img.GetSpacing()
    mz = int(np.ceil(float(margin_mm) / max(float(sz), 1e-6)))
    my = int(np.ceil(float(margin_mm) / max(float(sy), 1e-6)))
    mx = int(np.ceil(float(margin_mm) / max(float(sx), 1e-6)))
    return max(0, mz), max(0, my), max(0, mx)


def _bbox_from_binary_zyx(mask_zyx: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0).tolist()
    z1, y1, x1 = (coords.max(axis=0) + 1).tolist()  # exclusive
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def _crop_sitk_zyx(
    img: sitk.Image, bbox_zyx: Tuple[int, int, int, int, int, int]
) -> sitk.Image:
    z0, z1, y0, y1, x0, x1 = bbox_zyx
    idx = [int(x0), int(y0), int(z0)]  # xyz
    size = [int(max(1, x1 - x0)), int(max(1, y1 - y0)), int(max(1, z1 - z0))]
    return sitk.RegionOfInterest(img, size=size, index=idx)


def _match_fixed_fov_to_moving_mask(
    fixed_img: sitk.Image,
    fixed_mask_img: Optional[sitk.Image],
    moving_mask_img: Optional[sitk.Image],
    margin_mm: float = 20.0,
) -> Tuple[sitk.Image, Optional[sitk.Image], Optional[dict]]:
    if moving_mask_img is None:
        return fixed_img, fixed_mask_img, None

    moving_mask_on_fixed = _resample_binary_mask_to_ref(moving_mask_img, fixed_img)
    m = sitk.GetArrayFromImage(moving_mask_on_fixed) > 0
    bbox = _bbox_from_binary_zyx(m)
    if bbox is None:
        return fixed_img, fixed_mask_img, None

    z0, z1, y0, y1, x0, x1 = bbox
    mz, my, mx = _margin_vox_zyx(fixed_img, margin_mm)
    Z, Y, X = m.shape
    z0 = max(0, z0 - mz)
    y0 = max(0, y0 - my)
    x0 = max(0, x0 - mx)
    z1 = min(Z, z1 + mz)
    y1 = min(Y, y1 + my)
    x1 = min(X, x1 + mx)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return fixed_img, fixed_mask_img, None

    crop_bbox = (z0, z1, y0, y1, x0, x1)
    fixed_crop = _crop_sitk_zyx(fixed_img, crop_bbox)
    fixed_mask_crop = _crop_sitk_zyx(fixed_mask_img, crop_bbox) if fixed_mask_img is not None else None
    info = {
        "enabled": True,
        "margin_mm": float(margin_mm),
        "bbox_zyx": [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
    }
    return fixed_crop, fixed_mask_crop, info


def _neck_trim_and_skin_contour(
    body: np.ndarray,
    neck_area_ratio: float,
    neck_min_run: int,
    neck_margin: int,
    skin_thickness: int,
    disable_neck_removal: bool,
    neck_method: str = "heuristic",
    ref_img: Optional[sitk.Image] = None,
    robustfov_cmd: str = "robustfov",
    robustfov_brain_size_mm: Optional[float] = 170.0,
    robustfov_fallback_heuristic: bool = True,
) -> np.ndarray:
    body = _largest_component(body > 0)
    body = ndi.binary_closing(body, iterations=2)
    body = ndi.binary_fill_holes(body)

    def _heuristic_trim(current_body: np.ndarray) -> np.ndarray:
        areas = current_body.reshape(current_body.shape[0], -1).sum(axis=1).astype(np.float32)
        peak = int(np.argmax(areas))
        area_thr = float(max(1.0, areas.max() * float(neck_area_ratio)))
        run_needed = int(max(1, neck_min_run))

        z_start = 0
        run = 0
        for z in range(peak, -1, -1):
            if areas[z] < area_thr:
                run += 1
                if run >= run_needed:
                    z_start = z + run_needed
                    break
            else:
                run = 0

        z_end = current_body.shape[0] - 1
        run = 0
        for z in range(peak, current_body.shape[0]):
            if areas[z] < area_thr:
                run += 1
                if run >= run_needed:
                    z_end = z - run_needed
                    break
            else:
                run = 0

        z_start = max(0, z_start - int(neck_margin))
        z_end = min(current_body.shape[0] - 1, z_end + int(neck_margin))
        if z_end > z_start:
            keep = np.zeros_like(current_body, dtype=bool)
            keep[z_start : z_end + 1] = True
            current_body = current_body & keep
            current_body = _largest_component(current_body)
        return current_body

    if not disable_neck_removal and body.any():
        used_robustfov = False
        if neck_method == "robustfov":
            if ref_img is None:
                raise RuntimeError("neck_method=robustfov requires reference image metadata.")
            z_range = _compute_robustfov_z_range(
                image=ref_img,
                robustfov_cmd=robustfov_cmd,
                robustfov_brain_size_mm=robustfov_brain_size_mm,
            )
            if z_range is not None:
                body_before_trim = body.copy()
                z0, z1 = z_range
                keep = np.zeros_like(body, dtype=bool)
                keep[z0:z1] = True
                body = body & keep
                body = _largest_component(body)
                if body.any():
                    used_robustfov = True
                elif robustfov_fallback_heuristic:
                    # Some scans get an over-aggressive robustfov slab.
                    # Restore and fall back instead of returning an empty mask.
                    body = body_before_trim
                    print(
                        "[register][warning] robustfov trim removed the full mask; "
                        "falling back to heuristic neck trimming."
                    )
                else:
                    raise RuntimeError(
                        "robustfov trim removed the full mask and fallback is disabled."
                    )
            elif not robustfov_fallback_heuristic:
                raise RuntimeError("robustfov failed and fallback is disabled.")
            else:
                print("[register][warning] robustfov failed; falling back to heuristic neck trimming.")

        if (neck_method == "heuristic") or (neck_method == "robustfov" and not used_robustfov):
            body = _heuristic_trim(body)

    eroded = ndi.binary_erosion(body, iterations=int(max(1, skin_thickness)))
    contour = body & (~eroded)
    if not contour.any():
        contour = body
    return contour.astype(np.uint8)


def _skin_contour_from_mask_image(
    mask_img: sitk.Image,
    ref_img: sitk.Image,
    neck_area_ratio: float = 0.35,
    neck_min_run: int = 4,
    neck_margin: int = 2,
    skin_thickness: int = 3,
    disable_neck_removal: bool = False,
    neck_method: str = "heuristic",
    robustfov_cmd: str = "robustfov",
    robustfov_brain_size_mm: Optional[float] = 170.0,
    robustfov_fallback_heuristic: bool = True,
) -> sitk.Image:
    m = _resample_binary_mask_to_ref(mask_img, ref_img)
    body = sitk.GetArrayFromImage(m) > 0
    contour = _neck_trim_and_skin_contour(
        body=body,
        neck_area_ratio=neck_area_ratio,
        neck_min_run=neck_min_run,
        neck_margin=neck_margin,
        skin_thickness=skin_thickness,
        disable_neck_removal=disable_neck_removal,
        neck_method=neck_method,
        ref_img=ref_img,
        robustfov_cmd=robustfov_cmd,
        robustfov_brain_size_mm=robustfov_brain_size_mm,
        robustfov_fallback_heuristic=robustfov_fallback_heuristic,
    )
    out = sitk.GetImageFromArray(contour.astype(np.uint8))
    out.CopyInformation(ref_img)
    return sitk.Cast(out, sitk.sitkUInt8)


def _locate_skin_mask(seg_dir: Path) -> Optional[Path]:
    if not seg_dir.exists():
        return None
    direct = seg_dir / "skin.nii.gz"
    if direct.exists():
        return direct
    candidates = sorted(seg_dir.glob("*skin*.nii.gz"))
    return candidates[0] if candidates else None


def _derive_skin_from_body_outputs(seg_dir: Path) -> Optional[Path]:
    body_paths = [seg_dir / "body_trunc.nii.gz", seg_dir / "body_extremities.nii.gz"]
    present = [p for p in body_paths if p.exists()]
    if not present:
        return None

    ref = sitk.ReadImage(str(present[0]))
    body = np.zeros(tuple(reversed(ref.GetSize())), dtype=bool)
    for p in present:
        img = sitk.ReadImage(str(p))
        img = _resample_binary_mask_to_ref(img, ref)
        body |= sitk.GetArrayFromImage(img) > 0

    if not body.any():
        return None

    out_img = sitk.GetImageFromArray(body.astype(np.uint8))
    out_img.CopyInformation(ref)
    out_path = seg_dir / "skin.nii.gz"
    sitk.WriteImage(sitk.Cast(out_img, sitk.sitkUInt8), str(out_path))
    return out_path


def _totalseg_skin_key(cmd_name: str, task: str, roi_subset: str) -> Tuple[str, str, str]:
    return (str(cmd_name), str(task), str(roi_subset))


def _stderr_suggests_unsupported_roi(stderr: str, roi_subset: str) -> bool:
    s = (stderr or "").strip().lower()
    roi = str(roi_subset).strip().lower()
    if not s or not roi:
        return False

    roi_mentioned = (f"'{roi}'" in s) or (f"\"{roi}\"" in s) or (roi in s)
    if not roi_mentioned:
        return False

    markers = (
        "roi_subset",
        "roi subset",
        "unsupported",
        "not supported",
        "not available",
        "invalid",
        "unknown",
        "not in",
    )
    return any(m in s for m in markers)


def _print_skin_fallback_info_once(key: Tuple[str, str, str], fb_task: str) -> None:
    if key in _TOTALSEG_SKIN_FALLBACK_NOTIFIED:
        return
    _TOTALSEG_SKIN_FALLBACK_NOTIFIED.add(key)
    print(
        "[register][info] TotalSegmentator ROI 'skin' unsupported in this setup; "
        f"using task '{fb_task}' body masks to derive skin contour."
    )


def _run_totalsegmentator_skin(
    input_nii: Path,
    out_dir: Path,
    cmd_name: str = "TotalSegmentator",
    task: str = "total",
    roi_subset: str = "skin",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _locate_skin_mask(out_dir)
    if existing is not None:
        return existing

    key = _totalseg_skin_key(cmd_name, task, roi_subset)
    skip_primary_skin_call = key in _TOTALSEG_UNSUPPORTED_SKIN_KEYS
    primary_err = ""
    primary_rc: Optional[int] = None
    primary_cmd = [
        cmd_name,
        "-i",
        str(input_nii),
        "-o",
        str(out_dir),
        "-ta",
        str(task),
        "--roi_subset",
        str(roi_subset),
    ]
    unsupported_skin_roi = skip_primary_skin_call

    if not skip_primary_skin_call:
        proc = sp.run(primary_cmd, text=True, capture_output=True)
        primary_rc = int(proc.returncode)
        if proc.returncode == 0:
            produced = _locate_skin_mask(out_dir)
            if produced is not None:
                return produced
            derived = _derive_skin_from_body_outputs(out_dir)
            if derived is not None:
                return derived
            raise RuntimeError(
                "TotalSegmentator finished but skin/body masks were not found in '{}'.".format(
                    out_dir
                )
            )

        primary_err = (proc.stderr or "").strip()
        if _stderr_suggests_unsupported_roi(primary_err, roi_subset):
            _TOTALSEG_UNSUPPORTED_SKIN_KEYS.add(key)
            unsupported_skin_roi = True

    # Some TotalSegmentator versions do not expose ROI 'skin'.
    # Fallback: run body/body_mr and derive skin from body masks.
    fallback_tasks = ["body", "body_mr"]
    for fb_task in fallback_tasks:
        fb_cmd = [
            cmd_name,
            "-i",
            str(input_nii),
            "-o",
            str(out_dir),
            "-ta",
            fb_task,
        ]
        fb_proc = sp.run(fb_cmd, text=True, capture_output=True)
        if fb_proc.returncode != 0:
            continue
        derived = _derive_skin_from_body_outputs(out_dir)
        if derived is not None:
            if unsupported_skin_roi:
                _print_skin_fallback_info_once(key, fb_task)
            return derived

    if skip_primary_skin_call:
        raise RuntimeError(
            "TotalSegmentator skin ROI was previously marked unsupported for cmd/task '{} {}'; "
            "fallback body tasks also failed in '{}'.".format(cmd_name, task, out_dir)
        )
    raise RuntimeError(
        "TotalSegmentator failed (code {}). cmd={} stderr={}".format(
            primary_rc, " ".join(primary_cmd), primary_err
        )
    )


def _skin_contour_mask(
    image: sitk.Image,
    neck_area_ratio: float = 0.35,
    neck_min_run: int = 4,
    neck_margin: int = 2,
    skin_thickness: int = 3,
    disable_neck_removal: bool = False,
    neck_method: str = "heuristic",
    robustfov_cmd: str = "robustfov",
    robustfov_brain_size_mm: Optional[float] = 170.0,
    robustfov_fallback_heuristic: bool = True,
) -> sitk.Image:
    arr = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32))
    valid = np.isfinite(arr)
    if not valid.any():
        out = np.ones_like(arr, dtype=np.uint8)
    else:
        vals = arr[valid]
        lo, hi = np.percentile(vals, [1, 99])
        arr_clip = np.clip(arr, lo, hi)
        # Foreground body threshold (works for both CT/MR backgrounds).
        thr = np.percentile(arr_clip[valid], 10)
        body = arr_clip > thr
        body = _largest_component(body)
        body = ndi.binary_closing(body, iterations=2)
        body = ndi.binary_fill_holes(body)

        out = _neck_trim_and_skin_contour(
            body=body,
            neck_area_ratio=neck_area_ratio,
            neck_min_run=neck_min_run,
            neck_margin=neck_margin,
            skin_thickness=skin_thickness,
            disable_neck_removal=disable_neck_removal,
            neck_method=neck_method,
            ref_img=image,
            robustfov_cmd=robustfov_cmd,
            robustfov_brain_size_mm=robustfov_brain_size_mm,
            robustfov_fallback_heuristic=robustfov_fallback_heuristic,
        )

    mask_img = sitk.GetImageFromArray(out)
    mask_img.CopyInformation(image)
    return sitk.Cast(mask_img, sitk.sitkUInt8)


def _prepare_mask_on_ref(
    ref_img: sitk.Image,
    mask_path: Optional[str],
    mode: str,
    ref_nii_for_totalseg: Optional[Path],
    totalseg_out_dir: Optional[Path],
    totalseg_cmd: str,
    totalseg_task: str,
    totalseg_roi_subset: str,
    totalseg_fallback_intensity: bool,
    hdbet_cmd: str,
    hdbet_mode: Optional[str],
    hdbet_device: Optional[str],
    hdbet_tta: Optional[int],
    hdbet_postprocess: Optional[int],
    hdbet_fallback_simple: bool,
    neck_area_ratio: float,
    neck_min_run: int,
    neck_margin: int,
    skin_thickness: int,
    disable_neck_removal: bool,
    neck_method: str,
    robustfov_cmd: str,
    robustfov_brain_size_mm: Optional[float],
    robustfov_fallback_heuristic: bool,
) -> Tuple[Optional[sitk.Image], Optional[Path]]:
    if mask_path:
        m = _read_image(mask_path)
        if mode == "skin":
            return (
                _skin_contour_from_mask_image(
                    m,
                    ref_img,
                    neck_area_ratio=neck_area_ratio,
                    neck_min_run=neck_min_run,
                    neck_margin=neck_margin,
                    skin_thickness=skin_thickness,
                    disable_neck_removal=disable_neck_removal,
                    neck_method=neck_method,
                    robustfov_cmd=robustfov_cmd,
                    robustfov_brain_size_mm=robustfov_brain_size_mm,
                    robustfov_fallback_heuristic=robustfov_fallback_heuristic,
                ),
                Path(mask_path),
            )
        return _resample_binary_mask_to_ref(m, ref_img), Path(mask_path)

    if mode == "none":
        return None, None
    if mode == "hdbet":
        arr = make_hdbet_brain_mask(
            image=ref_img,
            hdbet_cmd=hdbet_cmd,
            mode=hdbet_mode,
            device=hdbet_device,
            tta=hdbet_tta,
            postprocess=hdbet_postprocess,
            save_mask=1,
        )
        if arr is not None and arr.shape == tuple(reversed(ref_img.GetSize())):
            mask_img = sitk.GetImageFromArray((arr > 0).astype(np.uint8))
            mask_img.CopyInformation(ref_img)
            return sitk.Cast(mask_img, sitk.sitkUInt8), None

        if hdbet_fallback_simple:
            print("[register][warning] HD-BET failed, falling back to simple mask.")
            return _simple_brain_mask(ref_img), None
        raise RuntimeError("HD-BET failed to produce a valid brain mask.")
    if mode == "skin":
        if ref_nii_for_totalseg is None or totalseg_out_dir is None:
            raise RuntimeError("Internal error: TotalSegmentator inputs not set for skin mode.")
        try:
            raw_skin_path = _run_totalsegmentator_skin(
                input_nii=ref_nii_for_totalseg,
                out_dir=totalseg_out_dir,
                cmd_name=totalseg_cmd,
                task=totalseg_task,
                roi_subset=totalseg_roi_subset,
            )
            raw_skin_img = _read_image(str(raw_skin_path))
            contour_img = _skin_contour_from_mask_image(
                raw_skin_img,
                ref_img,
                neck_area_ratio=neck_area_ratio,
                neck_min_run=neck_min_run,
                neck_margin=neck_margin,
                skin_thickness=skin_thickness,
                disable_neck_removal=disable_neck_removal,
                neck_method=neck_method,
                robustfov_cmd=robustfov_cmd,
                robustfov_brain_size_mm=robustfov_brain_size_mm,
                robustfov_fallback_heuristic=robustfov_fallback_heuristic,
            )
            return contour_img, raw_skin_path
        except Exception as e:
            if not totalseg_fallback_intensity:
                raise RuntimeError(f"TotalSegmentator skin mask failed: {e}")
            print(
                "[register][warning] TotalSegmentator skin failed, "
                "falling back to intensity skin contour: {}".format(e)
            )
            return (
                _skin_contour_mask(
                    ref_img,
                    neck_area_ratio=neck_area_ratio,
                    neck_min_run=neck_min_run,
                    neck_margin=neck_margin,
                    skin_thickness=skin_thickness,
                    disable_neck_removal=disable_neck_removal,
                    neck_method=neck_method,
                    robustfov_cmd=robustfov_cmd,
                    robustfov_brain_size_mm=robustfov_brain_size_mm,
                    robustfov_fallback_heuristic=robustfov_fallback_heuristic,
                ),
                None,
            )
    if mode == "simple":
        return _simple_brain_mask(ref_img), None
    raise ValueError(f"Unknown mask mode: {mode}")


def _niftyreg_bins(root: Optional[str], reg_aladin: Optional[str], reg_f3d: Optional[str], reg_resample: Optional[str]):
    bin_root = Path(root) if root else DEFAULT_NIFTYREG_ROOT
    aladin = Path(reg_aladin) if reg_aladin else (bin_root / "reg_aladin")
    f3d = Path(reg_f3d) if reg_f3d else (bin_root / "reg_f3d")
    resample = Path(reg_resample) if reg_resample else (bin_root / "reg_resample")

    for p in (aladin, f3d, resample):
        if not p.exists():
            raise FileNotFoundError(f"NiftyReg binary not found: {p}")
    return aladin, f3d, resample


def _run(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(str(c) for c in cmd) + "\n\n")
    proc = sp.run(cmd, text=True, capture_output=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(proc.stdout or "")
        f.write("\n\nSTDERR:\n")
        f.write(proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}")


def _save_qa_png(fixed_nii: Path, moving_orig_nii: Path, moving_stage1_nii: Path, moving_final_nii: Path, out_png: Path):
    fixed = sitk.ReadImage(str(fixed_nii))
    moving_orig = sitk.ReadImage(str(moving_orig_nii))
    moving_stage1 = sitk.ReadImage(str(moving_stage1_nii))
    moving_final = sitk.ReadImage(str(moving_final_nii))

    moving_before = sitk.Resample(
        moving_orig,
        fixed,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    f = sitk.GetArrayFromImage(fixed).astype(np.float32)
    b = sitk.GetArrayFromImage(moving_before).astype(np.float32)
    s1 = sitk.GetArrayFromImage(moving_stage1).astype(np.float32)
    s2 = sitk.GetArrayFromImage(moving_final).astype(np.float32)
    z = f.shape[0] // 2

    def to_u8(x2d):
        vals = x2d[np.isfinite(x2d)]
        if vals.size == 0:
            vals = np.array([0.0], dtype=np.float32)
        lo, hi = np.percentile(vals, [1, 99])
        x = np.clip(x2d, lo, hi)
        x = (x - lo) / (hi - lo + 1e-8)
        return (255.0 * x).astype(np.uint8)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
    ax[0].imshow(to_u8(f[z]), cmap="gray")
    ax[0].set_title("Fixed (CT)")
    ax[0].axis("off")

    ax[1].imshow(to_u8(b[z]), cmap="gray")
    ax[1].set_title("Before")
    ax[1].axis("off")

    ax[2].imshow(to_u8(s1[z]), cmap="gray")
    ax[2].set_title("Stage-1 reg_aladin")
    ax[2].axis("off")

    ax[3].imshow(to_u8(s2[z]), cmap="gray")
    ax[3].set_title("Stage-2 reg_f3d")
    ax[3].axis("off")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def _save_qa_step_pngs(
    fixed_nii: Path,
    moving_orig_nii: Path,
    moving_stage1_nii: Path,
    moving_final_nii: Path,
    out_dir: Path,
) -> list:
    fixed = sitk.ReadImage(str(fixed_nii))
    moving_orig = sitk.ReadImage(str(moving_orig_nii))
    moving_stage1 = sitk.ReadImage(str(moving_stage1_nii))
    moving_final = sitk.ReadImage(str(moving_final_nii))

    moving_before = sitk.Resample(
        moving_orig,
        fixed,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    fixed_zyx = sitk.GetArrayFromImage(fixed).astype(np.float32)
    before_zyx = sitk.GetArrayFromImage(moving_before).astype(np.float32)
    stage1_zyx = sitk.GetArrayFromImage(moving_stage1).astype(np.float32)
    final_zyx = sitk.GetArrayFromImage(moving_final).astype(np.float32)
    z = fixed_zyx.shape[0] // 2

    def to_u8(x2d: np.ndarray) -> np.ndarray:
        vals = x2d[np.isfinite(x2d)]
        if vals.size == 0:
            vals = np.array([0.0], dtype=np.float32)
        lo, hi = np.percentile(vals, [1, 99])
        x = np.clip(x2d, lo, hi)
        x = (x - lo) / (hi - lo + 1e-8)
        return (255.0 * x).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_slice = fixed_zyx[z]
    steps = [
        ("00_fixed_reference", fixed_slice),
        ("01_before_resampled", before_zyx[z]),
        ("02_stage1_reg_aladin", stage1_zyx[z]),
        ("03_stage2_reg_f3d", final_zyx[z]),
    ]

    out_paths = []
    for step_name, current_slice in steps:
        diff = np.abs(current_slice.astype(np.float32) - fixed_slice.astype(np.float32))
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
        ax[0].imshow(to_u8(fixed_slice), cmap="gray")
        ax[0].set_title("Fixed")
        ax[0].axis("off")
        ax[1].imshow(to_u8(current_slice), cmap="gray")
        ax[1].set_title(step_name)
        ax[1].axis("off")
        ax[2].imshow(to_u8(diff), cmap="magma")
        ax[2].set_title("Abs diff vs fixed")
        ax[2].axis("off")
        fig.tight_layout()
        out_path = out_dir / f"{step_name}.png"
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        out_paths.append(str(out_path))
    return out_paths


def main(
    fixed: str,
    moving: str,
    out_dir: str,
    fixed_mask: Optional[str] = None,
    moving_mask: Optional[str] = None,
    moving_label: Optional[str] = None,
    mask_mode: str = "skin",
    totalseg_cmd: str = "TotalSegmentator",
    totalseg_task: str = "total",
    totalseg_roi_subset: str = "skin",
    totalseg_fallback_intensity: bool = False,
    hdbet_cmd: str = "hd-bet",
    hdbet_mode: Optional[str] = "accurate",
    hdbet_device: Optional[str] = "cpu",
    hdbet_tta: Optional[int] = 0,
    hdbet_postprocess: Optional[int] = 1,
    hdbet_fallback_simple: bool = True,
    neck_area_ratio: float = 0.35,
    neck_min_run: int = 4,
    neck_margin: int = 2,
    skin_thickness: int = 3,
    disable_neck_removal: bool = False,
    neck_method: str = "heuristic",
    robustfov_cmd: str = "robustfov",
    robustfov_brain_size_mm: Optional[float] = 170.0,
    robustfov_fallback_heuristic: bool = True,
    niftyreg_root: Optional[str] = None,
    reg_aladin: Optional[str] = None,
    reg_f3d: Optional[str] = None,
    reg_resample: Optional[str] = None,
    cores: int = 24,
    ln: int = 5,
    maxit: int = 1000,
    sx: int = 10,
    disable_stage2: bool = False,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    aladin_bin, f3d_bin, resample_bin = _niftyreg_bins(
        root=niftyreg_root,
        reg_aladin=reg_aladin,
        reg_f3d=reg_f3d,
        reg_resample=reg_resample,
    )

    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)

    fixed_img = _read_image(fixed)
    moving_img = _read_image(moving)

    # Persist source volumes in output cache for audit/reproducibility.
    fixed_original_nii = _write_image(fixed_img, out / "fixed_original.nii.gz", as_float=False)
    moving_original_nii = _write_image(moving_img, out / "moving_original.nii.gz", as_float=False)

    fixed_full_nii = _write_image(fixed_img, work / "fixed_float32.nii.gz", as_float=True)
    moving_nii = _write_image(moving_img, work / "moving_float32.nii.gz", as_float=True)

    moving_mask_img, moving_skin_raw_path = _prepare_mask_on_ref(
        moving_img,
        moving_mask,
        mask_mode,
        ref_nii_for_totalseg=moving_nii,
        totalseg_out_dir=work / "totalseg_moving",
        totalseg_cmd=totalseg_cmd,
        totalseg_task=totalseg_task,
        totalseg_roi_subset=totalseg_roi_subset,
        totalseg_fallback_intensity=totalseg_fallback_intensity,
        hdbet_cmd=hdbet_cmd,
        hdbet_mode=hdbet_mode,
        hdbet_device=hdbet_device,
        hdbet_tta=hdbet_tta,
        hdbet_postprocess=hdbet_postprocess,
        hdbet_fallback_simple=hdbet_fallback_simple,
        neck_area_ratio=neck_area_ratio,
        neck_min_run=neck_min_run,
        neck_margin=neck_margin,
        skin_thickness=skin_thickness,
        disable_neck_removal=disable_neck_removal,
        neck_method=neck_method,
        robustfov_cmd=robustfov_cmd,
        robustfov_brain_size_mm=robustfov_brain_size_mm,
        robustfov_fallback_heuristic=robustfov_fallback_heuristic,
    )
    fixed_mask_img, fixed_skin_raw_path = _prepare_mask_on_ref(
        fixed_img,
        fixed_mask,
        mask_mode,
        ref_nii_for_totalseg=fixed_full_nii,
        totalseg_out_dir=work / "totalseg_fixed",
        totalseg_cmd=totalseg_cmd,
        totalseg_task=totalseg_task,
        totalseg_roi_subset=totalseg_roi_subset,
        totalseg_fallback_intensity=totalseg_fallback_intensity,
        hdbet_cmd=hdbet_cmd,
        hdbet_mode=hdbet_mode,
        hdbet_device=hdbet_device,
        hdbet_tta=hdbet_tta,
        hdbet_postprocess=hdbet_postprocess,
        hdbet_fallback_simple=hdbet_fallback_simple,
        neck_area_ratio=neck_area_ratio,
        neck_min_run=neck_min_run,
        neck_margin=neck_margin,
        skin_thickness=skin_thickness,
        disable_neck_removal=disable_neck_removal,
        neck_method=neck_method,
        robustfov_cmd=robustfov_cmd,
        robustfov_brain_size_mm=robustfov_brain_size_mm,
        robustfov_fallback_heuristic=robustfov_fallback_heuristic,
    )

    fixed_mask_nii = None
    moving_mask_nii = None
    # Final mask cleanup for registration robustness:
    # largest component + closing + hole fill + optional dilation.
    if fixed_mask_img is not None:
        fixed_mask_img = _clean_binary_mask(
            fixed_mask_img,
            closing_radius_vox=2,
            dilate_radius_vox=1,
        )
    if moving_mask_img is not None:
        moving_mask_img = _clean_binary_mask(
            moving_mask_img,
            closing_radius_vox=2,
            dilate_radius_vox=1,
        )

    # Match FOV before optimization: crop fixed image/mask to moving-mask ROI (+ margin).
    fov_match_info = None
    fixed_reg_img, fixed_reg_mask_img, fov_match_info = _match_fixed_fov_to_moving_mask(
        fixed_img=fixed_img,
        fixed_mask_img=fixed_mask_img,
        moving_mask_img=moving_mask_img,
        margin_mm=20.0,
    )
    fixed_nii = _write_image(
        fixed_reg_img,
        work / "fixed_for_registration_float32.nii.gz",
        as_float=True,
    )
    fixed_mask_img = fixed_reg_mask_img
    if fixed_mask_img is not None:
        fixed_mask_nii = _write_image(fixed_mask_img, out / "fixed_mask_used.nii.gz", as_float=False)
    if moving_mask_img is not None:
        moving_mask_nii = _write_image(moving_mask_img, out / "moving_mask_used.nii.gz", as_float=False)
    if fixed_skin_raw_path is not None:
        _write_image(_read_image(str(fixed_skin_raw_path)), out / "fixed_skin_raw_totalseg.nii.gz", as_float=False)
    if moving_skin_raw_path is not None:
        _write_image(_read_image(str(moving_skin_raw_path)), out / "moving_skin_raw_totalseg.nii.gz", as_float=False)

    stage1_res = out / "moving_stage1_registered_to_fixed.nii.gz"
    stage1_aff = out / "stage1_affine.txt"

    cmd_stage1 = [
        str(aladin_bin),
        "-ln", str(ln),
        "-omp", str(cores),
        "-ref", str(fixed_nii),
        "-flo", str(moving_nii),
        "-res", str(stage1_res),
        "-aff", str(stage1_aff),
    ]
    if fixed_mask_nii is not None:
        cmd_stage1 += ["-rmask", str(fixed_mask_nii)]
    if moving_mask_nii is not None:
        cmd_stage1 += ["-fmask", str(moving_mask_nii)]
    _run(cmd_stage1, out / "logs" / "stage1_reg_aladin.log")

    final_img = out / "moving_registered_to_fixed.nii.gz"
    stage2_cpp = out / "stage2_deformation_cpp.nii.gz"

    if disable_stage2:
        shutil.copyfile(stage1_res, final_img)
    else:
        cmd_stage2 = [
            str(f3d_bin),
            "-ln", str(ln),
            "-omp", str(cores),
            "-maxit", str(maxit),
            "-sx", str(sx),
            "-ref", str(fixed_nii),
            "-flo", str(moving_nii),
            "-aff", str(stage1_aff),
            "-cpp", str(stage2_cpp),
            "-res", str(final_img),
        ]
        if fixed_mask_nii is not None:
            cmd_stage2 += ["-rmask", str(fixed_mask_nii)]
        if moving_mask_nii is not None:
            cmd_stage2 += ["-fmask", str(moving_mask_nii)]

        _run(cmd_stage2, out / "logs" / "stage2_reg_f3d.log")

    if moving_label:
        moving_label_img = _read_image(moving_label)
        if (
            moving_label_img.GetSize() != moving_img.GetSize()
            or moving_label_img.GetSpacing() != moving_img.GetSpacing()
            or moving_label_img.GetOrigin() != moving_img.GetOrigin()
            or moving_label_img.GetDirection() != moving_img.GetDirection()
        ):
            moving_label_img = sitk.Resample(
                moving_label_img,
                moving_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
        moving_label_nii = _write_image(moving_label_img, work / "moving_label_uint8.nii.gz", as_float=False)

        label_stage1 = out / "moving_label_stage1_registered_to_fixed.nii.gz"
        cmd_lab_1 = [
            str(resample_bin),
            "-ref", str(fixed_nii),
            "-flo", str(moving_label_nii),
            "-res", str(label_stage1),
            "-aff", str(stage1_aff),
            "-inter", "0",
        ]
        _run(cmd_lab_1, out / "logs" / "label_stage1_reg_resample.log")

        label_final = out / "moving_label_registered_to_fixed.nii.gz"
        if disable_stage2:
            shutil.copyfile(label_stage1, label_final)
        else:
            cmd_lab_2 = [
                str(resample_bin),
                "-ref", str(fixed_nii),
                "-flo", str(label_stage1),
                "-res", str(label_final),
                "-cpp", str(stage2_cpp),
                "-inter", "0",
            ]
            _run(cmd_lab_2, out / "logs" / "label_stage2_reg_resample.log")

    qa_png = out / "qa_registration_before_after.png"
    _save_qa_png(
        fixed_nii=fixed_nii,
        moving_orig_nii=moving_nii,
        moving_stage1_nii=stage1_res,
        moving_final_nii=final_img,
        out_png=qa_png,
    )
    qa_steps_dir = out / "qa_steps"
    qa_step_pngs = _save_qa_step_pngs(
        fixed_nii=fixed_nii,
        moving_orig_nii=moving_nii,
        moving_stage1_nii=stage1_res,
        moving_final_nii=final_img,
        out_dir=qa_steps_dir,
    )

    summary = {
        "fixed": fixed,
        "moving": moving,
        "out_dir": str(out),
        "niftyreg": {
            "reg_aladin": str(aladin_bin),
            "reg_f3d": str(f3d_bin),
            "reg_resample": str(resample_bin),
        },
        "params": {
            "cores": int(cores),
            "ln": int(ln),
            "maxit": int(maxit),
            "sx": int(sx),
            "disable_stage2": bool(disable_stage2),
            "mask_mode": mask_mode,
            "totalseg_cmd": totalseg_cmd,
            "totalseg_task": totalseg_task,
            "totalseg_roi_subset": totalseg_roi_subset,
            "totalseg_fallback_intensity": bool(totalseg_fallback_intensity),
            "hdbet_cmd": hdbet_cmd,
            "hdbet_mode": hdbet_mode,
            "hdbet_device": hdbet_device,
            "hdbet_tta": hdbet_tta,
            "hdbet_postprocess": hdbet_postprocess,
            "hdbet_fallback_simple": bool(hdbet_fallback_simple),
            "neck_area_ratio": float(neck_area_ratio),
            "neck_min_run": int(neck_min_run),
            "neck_margin": int(neck_margin),
            "skin_thickness": int(skin_thickness),
            "disable_neck_removal": bool(disable_neck_removal),
            "neck_method": neck_method,
            "robustfov_cmd": robustfov_cmd,
            "robustfov_brain_size_mm": robustfov_brain_size_mm,
            "robustfov_fallback_heuristic": bool(robustfov_fallback_heuristic),
            "mask_cleanup": {
                "enabled": True,
                "largest_component": True,
                "closing_radius_vox": 2,
                "fill_holes": True,
                "dilate_radius_vox": 1,
            },
            "fov_match": (
                fov_match_info
                if fov_match_info is not None
                else {"enabled": False}
            ),
        },
        "outputs": {
            "fixed_original": str(fixed_original_nii),
            "moving_original": str(moving_original_nii),
            "fixed_for_registration": str(fixed_nii),
            "fixed_skin_raw_totalseg": None if fixed_skin_raw_path is None else str(fixed_skin_raw_path),
            "moving_skin_raw_totalseg": None if moving_skin_raw_path is None else str(moving_skin_raw_path),
            "stage1_image": str(stage1_res),
            "stage1_affine": str(stage1_aff),
            "stage2_cpp": None if disable_stage2 else str(stage2_cpp),
            "final_image": str(final_img),
            "qa_png": str(qa_png),
            "qa_steps_dir": str(qa_steps_dir),
            "qa_step_pngs": qa_step_pngs,
        },
    }
    with open(out / "registration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Registration complete.")
    print(f"Fixed:  {fixed}")
    print(f"Moving: {moving}")
    print(f"Output: {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True, help="Reference/fixed image path (NIfTI or DICOM dir)")
    parser.add_argument("--moving", required=True, help="Moving image path (NIfTI or DICOM dir)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fixed-mask", default=None, help="Optional fixed brain mask (NIfTI or DICOM dir)")
    parser.add_argument("--moving-mask", default=None, help="Optional moving brain mask (NIfTI or DICOM dir)")
    parser.add_argument("--moving-label", default=None, help="Optional moving label/mask to resample")

    parser.add_argument("--mask-mode", default="skin", choices=["skin", "hdbet", "simple", "none"])
    parser.add_argument("--totalseg-cmd", default="TotalSegmentator")
    parser.add_argument("--totalseg-task", default="total")
    parser.add_argument("--totalseg-roi-subset", default="skin")
    parser.add_argument(
        "--totalseg-fallback-intensity",
        action="store_true",
        help="If TotalSegmentator fails in skin mode, fall back to intensity-based contour.",
    )
    parser.add_argument("--hdbet-cmd", default="hd-bet")
    parser.add_argument("--hdbet-mode", default="accurate")
    parser.add_argument("--hdbet-device", default="cpu")
    parser.add_argument("--hdbet-tta", type=int, default=0)
    parser.add_argument("--hdbet-postprocess", type=int, default=1)
    parser.add_argument("--no-hdbet-fallback-simple", action="store_true")
    parser.add_argument("--neck-area-ratio", type=float, default=0.35)
    parser.add_argument("--neck-min-run", type=int, default=4)
    parser.add_argument("--neck-margin", type=int, default=2)
    parser.add_argument("--skin-thickness", type=int, default=3)
    parser.add_argument("--disable-neck-removal", action="store_true")
    parser.add_argument("--neck-method", default="heuristic", choices=["heuristic", "robustfov"])
    parser.add_argument("--robustfov-cmd", default="robustfov")
    parser.add_argument(
        "--robustfov-brain-size-mm",
        type=float,
        default=170.0,
        help="Passed to robustfov -b (ignored for heuristic neck method).",
    )
    parser.add_argument(
        "--no-robustfov-fallback-heuristic",
        action="store_true",
        help="If neck_method=robustfov and robustfov fails, do not fall back to heuristic trimming.",
    )

    parser.add_argument("--niftyreg-root", default=str(DEFAULT_NIFTYREG_ROOT))
    parser.add_argument("--reg-aladin", default=None)
    parser.add_argument("--reg-f3d", default=None)
    parser.add_argument("--reg-resample", default=None)

    parser.add_argument("--cores", type=int, default=24)
    parser.add_argument("--ln", type=int, default=5)
    parser.add_argument("--maxit", type=int, default=1000)
    parser.add_argument("--sx", type=int, default=10)
    parser.add_argument("--disable-stage2", action="store_true")

    args = parser.parse_args()

    main(
        fixed=args.fixed,
        moving=args.moving,
        out_dir=args.out_dir,
        fixed_mask=args.fixed_mask,
        moving_mask=args.moving_mask,
        moving_label=args.moving_label,
        mask_mode=args.mask_mode,
        totalseg_cmd=args.totalseg_cmd,
        totalseg_task=args.totalseg_task,
        totalseg_roi_subset=args.totalseg_roi_subset,
        totalseg_fallback_intensity=args.totalseg_fallback_intensity,
        hdbet_cmd=args.hdbet_cmd,
        hdbet_mode=args.hdbet_mode,
        hdbet_device=args.hdbet_device,
        hdbet_tta=args.hdbet_tta,
        hdbet_postprocess=args.hdbet_postprocess,
        hdbet_fallback_simple=not args.no_hdbet_fallback_simple,
        neck_area_ratio=args.neck_area_ratio,
        neck_min_run=args.neck_min_run,
        neck_margin=args.neck_margin,
        skin_thickness=args.skin_thickness,
        disable_neck_removal=args.disable_neck_removal,
        neck_method=args.neck_method,
        robustfov_cmd=args.robustfov_cmd,
        robustfov_brain_size_mm=args.robustfov_brain_size_mm,
        robustfov_fallback_heuristic=not args.no_robustfov_fallback_heuristic,
        niftyreg_root=args.niftyreg_root,
        reg_aladin=args.reg_aladin,
        reg_f3d=args.reg_f3d,
        reg_resample=args.reg_resample,
        cores=args.cores,
        ln=args.ln,
        maxit=args.maxit,
        sx=args.sx,
        disable_stage2=args.disable_stage2,
    )
