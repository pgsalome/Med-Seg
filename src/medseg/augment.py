import random
from typing import Dict

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandCoarseDropoutd,
    RandFlipd,
    RandGaussianNoised,
)

try:
    from monai.transforms import RandGammaD as _RandGammaTransform
except ImportError:
    try:
        from monai.transforms import RandGammad as _RandGammaTransform
    except ImportError:
        _RandGammaTransform = None


class ThickSliceSimd:
    def __init__(self, keys=("image",), prob=0.25, z_factors=(2, 4)):
        self.keys = keys
        self.prob = prob
        self.z_factors = tuple(z_factors)

    def __call__(self, data: Dict):
        if random.random() > self.prob:
            return data

        zf = random.choice(self.z_factors)
        for key in self.keys:
            x = data[key]
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            to_torch = isinstance(x, torch.Tensor)

            channels, z_dim, y_dim, x_dim = x_np.shape
            _ = channels, y_dim, x_dim
            z_small = max(1, z_dim // zf)
            idx_ds = np.linspace(0, z_dim - 1, z_small).round().astype(int)
            x_ds = x_np[:, idx_ds, :, :]
            idx_up = np.linspace(0, z_small - 1, z_dim).round().astype(int)
            x_us = x_ds[:, idx_up, :, :]

            if to_torch:
                data[key] = torch.from_numpy(x_us).to(x.device)
            else:
                data[key] = x_us
        return data


def _coarse_dropout_kwargs(intensity_cfg: dict) -> dict:
    size_cfg = intensity_cfg["coarse_dropout_size"]

    max_spatial_size = None
    if isinstance(size_cfg, (int, float)):
        val = max(1, int(size_cfg))
        spatial_size = (val, val, val)
    elif isinstance(size_cfg, (list, tuple)):
        vals = [int(v) for v in size_cfg]
        if len(vals) == 1:
            val = max(1, vals[0])
            spatial_size = (val, val, val)
        elif len(vals) == 2:
            lo, hi = vals
            if lo > hi:
                lo, hi = hi, lo
            lo = max(1, lo)
            hi = max(lo, hi)
            spatial_size = (lo, lo, lo)
            max_spatial_size = (hi, hi, hi)
        elif len(vals) == 3:
            spatial_size = tuple(max(1, v) for v in vals)
        else:
            raise ValueError(
                "augment.intensity.coarse_dropout_size must be int, [s], [lo,hi], or [z,y,x]."
            )
    else:
        raise TypeError("augment.intensity.coarse_dropout_size has unsupported type.")

    kwargs = {
        "keys": ["image"],
        "prob": float(intensity_cfg["coarse_dropout_prob"]),
        "holes": int(intensity_cfg["coarse_dropout_holes"]),
        "spatial_size": spatial_size,
        "fill_value": 0.0,
    }
    if max_spatial_size is not None:
        kwargs["max_spatial_size"] = max_spatial_size
    return kwargs


def build_train_transforms(cfg: dict):
    spatial_keys = ["image", "label", "weight"]
    if not cfg["augment"]["enabled"]:
        return Compose([EnsureTyped(keys=spatial_keys, track_meta=False)])

    spatial_cfg = cfg["augment"]["spatial"]
    intensity_cfg = cfg["augment"]["intensity"]
    domain_cfg = cfg["augment"]["domain"]

    degrees = float(spatial_cfg["rotate_range_deg"])
    _, scale_hi = spatial_cfg["scale_range"]

    transforms = [
        EnsureTyped(keys=spatial_keys, track_meta=False),
        ThickSliceSimd(
            prob=float(domain_cfg["thick_slice_sim_prob"]),
            z_factors=domain_cfg["thick_slice_z_downsample"],
        ),
        RandFlipd(
            keys=spatial_keys,
            prob=float(spatial_cfg["rand_flip_prob"]),
            spatial_axis=0,
        ),
        RandFlipd(
            keys=spatial_keys,
            prob=float(spatial_cfg["rand_flip_prob"]),
            spatial_axis=1,
        ),
        RandFlipd(
            keys=spatial_keys,
            prob=float(spatial_cfg["rand_flip_prob"]),
            spatial_axis=2,
        ),
        RandAffined(
            keys=spatial_keys,
            prob=float(spatial_cfg["affine_prob"]),
            rotate_range=(
                np.deg2rad(degrees),
                np.deg2rad(degrees),
                np.deg2rad(degrees),
            ),
            scale_range=(scale_hi - 1.0, scale_hi - 1.0, scale_hi - 1.0),
            mode=("bilinear", "nearest", "nearest"),
            padding_mode="border",
        ),
        Rand3DElasticd(
            keys=spatial_keys,
            prob=float(spatial_cfg["elastic_prob"]),
            sigma_range=(3, 6),
            magnitude_range=(30, 90),
            mode=("bilinear", "nearest", "nearest"),
            padding_mode="border",
        ),
        RandBiasFieldd(keys=["image"], prob=float(intensity_cfg["bias_field_prob"])),
    ]

    gamma_prob = float(intensity_cfg["gamma_prob"])
    gamma_range = intensity_cfg["gamma_range"]
    if _RandGammaTransform is not None:
        transforms.append(
            _RandGammaTransform(
                keys=["image"],
                prob=gamma_prob,
                gamma=gamma_range,
            )
        )
    else:
        # Fallback for MONAI versions without RandGammaD/RandGammad.
        transforms.append(
            RandAdjustContrastd(
                keys=["image"],
                prob=gamma_prob,
                gamma=gamma_range,
            )
        )

    transforms.extend(
        [
            RandGaussianNoised(
                keys=["image"],
                prob=float(intensity_cfg["noise_prob"]),
                std=float(intensity_cfg["noise_std"]),
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=float(intensity_cfg["contrast_prob"]),
                gamma=intensity_cfg["contrast_range"],
            ),
            RandCoarseDropoutd(**_coarse_dropout_kwargs(intensity_cfg)),
        ]
    )

    return Compose(transforms)
