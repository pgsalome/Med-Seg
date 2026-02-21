from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
import yaml


def load_tta_preset(path: str) -> List[Dict]:
    """Load YAML and return list of transform specs."""
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    tta_cfg = data.get("tta", {}) if isinstance(data, dict) else {}
    transforms = tta_cfg.get("transforms", [])
    if not isinstance(transforms, list):
        raise ValueError(f"Invalid TTA preset at {path}: transforms must be a list.")
    return transforms


def _axis_to_tensor_dim(axis: int) -> int:
    # Spec axes are Z/Y/X => 0/1/2; tensor dims are B,C,Z,Y,X => +2.
    if axis not in (0, 1, 2):
        raise ValueError(f"Unsupported flip axis '{axis}'. Expected one of 0,1,2.")
    return axis + 2


def _axes_to_tensor_dims(axes: List[int]) -> Tuple[int, int]:
    if len(axes) != 2:
        raise ValueError(f"rot90 expects exactly 2 axes, got {axes}")
    dim0 = _axis_to_tensor_dim(int(axes[0]))
    dim1 = _axis_to_tensor_dim(int(axes[1]))
    return dim0, dim1


def apply_transform(x: torch.Tensor, spec: Dict) -> torch.Tensor:
    ttype = str(spec.get("type", "identity")).lower()
    if ttype == "identity":
        return x
    if ttype == "flip":
        axis = int(spec.get("axis", 0))
        return torch.flip(x, dims=[_axis_to_tensor_dim(axis)])
    if ttype == "rot90":
        k = int(spec.get("k", 1)) % 4
        axes = spec.get("axes", [1, 2])
        dims = _axes_to_tensor_dims([int(axes[0]), int(axes[1])])
        return torch.rot90(x, k=k, dims=dims)
    raise ValueError(f"Unsupported TTA transform type: {ttype}")


def inverse_transform(p: torch.Tensor, spec: Dict) -> torch.Tensor:
    ttype = str(spec.get("type", "identity")).lower()
    if ttype == "identity":
        return p
    if ttype == "flip":
        axis = int(spec.get("axis", 0))
        return torch.flip(p, dims=[_axis_to_tensor_dim(axis)])
    if ttype == "rot90":
        k = int(spec.get("k", 1)) % 4
        inv_k = (4 - k) % 4
        axes = spec.get("axes", [1, 2])
        dims = _axes_to_tensor_dims([int(axes[0]), int(axes[1])])
        return torch.rot90(p, k=inv_k, dims=dims)
    raise ValueError(f"Unsupported TTA transform type: {ttype}")


def build_tta_fns(transform_specs: List[Dict]) -> List[Callable]:
    """
    Returns list of callables:
      fn(x) -> (x_aug, inv_fn)
    Where inv_fn maps prediction back to original space.
    """
    fns = []
    for spec in transform_specs:
        spec_local = dict(spec)

        def _fn(x: torch.Tensor, spec_ref: Dict = spec_local):
            x_aug = apply_transform(x, spec_ref)

            def _inv_fn(pred: torch.Tensor, spec_inv: Dict = spec_ref) -> torch.Tensor:
                return inverse_transform(pred, spec_inv)

            return x_aug, _inv_fn

        fns.append(_fn)
    return fns
