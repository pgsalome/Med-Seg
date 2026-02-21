import os
from typing import List

import torch
from torch.utils.data import Dataset

from .dataset import brain_mask_from_zscored, case_cache_key, load_npz, save_npz
from .io_registry import CaseItem
from .preprocess import load_and_preprocess_with_steps


class MedSegCacheVolumeDataset(Dataset):
    """
    Deterministic full-volume dataset for validation/inference.

    Returns full cached tensors with shape CZYX:
      - image: float32 tensor
      - label: float32 tensor
      - brain_mask: float32 tensor
    """

    def __init__(self, cases: List[CaseItem], cfg: dict, transforms=None):
        self.cases = cases
        self.cfg = cfg
        self.transforms = transforms

        self.cache_dir = cfg["cache"]["cache_dir"]
        os.makedirs(self.cache_dir, exist_ok=True)

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

    def __getitem__(self, idx):
        case = self.cases[idx]
        img_czyx, lab_czyx, spacing_xyz, brain_zyx = self._get_cached(case)
        if brain_zyx is None:
            brain_zyx = brain_mask_from_zscored(img_czyx[0])
        brain_czyx = brain_zyx[None, ...].astype("float32")

        sample = {
            "image": torch.from_numpy(img_czyx),
            "label": torch.from_numpy(lab_czyx.astype("float32")),
            "brain_mask": torch.from_numpy(brain_czyx),
            "uid": case.uid,
            "spacing_xyz": torch.tensor(spacing_xyz, dtype=torch.float32),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
