import os
import sys

import torch
from torch import nn


class BrainIACEncoder(nn.Module):
    """
    Loads BrainIAC encoder from a local clone.
    """

    def __init__(self, repo_root: str, checkpoint_path: str):
        super().__init__()
        if not os.path.isdir(repo_root):
            raise FileNotFoundError(f"BrainIAC repo not found: {repo_root}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"BrainIAC checkpoint not found: {checkpoint_path}")

        sys.path.insert(0, repo_root)
        sys.path.insert(0, os.path.join(repo_root, "src"))

        model = None
        tried = []
        for module_path, class_name in [
            ("src.models.brainiac", "BrainIAC"),
            ("src.model", "BrainIAC"),
            ("models.brainiac", "BrainIAC"),
            ("brainiac", "BrainIAC"),
        ]:
            try:
                module = __import__(module_path, fromlist=[class_name])
                model = getattr(module, class_name)()
                break
            except Exception as exc:  # pragma: no cover
                tried.append((f"{module_path}.{class_name}", str(exc)))

        if model is None:
            tried_msg = "\n".join(f"- {name}: {err}" for name, err in tried)
            raise ImportError(f"Could not import BrainIAC model. Tried:\n{tried_msg}")

        self.encoder = model
        checkpoint = self._safe_torch_load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.encoder(x)

    @staticmethod
    def _safe_torch_load(path: str, map_location: str = "cpu"):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)
        except Exception:
            return torch.load(path, map_location=map_location, weights_only=False)
