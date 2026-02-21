import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


class SimpleSegHead3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        mid = max(8, in_ch // 2)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class EncoderToSegModel(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_type: str, out_channels: int = 1):
        super().__init__()
        self.encoder = encoder
        self.encoder_type = encoder_type
        self.out_channels = out_channels
        self.seg_head: Optional[nn.Module] = None

    def _init_head(self, feat: torch.Tensor):
        if self.seg_head is None:
            self.seg_head = SimpleSegHead3D(feat.shape[1], self.out_channels).to(feat.device)

    def forward(self, x):
        if self.encoder_type in {"triad_plain", "triad_swin"}:
            feat = self.encoder(x)[-1]
        elif self.encoder_type == "brainiac":
            feat = self.encoder(x)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self._init_head(feat)
        logits_low = self.seg_head(feat)
        logits = F.interpolate(
            logits_low,
            size=x.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        return logits
