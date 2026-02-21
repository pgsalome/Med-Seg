import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Sequence


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


class SkipDecoder3D(nn.Module):
    """
    Lightweight UNet-style decoder operating on encoder skip tensors.

    Expects skips in encoder order: high-resolution first, bottleneck last.
    """

    def __init__(
        self,
        skip_channels: Sequence[int],
        out_channels: int = 1,
        deep_supervision: bool = False,
        num_outputs: int = 4,
    ):
        super().__init__()
        if len(skip_channels) < 2:
            raise ValueError("SkipDecoder3D requires at least 2 feature maps.")

        self.deep_supervision = bool(deep_supervision)
        self.num_outputs = max(1, int(num_outputs))
        self.blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        num_stages = len(skip_channels) - 1
        for stage_idx in range(num_stages):
            in_below = int(skip_channels[-(stage_idx + 1)])
            in_skip = int(skip_channels[-(stage_idx + 2)])
            in_ch = in_below + in_skip
            out_ch = in_skip
            self.blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(out_ch, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(out_ch, affine=True),
                    nn.LeakyReLU(inplace=True),
                )
            )
            self.seg_layers.append(nn.Conv3d(out_ch, int(out_channels), kernel_size=1))

    def forward(self, skips: Sequence[torch.Tensor]):
        x = skips[-1]
        stage_logits = []
        for stage_idx, block in enumerate(self.blocks):
            skip = skips[-(stage_idx + 2)]
            x = F.interpolate(
                x,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            x = torch.cat((x, skip), dim=1)
            x = block(x)
            stage_logits.append(self.seg_layers[stage_idx](x))

        if not stage_logits:
            raise RuntimeError("SkipDecoder3D did not produce logits.")

        # stage_logits order is low->high resolution; reverse to high->low.
        stage_logits = stage_logits[::-1]
        if (not self.deep_supervision) or (not self.training):
            return stage_logits[0]
        return stage_logits[: min(self.num_outputs, len(stage_logits))]


class EncoderToSegModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_type: str,
        out_channels: int = 1,
        deep_supervision: bool = False,
        num_deep_supervision_outputs: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_type = encoder_type
        self.out_channels = out_channels
        self.deep_supervision = bool(deep_supervision)
        self.num_deep_supervision_outputs = max(1, int(num_deep_supervision_outputs))
        self.decoder: Optional[nn.Module] = None
        self.seg_head: Optional[nn.Module] = None

    def _init_head(self, feat: torch.Tensor):
        if self.seg_head is None:
            self.seg_head = SimpleSegHead3D(feat.shape[1], self.out_channels).to(feat.device)

    def _init_triad_decoder(self, skips: List[torch.Tensor]):
        if self.decoder is not None:
            return
        skip_channels = [int(t.shape[1]) for t in skips]
        self.decoder = SkipDecoder3D(
            skip_channels=skip_channels,
            out_channels=self.out_channels,
            deep_supervision=self.deep_supervision,
            num_outputs=self.num_deep_supervision_outputs,
        )
        self.decoder = self.decoder.to(skips[0].device)

    def forward(self, x):
        if self.encoder_type in {"triad_plain", "triad_swin"}:
            enc_out = self.encoder(x)
            if isinstance(enc_out, torch.Tensor):
                skips = [enc_out]
            else:
                skips = list(enc_out)

            if len(skips) >= 2:
                self._init_triad_decoder(skips)
                logits_low = self.decoder(skips)
            else:
                feat = skips[-1]
                self._init_head(feat)
                logits_low = self.seg_head(feat)
        elif self.encoder_type == "brainiac":
            feat = self.encoder(x)
            self._init_head(feat)
            logits_low = self.seg_head(feat)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        if isinstance(logits_low, (list, tuple)):
            logits_list = list(logits_low)
            if not logits_list:
                raise RuntimeError("Model returned empty logits list.")
            primary = F.interpolate(
                logits_list[0],
                size=x.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            return [primary] + logits_list[1:]

        return F.interpolate(
            logits_low,
            size=x.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
