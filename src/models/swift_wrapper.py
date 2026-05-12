import sys
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


SWIFT_REPO = Path(__file__).resolve().parents[2] / "SwiFT"
sys.path.insert(0, str(SWIFT_REPO))

from project.module.models.swin4d_transformer_ver7 import SwinTransformer4D


class SwiFTEncoder(nn.Module):
    """Frozen SwiFT encoder for feature extraction.

    Loads the pre-trained SwiFT checkpoint, strips the output head,
    and exposes the final layer features (spatio-temporal feature maps).

    Input:  (B, 1, 96, 96, 96, T)  float32 tensor
    Output: (B, C, D, H, W, T)      feature map from last layer
    """

    def __init__(
        self,
        ckpt_path: str,
        img_size: Tuple[int, int, int, int] = (96, 96, 96, 20),
        patch_size: Tuple[int, int, int, int] = (6, 6, 6, 1),
        window_size: Tuple[int, int, int, int] = (4, 4, 4, 4),
        first_window_size: Tuple[int, int, int, int] = (2, 2, 2, 2),
        embed_dim: int = 24,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        c_multiplier: int = 2,
        in_chans: int = 1,
        attn_drop_rate: float = 0.0,
        last_layer_full_MSA: bool = False,
        feature_layer: Optional[int] = None,
        spatial_pool: str = "avg",
    ):
        super().__init__()
        self.spatial_pool = spatial_pool

        # Build SwiFT model architecture (no head)
        self.model = SwinTransformer4D(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            first_window_size=first_window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            c_multiplier=c_multiplier,
            last_layer_full_MSA=last_layer_full_MSA,
            drop_rate=attn_drop_rate,
            drop_path_rate=attn_drop_rate,
            attn_drop_rate=attn_drop_rate,
            to_float=True,
        )

        self._load_weights(ckpt_path)
        self._freeze()

        n_stages = len(depths)
        self.feature_dim = int(embed_dim * c_multiplier ** (n_stages - 1))
        self.patch_grid = (
            (img_size[0] // patch_size[0]) // (2 ** (n_stages - 1)),
            (img_size[1] // patch_size[1]) // (2 ** (n_stages - 1)),
            (img_size[2] // patch_size[2]) // (2 ** (n_stages - 1)),
        )

    def _load_weights(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            key = k
            if key.startswith("model."):
                key = key[len("model."):]
            if "output_head" in key:
                continue
            new_sd[key] = v
        missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"SwiFTEncoder: missing keys: {missing}")
        if unexpected:
            print(f"SwiFTEncoder: unexpected keys: {unexpected}")

    def _freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x)
        return x

    def extract_features(
        self, x: torch.Tensor, pool_spatial: bool = True, pool_time: bool = False
    ) -> torch.Tensor:
        """Run encoder and produce feature vectors per sample.

        Args:
            x: (B, 1, 96, 96, 96, T) input.
            pool_spatial: average pool spatial dims -> (B, C, T).
            pool_time: average pool temporal dim -> (B, C).

        Returns:
            Features with shape depending on pooling options.
        """
        feats = self.forward(x)
        # feats: (B, C, D, H, W, T)
        if pool_spatial:
            b, c, d, h, w, t = feats.shape
            feats = feats.mean(dim=(2, 3, 4))  # (B, C, T)
        if pool_time:
            feats = feats.mean(dim=-1)  # (B, C)
        return feats
