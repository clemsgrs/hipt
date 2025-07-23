import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from src.models.base import BaseModel
from src.models.utils import update_state_dict
from src.models.vision_transformer import vit4k_xs
from src.models.components import Attn_Net_Gated


class GlobalHIPT(BaseModel):
    def __init__(
        self,
        num_classes: int,
        embed_dim_region: int = 192,
        d_model: int = 192,
        dropout: float = 0.25,
    ):
        super(GlobalHIPT, self).__init__()

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, d_model), nn.ReLU(), nn.Dropout(dropout)
        )

        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=3,
                dim_feedforward=d_model,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=d_model, D=d_model, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout)]
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x = [M, embed_dim_region]
        x = self.global_phi(x)  # (M, d_model))

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)

        return logits


class LocalHIPT(BaseModel):
    def __init__(
        self,
        num_classes: int,
        region_size: int,
        patch_size: int,
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        embed_dim_slide: int = 192,
        dropout: float = 0.25,
        mask_attn_region: bool = False,
        num_register_tokens_region: int = 0,
        pretrain_vit_region: str | None = None,
        img_size_pretrained: int | None = None,
    ):
        super(LocalHIPT, self).__init__()
        self.npatch = int(region_size // patch_size)
        self.num_register_tokens_region = num_register_tokens_region

        checkpoint_key = "teacher"

        self.vit_region = vit4k_xs(
            img_size=region_size,
            patch_size=patch_size,
            input_embed_dim=embed_dim_patch,
            output_embed_dim=embed_dim_region,
            mask_attn=mask_attn_region,
            img_size_pretrained=img_size_pretrained,
            num_register_tokens=num_register_tokens_region,
        )

        if pretrain_vit_region and Path(pretrain_vit_region).is_file():
            print("Loading pretrained weights for region-level Transformer...")
            state_dict = torch.load(pretrain_vit_region, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(
                self.vit_region.state_dict(), state_dict
            )
            self.vit_region.load_state_dict(state_dict, strict=True)
            print(f"Pretrained weights found at {pretrain_vit_region}")
            print(msg)

        elif pretrain_vit_region:
            print(
                f"{pretrain_vit_region} doesnt exist ; please provide path to existing file"
            )

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, embed_dim_slide), nn.ReLU(), nn.Dropout(dropout)
        )

        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim_slide,
                nhead=3,
                dim_feedforward=embed_dim_slide,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=embed_dim_slide, D=embed_dim_slide, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(embed_dim_slide, embed_dim_slide), nn.ReLU(), nn.Dropout(dropout)]
        )

        self.classifier = nn.Linear(embed_dim_slide, num_classes)

    def forward(self, x, pct: torch.Tensor | None = None, pct_thresh: float = 0.0):
        mask_patch = None
        if pct is not None:
            pct_patch = torch.sum(pct, axis=-1) / pct[0].numel()
            mask_patch = (pct_patch > pct_thresh).int()  # (M, npatch**2) e.g. (M, 64)
            # add the [CLS] token to the mask
            cls_token = mask_patch.new_ones((mask_patch.size(0), 1))
            # eventually add register tokens to the mask
            # they're added after the [CLS] token in the input sequence
            if self.num_register_tokens_region:
                register_tokens = mask_patch.new_ones(
                    (mask_patch.size(0), self.num_register_tokens_region)
                )
                mask_patch = torch.cat((cls_token, register_tokens, mask_patch), dim=1)  # [M, num_patches+1+self.num_register_tokens_region]
            else:
                mask_patch = torch.cat((cls_token, mask_patch), dim=1)  # [M, num_patches+1]

        # x = [M, 256, 384]
        x = self.vit_region(
            x.unfold(1, self.npatch, self.npatch).transpose(1, 2),
            mask=mask_patch,
        )  # [M, 192]
        x = self.global_phi(x)  # [M, 192]

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)

        return logits