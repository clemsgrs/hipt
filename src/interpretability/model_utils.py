from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision

import src.models.vision_transformer as vits
from src.data.augmentations import RegionUnfolding
from src.models.components import Attn_Net_Gated
from src.models.utils import update_state_dict
from src.interpretability.fm import FoundationModelFactory
from src.interpretability.utils import hf_login


def create_transforms(model, level, patch_size):
    if level == "global":
        return model.get_transforms()
    elif level == "local":
        # check if model has get_transforms() method, if so add it in the compose
        transforms_list = [torchvision.transforms.ToTensor(), RegionUnfolding(patch_size)]
        if hasattr(model, "get_transforms") and callable(model.get_transforms):
            transforms_list.append(model.get_transforms())
        return torchvision.transforms.Compose(transforms_list)
    else:
        raise ValueError(f"Unknown model level: {level}")


def get_patch_transformer(
        model_options: Dict,
        model_device: torch.device = torch.device("cpu"),
):
    if model_options.name is not None:
        hf_login()
        patch_transformer = FoundationModelFactory(model_options).get_model()
    elif model_options.arch is not None:
        patch_transformer = get_custom_patch_transformer(
            model_options.pretrained_weights,
            arch=model_options.arch,
            mask_attn=model_options.mask_attn,
            device=model_device,
            verbose=True,
        )
        # add 'tile_size' attribute to patch_transformer
        patch_transformer.tile_size = model_options.patch_size
    return patch_transformer


def get_custom_patch_transformer(
    pretrained_weights: str,
    arch: str = "vit_small",
    token_size: int = 16,
    mask_attn: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    checkpoint_key = "teacher"
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    patch_transformer = vits.__dict__[arch](
        patch_size=token_size,
        num_classes=0,
        mask_attn=mask_attn,
    )
    for p in patch_transformer.parameters():
        p.requires_grad = False
    patch_transformer.eval()
    patch_transformer.to(device)

    if Path(pretrained_weights).is_file():
        if verbose:
            print("Loading pretrained weights for patch-level Transformer...")
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            if verbose:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict, msg = update_state_dict(model_dict=patch_transformer.state_dict(), state_dict=state_dict)
        patch_transformer.load_state_dict(state_dict, strict=True)
        if verbose:
            print(f"Pretrained weights found at {pretrained_weights}")
            print(msg)

    return patch_transformer


def get_region_transformer(
    state_dict: dict,
    input_embed_dim: int,
    region_size: int,
    patch_size: int,
    arch: str = "vit4k_xs",
    mask_attn: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    region_transformer = vits.__dict__[arch](
        img_size=region_size,
        patch_size=patch_size,
        input_embed_dim=input_embed_dim,
        num_classes=0,
        mask_attn=mask_attn,
    )
    for p in region_transformer.parameters():
        p.requires_grad = False
    region_transformer.eval()
    region_transformer.to(device)

    if verbose:
        print("Loading weights for region-level Transformer...")
    state_dict = {k.replace("vit_region.", ""): v for k, v in state_dict.items()}
    state_dict, msg = update_state_dict(model_dict=region_transformer.state_dict(), state_dict=state_dict)
    region_transformer.load_state_dict(state_dict, strict=False)
    if verbose:
        print(msg)

    return region_transformer


class SlideAgg(nn.Module):
    def __init__(
        self,
        embed_dim_region: int = 192,
        dropout: float = 0.25,
    ):
        super(SlideAgg, self).__init__()

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )

        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192,
                nhead=3,
                dim_feedforward=192,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=192, D=192, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(192, 192), nn.ReLU(), nn.Dropout(dropout)]
        )

    def forward(self, x, return_attention: bool = False):
        # x = [M, 192]
        x = self.global_phi(x)

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1) # [M, 192]
        att, x = self.global_attn_pool(x) # [M, 1], [M, 192]
        att = torch.transpose(att, 1, 0) # [1, M]
        att = torch.nn.functional.softmax(att, dim=1) # [1, M], softmaxed attention scores across M regions
        if return_attention:
            return att
        x_att = torch.mm(att, x) # [1, 192], sum of regions embeddings, weighted by corresponding attention score
        x_wsi = self.global_rho(x_att) # [1, 192]
        return x_wsi


def get_slide_transformer(
    state_dict: dict,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    slide_transformer = SlideAgg()
    for p in slide_transformer.parameters():
        p.requires_grad = False
    slide_transformer.eval()
    slide_transformer.to(device)

    if verbose:
        print("Loading weights for slide-level Transformer...")
    state_dict, msg = update_state_dict(model_dict=slide_transformer.state_dict(), state_dict=state_dict)
    slide_transformer.load_state_dict(state_dict, strict=False)
    if verbose:
        print(msg)
    return slide_transformer


def get_classifier(
    input_dim: int,
    num_classes: int,
    state_dict: dict,
    device: torch.device | None = None,
    verbose: bool = True,
):

    classifier = torch.nn.Linear(input_dim, num_classes)
    for p in classifier.parameters():
        p.requires_grad = False
    classifier.eval()
    classifier.to(device)

    if verbose:
        print("Loading weights for classifier...")
    state_dict = {k.replace("classifier.", ""): v for k, v in state_dict.items()}
    state_dict, msg = update_state_dict(model_dict=classifier.state_dict(), state_dict=state_dict)
    classifier.load_state_dict(state_dict, strict=False)
    if verbose:
        print(msg)
    return classifier