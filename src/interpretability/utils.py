import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import getpass
from huggingface_hub import login

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
import omegaconf
import wholeslidedata as wsd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, binary_closing
from scipy.stats import rankdata

import src.models.vision_transformer as vits
from src.data.augmentations import RegionUnfolding
from src.models.components import Attn_Net_Gated
from src.models.utils import update_state_dict
from src.interpretability.wsi import WholeSlideImage, SegmentationParameters
from src.interpretability.fm import FoundationModelFactory


def transform_tiling_config(old_config: dict) -> dict:
    """
    Transforms old tiling config format (list of single-key dicts) into the new structured format.
    """

    # Extract and flatten top-level tiling keys
    seg_params = old_config.get('seg_params', [])
    filter_params = old_config.get('params', [])

    # Build the new config
    new_config = {
            'read_coordinates_from': None,
            'backend': old_config['backend'],
            'params': {
                'spacing': old_config['spacing'],
                'tolerance': 0.07,
                'tile_size': old_config['tile_size'],
                'overlap': filter_params.get('overlap', 0.0),
                'min_tissue_percentage': filter_params.get('tissue_thresh', 0.01),
                'drop_holes': filter_params.get('drop_holes', False),
                'use_padding': filter_params.get('use_padding', True),
            },
            'seg_params': {
                'downsample': old_config['downsample'],
                'sthresh': seg_params.get('sthresh', 8),
                'sthresh_up': 255,
                'mthresh': seg_params.get('mthresh', 7),
                'close': seg_params.get('close', 4),
                'use_otsu': seg_params.get('use_otsu', False),
                'tissue_pixel_value': old_config.get('tissue_pixel_value', 1),
            },
            'filter_params': {
                'ref_tile_size': filter_params.get('ref_tile_size', 16),
                'a_t': filter_params.get('a_t', 4),
                'a_h': filter_params.get('a_h', 2),
                'max_n_holes': filter_params.get('max_n_holes', 8),
            },
            'visu_params': {
                'downsample': 32
            }
    }

    return new_config


def get_config_from_path(config_path: Path | str):
    full_cfg = omegaconf.OmegaConf.load(config_path)
    cfg = full_cfg.tiling
    if hasattr(cfg, "read_coordinates_from"):
        if cfg.read_coordinates_from:
            raise RuntimeError(
                f"Provided tiling configuration might not be accurate as coordinates were read from: {cfg.read_coordinates_from}, please grab the tiling configuration from that folder instead"
            )
    else:
        # reorganize older config
        cfg = transform_tiling_config(cfg)
        # Convert dict to an object with attribute access
        cfg = omegaconf.OmegaConf.create(cfg)
    return cfg


def hf_login():
    if "HF_TOKEN" not in os.environ:
        hf_token = getpass.getpass(
            "Enter your Hugging Face API token (input will not be visible): "
        )
        os.environ["HF_TOKEN"] = hf_token
    login(os.environ["HF_TOKEN"])


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


def add_margin(pil_img, top, right, bottom, left, color):
    """
    Adds custom margin to PIL.Image.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def tensorbatch2im(input_image, imtype=np.uint8):
    """
    Converts a Tensor array into a numpy image array.

    Args:
        - input_image (torch.Tensor): (B, C, W, H) Torch Tensor.
        - imtype (type): the desired type of the converted numpy array

    Returns:
        - image_numpy (np.array): (B, W, H, C) Numpy Array.
    """
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (
            (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def normalize_patch_scores(
    attns,
    size=(256, 256),
    method: str = 'min-max',
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
):
    if method == 'min-max':
        assert min_val is not None and max_val is not None, "min_val and max_val must be provided for min-max normalization"
        rank = lambda v: (v - min_val) / (max_val - min_val)
    elif method == 'rank':
        rank = lambda v: (rankdata(v, method='min')-1) / len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns][0]
    return color_block


def concat_patch_scores(
    attns,
    region_size: int,
    patch_size: int,
    size: Optional[Tuple[int, int]] = None,
    method: str = 'min-max',
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
):
    n_patch = region_size // patch_size
    if method == 'min-max':
        assert min_val is not None and max_val is not None, "min_val and max_val must be provided for min-max normalization"
        norm = lambda v: (v - min_val) / (max_val - min_val)
    elif method == 'rank':
        norm = lambda v: (rankdata(v, method='min')-1) / len(v)
    normalized_attn = [
        norm(attn.flatten()).reshape(size) for attn in attns
    ]  # [(256, 256)] of length len(attns)
    normalized_attns = np.concatenate(
        [
            np.concatenate(normalized_attn[i : (i + n_patch)], axis=1)
            for i in range(0, n_patch**2, n_patch)
        ]
    )
    # (16*256, 16*256)
    return normalized_attns


def normalize_region_scores(
    attn,
    size: Optional[Tuple[int, int]] = None,
    method: str = 'min-max',
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
):
    if method == 'min-max':
        assert min_val is not None and max_val is not None, "min_val and max_val must be provided for min-max normalization"
        norm = lambda v: (v - min_val) / (max_val - min_val)
    elif method == 'rank':
        norm = lambda v: (rankdata(v, method='min')-1) / len(v)
    normalized_attn = norm(attn.flatten()).reshape(size)
    return normalized_attn


def normalize_slide_scores(
    attn,
    method: str = 'min-max',
):
    if method == 'min-max':
        min_val, max_val = attn.min(), attn.max()
        norm = lambda v: (v - min_val) / (max_val - min_val)
    elif method == 'rank':
        norm = lambda v: (rankdata(v, method='min')-1) / len(v)
    normalized_attn = norm(attn)
    return normalized_attn


def getConcatImage(imgs, how="horizontal", gap=0):
    """
    Function to concatenate list of images (vertical or horizontal).

    Args:
        - imgs (list of PIL.Image): List of PIL Images to concatenate.
        - how (str): How the images are concatenated (either 'horizontal' or 'vertical')
        - gap (int): Gap (in px) between images

    Return:
        - dst (PIL.Image): Concatenated image result.
    """
    gap_dist = (len(imgs) - 1) * gap

    if how == "vertical":
        w, h = np.max([img.width for img in imgs]), np.sum([img.height for img in imgs])
        h += gap_dist
        curr_h = 0
        dst = Image.new("RGBA", (w, h), color=(255, 255, 255, 0))
        for img in imgs:
            dst.paste(img, (0, curr_h))
            curr_h += img.height + gap

    elif how == "horizontal":
        w, h = np.sum([img.width for img in imgs]), np.min([img.height for img in imgs])
        w += gap_dist
        curr_w = 0
        dst = Image.new("RGBA", (w, h), color=(255, 255, 255, 0))

        for idx, img in enumerate(imgs):
            dst.paste(img, (curr_w, 0))
            curr_w += img.width + gap

    return dst


def cmap_map(function, cmap):
    """
    Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.

    Args:
    - function (function)
    - cmap (matplotlib.colormap)

    Returns:
    - matplotlib.colormap
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ("red", "green", "blue"):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(["red", "green", "blue"]):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1],), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap("colormap", cdict, 1024)


def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(
        img,
        tuple(np.maximum([0, 0], np.array(coord) - thickness // 2)),
        tuple(np.array(coord) - thickness // 2 + np.array(shape)),
        color,
        thickness=thickness,
    )
    return img


def DrawMapFromCoords(
    canvas,
    wsi_object,
    coordinates,
    patch_size,
    vis_level: int,
    draw_grid: bool = True,
    thickness: int = 2,
    verbose: bool = False,
):
    downsamples = wsi_object.level_downsamples[vis_level]

    for coord in coordinates:
        x, y = coord # defined w.rt. level 0

        # extract tile
        width, height = patch_size
        vis_spacing = wsi_object.get_level_spacing(vis_level)
        tile = wsi_object.get_tile(x, y, width, height, vis_spacing)

        # downsample coordinates from level 0 to vis_level
        downsampled_x, downsampled_y = tuple(
            (np.array(coord) / np.array(downsamples)).astype(int)
        )

        canvas_crop_shape = canvas[
            downsampled_y : downsampled_y + height,
            downsampled_x : downsampled_x + width,
            :3,
        ].shape[:2]
        canvas[
            downsampled_y : downsampled_y + height,
            downsampled_x : downsampled_x + width,
            :3,
        ] = tile[: canvas_crop_shape[0], : canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, (downsampled_x, downsampled_y), patch_size, thickness=thickness)

    return Image.fromarray(canvas)


def get_mask(
    wsi_path: str,
    segmentation_mask_path: str,
    x: int,
    y: int,
    region_size: int,
    patch_size: int,
    token_size: int,
    spacing: float,
    backend: str = 'asap',
    downsample: int = 4,
    background_pixel_value: int = 0,
    tissue_pixel_value: int = 1,
    pct_thresh: float = 0.0,
    offset: Optional[int] = None,
):
    # load the slide
    wsi = wsd.WholeSlideImage(Path(wsi_path), backend=backend)
    # load segmentation mask
    mask = wsd.WholeSlideImage(Path(segmentation_mask_path), backend=backend)
    # scale coordinates from slide's level 0 to mask's level 0
    sx, sy = tuple(i / j for i, j in zip(mask.shapes[0], wsi.shapes[0]))
    # find spacing of level closest to desired downsample
    idx = np.argmin([abs(downsample - d) for d in mask.downsamplings])
    downsample_spacing = mask.spacings[idx]
    # scale region size from true spacing to downsample spacing
    # we excepct mask spacings to be a subset of slide spacings
    # the ratio should thus give an integer
    sr = int(downsample_spacing / wsi.get_real_spacing(spacing))
    scaled_region_size = region_size // sr
    # scale patch_size and token_size
    scaled_patch_size = patch_size // sr
    scaled_token_size = token_size // sr

    x_mask, y_mask = int(x * sx), int(y * sy)
    region_mask = mask.get_patch(
        x=x_mask,
        y=y_mask,
        width=scaled_region_size,
        height=scaled_region_size,
        spacing=downsample_spacing,
        center=False,
    )
    if offset:
        scaled_offset = offset // sr
        region_mask = region_mask[scaled_offset:, scaled_offset:]
        height, width = region_mask.shape[:2]
        new_height = height + scaled_offset
        new_width = width + scaled_offset
        new_region_mask = np.full((new_height, new_width, region_mask.shape[2]), background_pixel_value, dtype=region_mask.dtype)
        new_region_mask[0:height, 0:width] = region_mask
        region_mask = new_region_mask

    assert region_mask.shape[-1] == 1, f"expecting 1 channel, found {region_mask.shape[-1]} channels"
    region_mask = region_mask[...,0]

    def split_into_blocks(arr, block_size):
        """
        Split a 2D array into smaller blocks of given block_size.
        """
        # Split the array into sub-arrays of size block_size along the rows
        rows_split = np.split(arr, arr.shape[0] // block_size)
        # Further split these sub-arrays along the columns
        blocks = [
            np.split(block, arr.shape[1] // block_size, axis=1)
            for block in rows_split
        ]
        # Flatten the list of blocks
        flattened_blocks = [
            block for sublist in blocks for block in sublist
        ]
        return np.array(flattened_blocks)

    # compute tissue percentage for each mini patch in each patch
    region_patches = split_into_blocks(region_mask, scaled_patch_size)

    region_tokenes = []
    for p in region_patches:
        mp = split_into_blocks(p, scaled_token_size)
        region_tokenes.append(mp)
    region_tokenes = np.stack(region_tokenes)
    tissue = region_tokenes == tissue_pixel_value
    pct = np.sum(tissue, axis=(-2, -1)) / tissue[0][0].size # (npatch**2, nminipatch**2)
    pct = torch.Tensor(pct)

    mask_token = (pct > pct_thresh).int()  # (num_patches, nminipatch**2)
    # add the [CLS] token to the mask
    cls_token = mask_token.new_ones((mask_token.size(0),1))
    mask_token = torch.cat((cls_token, mask_token), dim=1)  # [num_patches, nminipatch**2+1]
    # infer patch-level mask
    pct_patch = torch.sum(pct, axis=-1) / pct.numel()
    mask_patch = (pct_patch > pct_thresh).int().unsqueeze(0)
    # add the [CLS] token to the mask
    cls_token = mask_patch.new_ones((mask_patch.size(0),1))
    mask_patch = torch.cat((cls_token, mask_patch), dim=1)  # [1, num_patches+1]

    return mask_patch, mask_token


def generate_masks(
    slide_id: str,
    wsi_path: str,
    segmentation_mask_path: str,
    region_dir: str,
    region_size: int,
    patch_size: int,
    token_size: int,
    spacing: float,
    backend: str = 'asap',
    downsample: int = 4,
    region_format: str = "jpg",
    tissue_pixel_value: int = 1,
    pct_thresh: float = 0.0,
):
    # load the slide
    wsi = wsd.WholeSlideImage(Path(wsi_path), backend=backend)
    # load segmentation mask
    mask = wsd.WholeSlideImage(Path(segmentation_mask_path), backend=backend)
    # scale coordinates from slide's level 0 to mask's level 0
    sx, sy = tuple(i / j for i, j in zip(mask.shapes[0], wsi.shapes[0]))
    # find spacing of level closest to desired downsample
    idx = np.argmin([abs(downsample - d) for d in mask.downsamplings])
    downsample_spacing = mask.spacings[idx]
    # scale region size from true spacing to downsample spacing
    # we excepct mask spacings to be a subset of slide spacings
    # the ratio should thus give an integer
    sr = round(downsample_spacing / wsi.get_real_spacing(spacing))
    scaled_region_size = region_size // sr
    # scale patch_size and token_size
    scaled_patch_size = patch_size // sr
    scaled_token_size = token_size // sr
    # retrieve region's (x,y) coordinates
    # should appear in the same order as in the corresponding slide feature vector
    coordinates = sorted(
        [p.stem for p in Path(region_dir, slide_id, "imgs").glob(f"*.{region_format}")]
    )
    coordinates = [
        (int(p.split("_")[0]), int(p.split("_")[1])) for p in coordinates
    ]
    tissue_pcts = []
    for i, (x, y) in enumerate(coordinates):
        x_mask, y_mask = int(x * sx), int(y * sy)
        region = mask.get_patch(
            x=x_mask,
            y=y_mask,
            width=scaled_region_size,
            height=scaled_region_size,
            spacing=downsample_spacing,
            center=False,
        )
        assert region.shape[-1] == 1, f"expecting 1 channel, found {region.shape[-1]} channels"
        region = region[...,0]

        def split_into_blocks(arr, block_size):
            """
            Split a 2D array into smaller blocks of given block_size.
            """
            # Split the array into sub-arrays of size block_size along the rows
            rows_split = np.split(arr, arr.shape[0] // block_size)
            # Further split these sub-arrays along the columns
            blocks = [
                np.split(block, arr.shape[1] // block_size, axis=1)
                for block in rows_split
            ]
            # Flatten the list of blocks
            flattened_blocks = [
                block for sublist in blocks for block in sublist
            ]
            return np.array(flattened_blocks)

        # compute tissue percentage for each mini patch in each patch
        region_patches = split_into_blocks(region, scaled_patch_size)

        region_tokenes = []
        for p in region_patches:
            mp = split_into_blocks(p, scaled_token_size)
            region_tokenes.append(mp)
        region_tokenes = np.stack(region_tokenes)
        tissue = region_tokenes == tissue_pixel_value
        tissue_pct = np.sum(tissue, axis=(-2, -1)) / tissue[0][0].size
        tissue_pcts.append(tissue_pct)

    pct = np.stack(tissue_pcts)  # (M, npatch**2, nminipatch**2)
    pct = torch.Tensor(pct)

    mask_token = (pct > pct_thresh).int()  # (M, num_patches, nminipatch**2)
    # add the [CLS] token to the mask
    cls_token = mask_token.new_ones((mask_token.size(0),mask_token.size(1),1))
    mask_token = torch.cat((cls_token, mask_token), dim=2)  # [M, num_patches, nminipatch**2+1]
    # infer patch-level mask
    pct_patch = torch.sum(pct, axis=-1) / pct[0].numel()
    mask_patch = (pct_patch > pct_thresh).int()
    # add the [CLS] token to the mask
    cls_token = mask_patch.new_ones((mask_patch.size(0),1))
    mask_patch = torch.cat((cls_token, mask_patch), dim=1)  # [M, num_patches+1]

    return mask_patch, mask_token


def get_region_attention_scores(
    region: Image,
    patch_transformer: nn.Module,
    region_transformer: nn.Module,
    patch_size: int,
    transforms: torchvision.transforms.Compose,
    downscale: int = 1,
    patch_attn_mask: Optional[torch.Tensor] = None,
    token_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    compute_patch_attention: bool = True,
):
    """
    Forward pass in hierarchical model with attention scores saved.

    Args:
    - region (PIL.Image): input region
    - patch_transformer (nn.Module): patch-level Transformer
    - region_transformer (nn.Module): region-level Transformer
    - downscale (int): how much to downscale the output regions by (e.g. downscale=4 will resize 4096x4096 regions to 1024x1024)
    - patch_device (torch.device): device on which patch_transformer is
    - region_device (torch.device): device on which region_transformer is

    Returns:
    - np.array: [n_patch**2, patch_size/downscale, patch_size/downscale, 3] array sequence of image patch_size-sized patches from the input region.
    - patch_attention (torch.Tensor): [n_patch**2, nhead, patch_size/downscale, patch_size/downscale] tensor sequence of attention maps for patch_size-sized patches.
    - region_attention (torch.Tensor): [nhead, region_size/downscale, region_size/downscale] tensor sequence of attention maps for input region.
    """
    token_size = patch_transformer.token_size
    region_size = region.size[0]

    n_patch = region_size // patch_size
    n_token = patch_size // token_size

    with torch.no_grad():
        patches = transforms(region)  # [n_patch**2, 3, patch_size, patch_size]
        patches = patches.to(patch_device, non_blocking=True)
        if token_attn_mask is not None:
            token_attn_mask = token_attn_mask.to(patch_device, non_blocking=True)
        patch_features = patch_transformer(patches, mask=token_attn_mask)  # (n_patch**2, 384)

        patch_attention = None
        if compute_patch_attention:
            patch_attention = patch_transformer.get_last_selfattention(
                patches,
                mask=token_attn_mask,
            )  # (n_patch**2, nhead, n_token**2+1, n_token**2+1)
            nh = patch_attention.shape[1]  # number of head
            patch_attention = patch_attention[:, :, 0, 1:].reshape(
                n_patch**2, nh, -1
            )  # (n_patch**2, nhead, n_token**2)
            patch_attention = patch_attention.reshape(
                n_patch**2, nh, n_token, n_token
            )  # (n_patch**2, nhead, n_token, n_token)
            patch_attention = (
                nn.functional.interpolate(
                    patch_attention,
                    scale_factor=int(token_size / downscale),
                    mode="nearest",
                )
                .cpu()
                .numpy()
            )  # (n_patch**2, nhead, patch_size, patch_size) when downscale = 1
            # 'nearest' interpolation guarantees the values in the up-sampled array
            # lie in the same set as the values in the original array

        region_features = (
            patch_features.unfold(0, n_patch, n_patch).transpose(0, 1).unsqueeze(dim=0)
        )  # (1, 384, n_patch, n_patch)
        if patch_attn_mask is not None:
            patch_attn_mask = patch_attn_mask.to(region_device)
        region_attention = region_transformer.get_last_selfattention(
            region_features.detach().to(region_device),
            mask=patch_attn_mask,
        )  # (1, nhead, n_patch**2+1, n_patch**2+1)
        nh = region_attention.shape[1]  # number of head
        region_attention = region_attention[0, :, 0, 1:].reshape(
            nh, -1
        )  # (nhead, 1, n_patch**2) -> (nhead, n_patch**2)
        region_attention = region_attention.reshape(
            nh, n_patch, n_patch
        )  # (nhead, n_patch, n_patch)
        region_attention = (
            nn.functional.interpolate(
                region_attention.unsqueeze(0),
                scale_factor=int(patch_size / downscale),
                mode="nearest",
            )[0]
            .cpu()
            .numpy()
        )  # (nhead, region_size, region_size) when downscale = 1

        if downscale != 1:
            patches = nn.functional.interpolate(
                patches, scale_factor=(1 / downscale), mode="nearest"
            )

    return tensorbatch2im(patches), patch_attention, region_attention


def get_slide_attention_scores(
    wsi_path: Path,
    coordinates_dir: Path,
    patch_transformer: nn.Module,
    region_transformer: nn.Module,
    slide_transformer: nn.Module,
    patch_size: int,
    transforms: torchvision.transforms.Compose,
    downscale: int = 1,
    granular: bool = False,
    offset: int = 1024,
    patch_attn_mask: Optional[torch.Tensor] = None,
    token_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    slide_device: torch.device = torch.device("cuda:0"),
    verbose: bool = True,
):
    """
    Forward pass in hierarchical model with attention scores saved.

    Args:
    - wsi_path (Path): path to the slide
    - coordinates_dir (Path): path to root folder containing tiles coordinates
    - patch_transformer (nn.Module): patch-level Transformer
    - region_transformer (nn.Module): region-level Transformer
    - slide_transformer (nn.Module): slide-level Transformer
    - downscale (int): how much to downscale the output regions by (e.g. downscale=4 will resize 4096x4096 regions to 1024x1024)
    - patch_device (torch.device): device on which patch_transformer is
    - region_device (torch.device): device on which region_transformer is
    - verbose (bool): controls tqdm display when running with multiple gpus

    Returns:
    - np.array: [n_patch**2, patch_size/downscale, patch_size/downscale, 3] array sequence of image patch_size-sized patches from the input region.
    - patch_attention (torch.Tensor): [n_patch**2, nhead, patch_size/downscale, patch_size/downscale] tensor sequence of attention maps for patch_size-sized patches.
    - region_attention (torch.Tensor): [nhead, region_size/downscale, region_size/downscale] tensor sequence of attention maps for input region.
    """
    wsi = wsd.WholeSlideImage(wsi_path, backend="asap")

    slide_id = wsi_path.stem.replace(" ", "_")
    coordinates_file = coordinates_dir / f"{slide_id}.npy"
    coordinates_arr = np.load(coordinates_file)
    nregions = len(coordinates_arr)

    try:
        coordinates = list(zip(coordinates_arr[:,0], coordinates_arr[:,1]))
        region_size_resized = coordinates_arr[0][2]
        region_level = coordinates_arr[0][3]
        resize_factor = coordinates_arr[0][4]
    except Exception as e:
        coordinates = list(zip(coordinates_arr["x"], coordinates_arr["y"]))
        region_size_resized = coordinates_arr["tile_size_resized"][0]
        region_level = coordinates_arr["tile_level"][0]
        resize_factor = coordinates_arr["resize_factor"][0]

    region_spacing = wsi.spacings[region_level]
    region_size = int(round(region_size_resized / resize_factor,0))

    if granular:

        assert wsi_path
        w, h = wsi.shapes[region_level]

        offset_ = offset
        ncomp = region_size // offset

        offset = offset // downscale
        s = region_size // downscale

        slide_overlay = np.zeros((nregions,s,s))
        combined_slide_attention = np.zeros((nregions,s,s))

        coords = []
        skip_count = 0

        for i, ix in enumerate([-1, 1]):
            for j, iy in enumerate([-1, 1]):
                for offset_x in range(ncomp):
                    for offset_y in range(ncomp):

                        if ix == -1 and iy == 1 and offset_y == 0:
                            skip_count += 1
                            continue
                        if ix == 1 and iy == -1 and offset_x == 0:
                            skip_count += 1
                            continue
                        if ix == 1 and iy == 1 and offset_x == 0:
                            skip_count += 1
                            continue
                        if ix == 1 and iy == 1 and offset_y == 0:
                            skip_count += 1
                            continue

                        features = []

                        with tqdm.tqdm(
                            coordinates,
                            desc=f"Gathering attention scores [{i*(2*ncomp*ncomp)+j*(ncomp*ncomp)+offset_x*ncomp+offset_y+1-skip_count}/{(4*ncomp**2)-4*ncomp+1}]",
                            unit="region",
                            leave=True,
                            disable=not verbose,
                        ) as t1:

                            for k, (x,y) in enumerate(t1):

                                if offset_x == offset_y == 0:
                                    coords.append((x, y))

                                new_x = max(0, min(x + ix * offset_x * offset_, w))
                                new_y = max(0, min(y + iy * offset_y * offset_, h))

                                offset_region_arr = wsi.get_patch(
                                    new_x,
                                    new_y,
                                    region_size_resized,
                                    region_size_resized,
                                    spacing=region_spacing,
                                    center=False,
                                )
                                offset_region = Image.fromarray(offset_region_arr)
                                if region_size != region_size_resized:
                                    offset_region = offset_region.resize((region_size, region_size))

                                n_patch = region_size // patch_size

                                with torch.no_grad():

                                    patches = transforms(offset_region)
                                    patches = patches.to(patch_device, non_blocking=True)
                                    token_mask = None
                                    if token_attn_mask is not None:
                                        token_mask = token_attn_mask[k].to(patch_device, non_blocking=True)
                                    patch_features = patch_transformer(patches, mask=token_mask)  # (n_patch**2, 384)

                                    regions = (
                                        patch_features.unfold(0, n_patch, n_patch)
                                        .transpose(0, 1)
                                        .unsqueeze(dim=0)
                                    )  # (1, 384, n_patch, n_patch)
                                    regions = regions.to(region_device, non_blocking=True)
                                    patch_mask = None
                                    if patch_attn_mask is not None:
                                        patch_mask = patch_attn_mask[k].unsqueeze(0).to(region_device, non_blocking=True)
                                    region_features = region_transformer(regions, mask=patch_mask)  # (1, 192)

                                    features.append(region_features)

                        with torch.no_grad():

                            feature_seq = torch.stack(features, dim=0).squeeze(1)  # (M, 192)
                            feature_seq = feature_seq.to(slide_device, non_blocking=True)
                            slide_attention = slide_transformer(feature_seq, return_attention=True)
                            slide_attention = slide_attention.squeeze(0)  # (M)
                            slide_attention = normalize_slide_scores(slide_attention.cpu().numpy())  # (M)
                            slide_attention *= 100
                            slide_attention = slide_attention.reshape(-1, 1, 1)  # (M, 1, 1)
                            slide_attention = torch.from_numpy(slide_attention).to(
                                slide_device, non_blocking=True
                            )

                            slide_attention = (
                                nn.functional.interpolate(
                                    slide_attention.unsqueeze(0),
                                    scale_factor=int(region_size / downscale),
                                    mode="nearest",
                                )[0]
                                .cpu()
                                .numpy()
                            )  # (M, region_size, region_size) when downscale = 1
                            # 'nearest' interpolation guarantees the values in the up-sampled array
                            # lie in the same set as the values in the original array

                            # only pick attention scores overlapping with the non-offset region
                            overlapping_slide_attention = np.zeros_like(slide_attention)
                            if ix == -1 and iy == -1:
                                # x shift is negative, y shift is negative
                                overlapping_slide_attention[
                                    :,
                                    :s-offset_x*offset,
                                    :s-offset_y*offset
                                ] = slide_attention[
                                    :,
                                    offset_x*offset:s,
                                    offset_y*offset:s
                                ]
                                slide_overlay[:, :s-offset_x*offset, :s-offset_y*offset] += 100
                            if ix == -1 and iy == 1:
                                # x shift is negative, y shift is positive
                                overlapping_slide_attention[
                                    :,
                                    :s-offset_x*offset,
                                    offset_y*offset:s,
                                ] = slide_attention[
                                    :,
                                    offset_x*offset:s,
                                    :s-offset_y*offset
                                ]
                                slide_overlay[:, :s-offset_x*offset, offset_y*offset:s] += 100
                            if ix == 1 and iy == -1:
                                # x shift is positive, y shift is negative
                                overlapping_slide_attention[
                                    :,
                                    offset_x*offset:s,
                                    :s-offset_y*offset
                                ] = slide_attention[
                                    :,
                                    :s-offset_x*offset,
                                    offset_y*offset:s
                                ]
                                slide_overlay[:, offset_x*offset:s, :s-offset_y*offset] += 100
                            if ix == 1 and iy == 1:
                                # x shift is positive, y shift is positive
                                overlapping_slide_attention[
                                    :,
                                    offset_x*offset:s,
                                    offset_y*offset:s
                                ] = slide_attention[
                                    :,
                                    :s-offset_x*offset,
                                    :s-offset_y*offset
                                ]
                                slide_overlay[:, offset_x*offset:s, offset_y*offset:s] += 100

                            combined_slide_attention += overlapping_slide_attention

        slide_attention = combined_slide_attention / slide_overlay

    else:

        coords = []
        features = []

        with tqdm.tqdm(
            coordinates,
            desc="Gathering attention scores",
            unit="region",
            leave=True,
            disable=not verbose,
        ) as t1:
            for k, (x,y) in enumerate(t1):
                region_arr = wsi.get_patch(
                    x,
                    y,
                    region_size_resized,
                    region_size_resized,
                    region_spacing,
                    center=False,
                )
                region = Image.fromarray(region_arr)
                if region_size != region_size_resized:
                    region = region.resize((region_size, region_size))
                n_patch = region_size // patch_size

                coords.append((x, y))

                with torch.no_grad():
                    patches = transforms(region)  # (n_patch**2, 3, patch_size, patch_size)
                    patches = patches.to(patch_device, non_blocking=True)
                    token_mask = None
                    if token_attn_mask is not None:
                        token_mask = token_attn_mask[k].to(patch_device, non_blocking=True)
                    patch_features = patch_transformer(patches, mask=token_mask)  # (n_patch**2, 384)

                    regions = (
                        patch_features.unfold(0, n_patch, n_patch)
                        .transpose(0, 1)
                        .unsqueeze(dim=0)
                    )  # (1, 384, n_patch, n_patch)
                    regions = regions.to(region_device, non_blocking=True)
                    patch_mask = None
                    if patch_attn_mask is not None:
                        patch_mask = patch_attn_mask[k].unsqueeze(0).to(region_device, non_blocking=True)
                    region_features = region_transformer(regions, mask=patch_mask)  # (1, 192)

                    features.append(region_features)

        with torch.no_grad():

            feature_seq = torch.stack(features, dim=0).squeeze(1)  # (M, 192)
            feature_seq = feature_seq.to(slide_device, non_blocking=True)
            slide_attention = slide_transformer(feature_seq, return_attention=True)
            slide_attention = slide_attention.squeeze(0)  # (M)
            slide_attention = normalize_slide_scores(slide_attention.cpu().numpy())  # (M)
            slide_attention = slide_attention.reshape(-1, 1, 1)  # (M, 1, 1)
            slide_attention = torch.from_numpy(slide_attention).to(
                slide_device, non_blocking=True
            )

            slide_attention = (
                nn.functional.interpolate(
                    slide_attention.unsqueeze(0),
                    scale_factor=int(region_size / downscale),
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )  # (M, region_size, region_size) when downscale = 1
            # 'nearest' interpolation guarantees the values in the up-sampled array
            # lie in the same set as the values in the original array

    return slide_attention, coords


def get_region_level_heatmaps(
    wsi_path: Path,
    coordinates_dir: Path,
    patch_transformer: nn.Module,
    region_transformer: nn.Module,
    patch_size: int,
    transforms: torchvision.transforms.Compose,
    output_dir: Path,
    downscale: int = 1,
    granular: bool = False,
    offset: int = 128,
    segmentation_mask_path: Optional[str] = None,
    segmentation_parameters: SegmentationParameters | None = None,
    spacing: Optional[float] = None,
    downsample: Optional[int] = None,
    background_pixel_value: Optional[int] = 0,
    tissue_pixel_value: Optional[int] = 1,
    patch_attn_mask: Optional[torch.Tensor] = None,
    token_attn_mask: Optional[torch.Tensor] = None,
    compute_patch_attention: bool =True,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    verbose: bool = True,
):
    """
    Returns heatmaps of region-level Transformer attention for each region extracted in the slide.
    These can later be stitched together for slide-level heatmap visualization.

    Args:
    - wsi_path (Path): path to the slide
    - coordinates_dir (Path): path to root folder containing region coordinates in .npy files
    - patch_transformer (nn.Module): patch-level Transformer
    - region_transformer (nn.Module): region-level Transformer
    - output_dir (Path): output directory for saving heatmaps
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - region_fmt (str): file format used for extracted regions
    - granular (bool): create additional offset regions to get more granular heatmaps
    - offset (int): if granular is True, uses this value to offset regions
    - patch_device (torch.device): device on which patch_transformer is
    - region_device (torch.device): device on which region_transformer is
    - verbose (bool): controls tqdm display when running with multiple gpus
    """
    slide_id = wsi_path.stem.replace(" ", "_")
    wsi_object = WholeSlideImage(wsi_path, segmentation_mask_path, segment_params=segmentation_parameters)

    coordinates_file = coordinates_dir / f"{slide_id}.npy"
    coordinates_arr = np.load(coordinates_file)
    nregions = len(coordinates_arr)

    try:
        coordinates = list(zip(coordinates_arr[:,0], coordinates_arr[:,1]))
        region_size_resized = coordinates_arr[0][2]
        region_level = coordinates_arr[0][3]
        resize_factor = coordinates_arr[0][4]
    except Exception as e:
        coordinates = list(zip(coordinates_arr["x"], coordinates_arr["y"]))
        region_size_resized = coordinates_arr["tile_size_resized"][0]
        region_level = coordinates_arr["tile_level"][0]
        resize_factor = coordinates_arr["resize_factor"][0]

    width, height = wsi_object.level_dimensions[0]

    region_spacing = wsi_object.spacings[region_level]
    region_size = int(round(region_size_resized / resize_factor,0))

    # scale offsset from desired spacing to region_spacing
    offset_resized = int(offset * resize_factor)

    attention_dir = output_dir / "attention" / "region" / f"{region_size}"
    attention_dir.mkdir(exist_ok=True, parents=True)

    nhead_region = region_transformer.num_heads
    head_min = [float('inf')] * nhead_region
    head_max = [float('-inf')] * nhead_region

    mask_attn_patch = (token_attn_mask is not None)
    mask_attn_region = (patch_attn_mask is not None)
    mask_attention = mask_attn_patch or mask_attn_region

    with tqdm.tqdm(
        coordinates,
        desc="Gathering attention scores",
        unit="region",
        leave=True,
        disable=not verbose,
    ) as t1:
        for k, (x,y) in enumerate(t1):
            region_arr = wsi_object.get_tile(
                x,
                y,
                region_size_resized,
                region_size_resized,
                region_spacing,
            )
            region = Image.fromarray(region_arr)
            if region_size != region_size_resized:
                region = region.resize((region_size, region_size))

            patch_mask, token_mask = None, None
            if mask_attn_patch:
                token_mask = token_attn_mask[k]
            if mask_attn_region:
                patch_mask = patch_attn_mask[k].unsqueeze(0)

            _, _, region_att = get_region_attention_scores(
                region,
                patch_transformer,
                region_transformer,
                patch_size,
                transforms=transforms,
                downscale=downscale,
                patch_attn_mask=patch_mask,
                token_attn_mask=token_mask,
                patch_device=patch_device,
                region_device=region_device,
                compute_patch_attention=compute_patch_attention,
            ) # (nhead, region_size, region_size) when downscale = 1

            if granular:
                directions = [
                    (-1, -1),  # Northwest
                    (0, -1),   # North
                    (1, -1),   # Northeast
                    (-1, 0),   # West
                    (1, 0),    # East
                    (-1, 1),   # Southwest
                    (0, 1),    # South
                    (1, 1),    # Southeast
                ]
                shifted_region_attention = defaultdict(list)
                for dx, dy in directions:
                    for e in range(offset_resized, region_size_resized, offset_resized):
                        # scale shift from region_level to level 0
                        e_x = int(e * wsi_object.level_downsamples[region_level][0])
                        e_y = int(e * wsi_object.level_downsamples[region_level][1])
                        shift_x = dx * e_x
                        shift_y = dy * e_y
                        x_shifted = min(max(x + shift_x, 0), width)
                        y_shifted = min(max(y + shift_y, 0), height)
                        shifted_region_arr = wsi_object.get_tile(
                            x_shifted,
                            y_shifted,
                            region_size_resized,
                            region_size_resized,
                            region_spacing,
                        )
                        shifted_region = Image.fromarray(shifted_region_arr)
                        if region_size != region_size_resized:
                            shifted_region = shifted_region.resize((region_size, region_size))
                        shifted_patch_mask, shifted_token_mask = None, None
                        if mask_attention:
                            shifted_patch_mask, shifted_token_mask = get_mask(
                                wsi_path,
                                segmentation_mask_path,
                                x,
                                y,
                                region_size,
                                patch_size,
                                patch_transformer.token_size,
                                spacing,
                                backend='asap',
                                downsample=downsample,
                                background_pixel_value=background_pixel_value,
                                tissue_pixel_value=tissue_pixel_value,
                                offset=e,
                            )
                            if not mask_attn_patch:
                                shifted_token_mask = None
                            if not mask_attn_region:
                                shifted_patch_mask = None

                        _, _, shifted_region_att = get_region_attention_scores(
                            shifted_region,
                            patch_transformer,
                            region_transformer,
                            patch_size,
                            transforms=transforms,
                            downscale=downscale,
                            patch_attn_mask=shifted_patch_mask,
                            token_attn_mask=shifted_token_mask,
                            patch_device=patch_device,
                            region_device=region_device,
                            compute_patch_attention=compute_patch_attention,
                        )
                        shifted_region_attention[(dx,dy)].append(shifted_region_att)

            with tqdm.tqdm(
                range(nhead_region),
                desc=f"Processing region [{k+1}/{nregions}]",
                unit="head",
                leave=False,
                disable=not verbose,
            ) as t2:
                for j in t2:

                    region_scores = region_att[j] # (region_size, region_size) when downscale = 1

                    if granular:
                        directions = [
                            (-1, -1),  # Northwest
                            (0, -1),   # North
                            (1, -1),   # Northeast
                            (-1, 0),   # West
                            (1, 0),    # East
                            (-1, 1),   # Southwest
                            (0, 1),    # South
                            (1, 1),    # Southeast
                        ]
                        region_overlay = np.ones_like(region_scores)
                        for dx, dy in directions:
                            for edx, e in enumerate(range(offset, region_size, offset)):
                                e_scaled = e // downscale
                                s = region_size // downscale
                                shifted_region_scores = np.zeros_like(region_scores)

                                # determine the overlapping region based on the sign of dx and dy
                                x_start = max(0, dx * e_scaled)
                                x_end = s - max(0, -dx * e_scaled)
                                y_start = max(0, dy * e_scaled)
                                y_end = s - max(0, -dy * e_scaled)

                                shifted_x_start = max(0, -dx * e_scaled)
                                shifted_x_end = s - max(0, dx * e_scaled)
                                shifted_y_start = max(0, -dy * e_scaled)
                                shifted_y_end = s - max(0, dy * e_scaled)

                                # fill only the overlapping values
                                shifted_region_scores[x_start:x_end, y_start:y_end] = shifted_region_attention[(dx, dy)][edx][j][shifted_x_start:shifted_x_end, shifted_y_start:shifted_y_end]
                                region_scores += shifted_region_scores

                                # increment the overlay only for the overlapping positions
                                region_overlay[x_start:x_end, y_start:y_end] += 1

                        region_scores = region_scores / region_overlay

                    head_min[j] = min(head_min[j], region_scores.min())
                    head_max[j] = max(head_max[j], region_scores.max())

                    attention_head_dir = attention_dir / f"head_{j}"
                    attention_head_dir.mkdir(exist_ok=True, parents=True)
                    np.save(Path(attention_head_dir, f"{x}_{y}.npy"), region_scores)

    head_min = [float(e) for e in head_min]
    head_max = [float(e) for e in head_max]
    with open(Path(attention_dir, 'min.json'), 'w') as f:
        json.dump(head_min, f)
    with open(Path(attention_dir, 'max.json'), 'w') as f:
        json.dump(head_max, f)

    return attention_dir


def get_slide_level_heatmaps(
    wsi_path: Path,
    coordinates_dir: Path,
    patch_transformer: nn.Module,
    region_transformer: nn.Module,
    slide_transformer: nn.Module,
    patch_size: int,
    transforms: torchvision.transforms.Compose,
    output_dir: Path,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    threshold: Optional[float] = None,
    highlight: Optional[float] = None,
    opacity: float = 0.3,
    granular: bool = False,
    offset: int = 1024,
    patch_attn_mask: Optional[torch.Tensor] = None,
    token_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    slide_device: torch.device = torch.device("cuda:0"),
    verbose: bool = True,
):
    """
    Returns heatmaps of slide-level Transformer attention for each region extracted in the slide.
    These can later be stitched together for slide-level heatmap visualization.

    Args:
    - wsi_path (Path): path to the slide
    - coordinates_dir (Path): path to root folder containing region coordinates in .npy files
    - patch_transformer (nn.Module): patch-level Transformer
    - region_transformer (nn.Module): region-level Transformer
    - slide_transformer (nn.Module): slide-level Transformer
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - threshold (float): filter out regions with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - highlight (float): filter out regions with attention scores lower than this value (set to None to disbale heatmap highlighting)
    - opacity (float): if highlight, set opacity for non-highlighted regions on stitched heatmap
    - patch_device (torch.device): device on which patch_transformer is
    - region_device (torch.device): device on which region_transformer is
    - verbose (bool): controls tqdm display when running with multiple gpus
    """
    slide_id = wsi_path.stem.replace(" ", "_")

    att, _ = get_slide_attention_scores(
        wsi_path,
        coordinates_dir,
        patch_transformer,
        region_transformer,
        slide_transformer,
        patch_size,
        transforms=transforms,
        downscale=downscale,
        granular=granular,
        offset=offset,
        patch_attn_mask=patch_attn_mask,
        token_attn_mask=token_attn_mask,
        patch_device=patch_device,
        region_device=region_device,
        slide_device=slide_device,
    )  # (M, region_size, region_size), (M)

    wsi = wsd.WholeSlideImage(wsi_path, backend="asap")

    coordinates_file = coordinates_dir / f"{slide_id}.npy"
    coordinates_arr = np.load(coordinates_file)

    try:
        coordinates = list(zip(coordinates_arr[:,0], coordinates_arr[:,1]))
        region_size_resized = coordinates_arr[0][2]
        region_level = coordinates_arr[0][3]
        resize_factor = coordinates_arr[0][4]
    except Exception as e:
        coordinates = list(zip(coordinates_arr["x"], coordinates_arr["y"]))
        region_size_resized = coordinates_arr["tile_size_resized"][0]
        region_level = coordinates_arr["tile_level"][0]
        resize_factor = coordinates_arr["resize_factor"][0]

    region_spacing = wsi.spacings[region_level]
    region_size = int(round(region_size_resized / resize_factor,0))

    attention_dir = output_dir / "attention" / "slide" / f"{region_size}"
    attention_dir.mkdir(exist_ok=True, parents=True)

    visualization_dir = output_dir / "visualization" / "slide" / f"{region_size}"
    visualization_dir.mkdir(exist_ok=True, parents=True)

    with tqdm.tqdm(
        coordinates,
        desc="Gathering attention scores",
        unit="region",
        leave=True,
        disable=not verbose,
    ) as t1:
        for k, (x,y) in enumerate(t1):
            region_arr = wsi.get_patch(
                x,
                y,
                region_size_resized,
                region_size_resized,
                spacing=region_spacing,
                center=False,
            )
            region = Image.fromarray(region_arr)
            if region_size != region_size_resized:
                region = region.resize((region_size, region_size))

            s = region_size // downscale

            if threshold != None:
                thresh_slide_hm_output_dir = visualization_dir / "thresholded"
                thresh_slide_hm_output_dir.mkdir(exist_ok=True, parents=True)
                save_region = np.array(region.resize((s, s)))
                att_mask = att[k].copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask >= threshold] = 0.95

                thresh_color_block = (cmap(att_mask) * 255)[:, :, :3].astype(np.uint8)
                thresh_hm = cv2.addWeighted(
                    thresh_color_block,
                    alpha,
                    save_region.copy(),
                    1 - alpha,
                    0,
                    save_region.copy(),
                )
                thresh_hm[att_mask == 0] = 0
                img_inverse = save_region.copy()
                img_inverse[att_mask == 0.95] = 0
                thresh_hm = thresh_hm + img_inverse
                img = Image.fromarray(thresh_hm)
                img.save(Path(thresh_slide_hm_output_dir, f"{x}_{y}.png"))

            if highlight != None:
                highlight_slide_hm_output_dir = visualization_dir / "highlighted"
                highlight_slide_hm_output_dir.mkdir(exist_ok=True, parents=True)
                save_region = np.array(region.resize((s, s)))
                rgba_region = np.dstack((save_region, np.zeros((s,s), dtype=np.uint8)+255))
                att_mask = att[k].copy()
                att_mask[att_mask < highlight] = 0
                att_mask[att_mask >= highlight] = 1
                highlighted_hm = rgba_region * (att_mask >= highlight)[..., np.newaxis]
                img = Image.fromarray(highlighted_hm)
                img.save(Path(highlight_slide_hm_output_dir, f"{x}_{y}.png"))

            # save raw attention scores to disk
            np.save(Path(attention_dir, f"{x}_{y}.npy"), att[k])

            # given region is an RGB image, so the default filter for resizing is Resampling.BICUBIC
            # which is fine as we're resizing the image here, not attention scores
            save_region = np.array(region.resize((s, s)))

            color_block = (cmap(att[k]) * 255)[:, :, :3].astype(np.uint8)
            hm = cv2.addWeighted(
                color_block,
                alpha,
                save_region.copy(),
                1 - alpha,
                0,
                save_region.copy(),
            )
            slide_hm_output_dir = visualization_dir / "regular"
            slide_hm_output_dir.mkdir(exist_ok=True, parents=True)
            img = Image.fromarray(hm)
            img.save(Path(slide_hm_output_dir, f"{x}_{y}.png"))

    return attention_dir


def get_factorized_heatmaps(
    wsi_path: Path,
    coordinates_dir: Path,
    patch_transformer: nn.Module,
    region_transformer: nn.Module,
    slide_transformer: nn.Module,
    patch_size: int,
    transforms: torchvision.transforms.Compose,
    level: str,
    output_dir: Path,
    gamma: float,
    method: str = "multiply",
    downscale: int = 1,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    threshold: Optional[float] = None,
    granularity: Optional[Dict] = None,
    segmentation_mask_path: Optional[str] = None,
    spacing: Optional[float] = None,
    downsample: Optional[int] = None,
    background_pixel_value: Optional[int] = 0,
    tissue_pixel_value: Optional[int] = 1,
    patch_attn_mask: Optional[torch.Tensor] = None,
    token_attn_mask: Optional[torch.Tensor] = None,
    compute_patch_attention: bool = True,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    slide_device: torch.device = torch.device("cuda:0"),
    verbose: bool = True,
):
    """
    Returns factorized heatmaps (patch-level, region-level & slide-level Transformer heatmaps factorized together) for each region extracted in the slide.
    These can later be stitched together for slide-level heatmap visualization.

    Args:
    - wsi_path (Path): path to the slide
    - coordinates_dir (Path): path to root folder containing region coordinates in .npy files
    - patch_transformer (nn.Module): patch-level Transformer
    - region_transformer (nn.Module): region-level Transformer
    - slide_transformer (nn.Module): slide-level Transformer
    - level (str): level at which the model was trained on
    - output_dir (Path): output directory for saving heatmaps
    - gamma (float): factor weighting the importance given to frozen model attention scores w.r.t finetuned model attention scores
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - threshold (float): filter out patches with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - save_to_disk (bool): wether to save individual region heatmaps to disk
    - granularity (dict): create additional offset patches to get more granular heatmaps
    - patch_device (torch.device): device on which patch_transformer is
    - region_device (torch.device): device on which region_transformer is
    - slide_device (torch.device): device on which slide_transformer is
    - verbose (bool): controls tqdm display when running with multiple gpus
    """
    slide_id = wsi_path.stem.replace(" ", "_")

    slide_attention, _ = get_slide_attention_scores(
        wsi_path,
        coordinates_dir,
        patch_transformer,
        region_transformer,
        slide_transformer,
        patch_size,
        transforms=transforms,
        downscale=downscale,
        granular=granularity.slide,
        offset=granularity.offset.slide,
        patch_attn_mask=patch_attn_mask,
        token_attn_mask=token_attn_mask,
        patch_device=patch_device,
        region_device=region_device,
        slide_device=slide_device,
        verbose=verbose,
    )  # (M, region_size, region_size), (M)

    wsi = wsd.WholeSlideImage(wsi_path, backend="asap")

    coordinates_file = coordinates_dir / f"{slide_id}.npy"
    coordinates_arr = np.load(coordinates_file)
    nregions = len(coordinates_arr)

    try:
        coordinates = list(zip(coordinates_arr[:,0], coordinates_arr[:,1]))
        region_size_resized = coordinates_arr[0][2]
        region_level = coordinates_arr[0][3]
        resize_factor = coordinates_arr[0][4]
    except Exception as e:
        coordinates = list(zip(coordinates_arr["x"], coordinates_arr["y"]))
        region_size_resized = coordinates_arr["tile_size_resized"][0]
        region_level = coordinates_arr["tile_level"][0]
        resize_factor = coordinates_arr["resize_factor"][0]

    width, height = wsi.shapes[0]
    region_spacing = wsi.spacings[region_level]
    region_size = int(round(region_size_resized / resize_factor,0))

    token_size = patch_transformer.token_size

    raw_attention_dir = output_dir / "attention" / "factorized" / "raw" / f"{region_size}-{patch_size}"
    raw_attention_dir.mkdir(exist_ok=True, parents=True)

    factorized_attention_dir = output_dir / "attention" / "factorized" / "factorized" / f"{region_size}-{patch_size}"
    factorized_attention_dir.mkdir(exist_ok=True, parents=True)

    visualization_dir = output_dir / "visualization" / "factorized" / f"{region_size}-{patch_size}"
    visualization_dir.mkdir(exist_ok=True, parents=True)

    nhead_patch = 1
    compute_patch_attention = compute_patch_attention and gamma != 1
    if compute_patch_attention:
        nhead_patch = patch_transformer.num_heads
    nhead_region = region_transformer.num_heads

    head_min_region = [float('inf')] * nhead_region
    head_max_region = [float('-inf')] * nhead_region

    head_min_patch = [float('inf')] * nhead_region
    head_max_patch = [float('-inf')] * nhead_region

    mask_attn_patch = (token_attn_mask is not None)
    mask_attn_region = (patch_attn_mask is not None)
    mask_attention = mask_attn_patch or mask_attn_region

    with tqdm.tqdm(
        coordinates,
        desc="Gathering attention scores",
        unit="region",
        leave=True,
        disable=not verbose,
    ) as t1:
        for k, (x,y) in enumerate(t1):
            region_arr = wsi.get_patch(
                x,
                y,
                region_size_resized,
                region_size_resized,
                spacing=region_spacing,
                center=False,
            )
            region = Image.fromarray(region_arr)
            if region_size != region_size_resized:
                region = region.resize((region_size, region_size))

            n_patch = region_size // patch_size

            patch_mask, token_mask = None, None
            if mask_attn_patch:
                token_mask = token_attn_mask[k]
            if mask_attn_region:
                patch_mask = patch_attn_mask[k].unsqueeze(0)

            _, patch_att, region_att = get_region_attention_scores(
                region,
                patch_transformer,
                region_transformer,
                patch_size,
                transforms=transforms,
                downscale=downscale,
                patch_attn_mask=patch_mask,
                token_attn_mask=token_mask,
                patch_device=patch_device,
                region_device=region_device,
                compute_patch_attention=compute_patch_attention,
            )
            # region_att is (nhead, region_size, region_size) when downscale = 1

            if granularity.region:
                directions = [
                    (-1, -1),  # Northwest
                    (0, -1),   # North
                    (1, -1),   # Northeast
                    (-1, 0),   # West
                    (1, 0),    # East
                    (-1, 1),   # Southwest
                    (0, 1),    # South
                    (1, 1),    # Southeast
                ]
                shifted_patch_attention = defaultdict(list)
                shifted_region_attention = defaultdict(list)
                offset = min(granularity.offset.patch, granularity.offset.region)
                # scale offsset from desired spacing to region_spacing
                offset_resized = int(offset * resize_factor)
                for dx, dy in directions:
                    for e in range(offset_resized, region_size_resized, offset_resized):
                        # scale shift from region_level to level 0
                        e_x = int(e * list(wsi.downsamplings)[region_level])
                        e_y = int(e * list(wsi.downsamplings)[region_level])
                        shift_x = dx * e_x
                        shift_y = dy * e_y
                        x_shifted = min(max(x + shift_x, 0), width)
                        y_shifted = min(max(y + shift_y, 0), height)
                        shifted_region_arr = wsi.get_patch(
                            x_shifted,
                            y_shifted,
                            region_size_resized,
                            region_size_resized,
                            spacing=region_spacing,
                            center=False,
                        )
                        shifted_region = Image.fromarray(shifted_region_arr)
                        if region_size != region_size_resized:
                            shifted_region = shifted_region.resize((region_size, region_size))
                        shifted_patch_mask, shifted_token_mask = None, None
                        if mask_attention:
                            shifted_patch_mask, shifted_token_mask = get_mask(
                                wsi_path,
                                segmentation_mask_path,
                                x,
                                y,
                                region_size,
                                patch_size,
                                token_size,
                                spacing,
                                backend='asap',
                                downsample=downsample,
                                background_pixel_value=background_pixel_value,
                                tissue_pixel_value=tissue_pixel_value,
                                offset=e,
                            )
                            if not mask_attn_patch:
                                shifted_token_mask = None
                            if not mask_attn_region:
                                shifted_patch_mask = None

                        _, shifted_patch_att, shifted_region_att = get_region_attention_scores(
                            shifted_region,
                            patch_transformer,
                            region_transformer,
                            patch_size,
                            transforms=transforms,
                            downscale=downscale,
                            patch_attn_mask=shifted_patch_mask,
                            token_attn_mask=shifted_token_mask,
                            patch_device=patch_device,
                            region_device=region_device,
                            compute_patch_attention=compute_patch_attention,
                        )
                        shifted_patch_attention[(dx,dy)].append(shifted_patch_att)
                        shifted_region_attention[(dx,dy)].append(shifted_region_att)

            with tqdm.tqdm(
                range(nhead_region),
                desc=f"Processing region [{k+1}/{nregions}]",
                unit="head",
                leave=False,
                disable=not verbose,
            ) as t2:
                for j in t2:

                    region_scores = region_att[j]
                    region_overlay = np.ones_like(region_scores)

                    if granularity.region:
                        directions = [
                            (-1, -1),  # Northwest
                            (0, -1),   # North
                            (1, -1),   # Northeast
                            (-1, 0),   # West
                            (1, 0),    # East
                            (-1, 1),   # Southwest
                            (0, 1),    # South
                            (1, 1),    # Southeast
                        ]
                        offset = min(granularity.offset.patch, granularity.offset.region)
                        for dx, dy in directions:
                            for edx, e in enumerate(range(offset, region_size, offset)):
                                e_scaled = e // downscale
                                s = region_size // downscale
                                shifted_region_scores = np.zeros_like(region_scores)

                                # determine the overlapping region based on the sign of dx and dy
                                x_start = max(0, dx * e_scaled)
                                x_end = s - max(0, -dx * e_scaled)
                                y_start = max(0, dy * e_scaled)
                                y_end = s - max(0, -dy * e_scaled)

                                shifted_x_start = max(0, -dx * e_scaled)
                                shifted_x_end = s - max(0, dx * e_scaled)
                                shifted_y_start = max(0, -dy * e_scaled)
                                shifted_y_end = s - max(0, dy * e_scaled)

                                # fill only the overlapping values
                                shifted_region_scores[x_start:x_end, y_start:y_end] = shifted_region_attention[(dx, dy)][edx][j][shifted_x_start:shifted_x_end, shifted_y_start:shifted_y_end]
                                region_scores += shifted_region_scores

                                # increment the overlay only for the overlapping positions
                                region_overlay[x_start:x_end, y_start:y_end] += 1

                        region_scores = region_scores / region_overlay

                    head_min_region[j] = min(head_min_region[j], region_scores.min())
                    head_max_region[j] = max(head_max_region[j], region_scores.max())

                    region_attention_head_dir = raw_attention_dir / f"head_{j}"
                    region_attention_head_dir.mkdir(exist_ok=True, parents=True)
                    np.save(Path(region_attention_head_dir, f"{x}_{y}.npy"), region_scores)
                    np.save(Path(region_attention_head_dir, f"{x}_{y}_overlay.npy"), region_overlay)

                    if compute_patch_attention:

                        #TODO: fix granularity based on new multi-direction logic (see above)

                        with tqdm.tqdm(
                            range(nhead_patch),
                            desc=f"Region head [{j+1}/{nhead_region}]",
                            unit="head",
                            leave=False,
                            disable=not verbose,
                        ) as t3:
                            for i in t3:

                                patch_scores = patch_att[:, i, :, :]

                                if granularity.patch:
                                    patch_overlay = np.ones_like(patch_scores)
                                    offset = min(granularity.offset.patch, granularity.offset.region)
                                    for edx, e in enumerate(range(offset, region_size, offset)):
                                        e_scaled = e // downscale
                                        shifted_patch_att_scores = np.zeros_like(patch_scores)
                                        shifted_patch_att_scores[
                                            e_scaled:s, e_scaled:s
                                        ] = shifted_patch_att[edx][:, i, :, :][
                                            : (s - e_scaled), : (s - e_scaled)
                                        ]
                                        patch_scores += shifted_patch_att_scores
                                        patch_overlay[e_scaled:s, e_scaled:s] += 1

                                    patch_scores = patch_scores / patch_overlay

                                head_min_patch[j] = min(head_min_patch[j], patch_scores.min())
                                head_max_patch[j] = max(head_max_patch[j], patch_scores.max())

                                patch_attention_head_dir = attention_dir / "patch" / f"head_{j}"
                                patch_attention_head_dir.mkdir(exist_ok=True, parents=True)
                                np.save(Path(patch_attention_head_dir, f"{x}_{y}.npy"), patch_scores)
                                np.save(Path(patch_attention_head_dir, f"{x}_{y}_overlay.npy"), patch_overlay)

    with tqdm.tqdm(
        coordinates,
        desc="Overlaying attention scores",
        unit="region",
        leave=True,
        disable=not verbose,
    ) as t1:
        for k, (x,y) in enumerate(t1):

            slide_att_scores = slide_attention[k]

            region_arr = wsi.get_patch(
                x,
                y,
                region_size_resized,
                region_size_resized,
                spacing=region_spacing,
                center=False,
            )
            region = Image.fromarray(region_arr)
            if region_size != region_size_resized:
                region = region.resize((region_size, region_size))

            s = region_size // downscale
            # given region is an RGB image, so the default filter for resizing is Resampling.BICUBIC
            # which is fine as we're resizing the image here, not attention scores
            save_region = np.array(region.resize((s, s)))

            with tqdm.tqdm(
                range(nhead_region),
                desc=f"Processing region [{k+1}/{nregions}]",
                unit="head",
                leave=False,
                disable=not verbose,
            ) as t2:
                for j in t2:

                    region_scores = np.load(Path(raw_attention_dir, f"head_{j}", f"{x}_{y}.npy"))
                    normalized_region_scores = normalize_region_scores(region_scores, size=(s,) * 2, min_val=head_min_region[j], max_val=head_max_region[j])
                    region_overlay = np.load(Path(raw_attention_dir, f"head_{j}", f"{x}_{y}_overlay.npy"))

                    with tqdm.tqdm(
                        range(nhead_patch),
                        desc=f"Region head [{j+1}/{nhead_region}]",
                        unit="head",
                        leave=False,
                        disable=not verbose,
                    ) as t3:
                        for i in t3:

                            if compute_patch_attention:

                                patch_scores = np.load(Path(attention_dir, "patch", f"head_{j}", f"{x}_{y}.npy"))
                                normalized_patch_scores = concat_patch_scores(
                                    patch_scores,
                                    region_size=region_size,
                                    patch_size=patch_size,
                                    size=(s // n_patch,) * 2,
                                    min_val=head_min_patch[i],
                                    max_val=head_max_patch[i],
                                )
                                patch_overlay = np.load(Path(attention_dir, "patch", f"head_{j}", f"{x}_{y}_overlay.npy"))

                                if granularity.region:
                                    if level == "global":
                                        if method == "average":
                                            n = 2
                                            score = (
                                                normalized_region_scores * region_overlay * (1-gamma)
                                                + normalized_patch_scores * patch_overlay * (1-gamma)
                                            )
                                            if gamma < 1:
                                                score = score / (region_overlay * (1-gamma) + patch_overlay * (1-gamma))
                                        elif method == "multiply":
                                            score = (
                                                (normalized_region_scores ** (1-gamma)) * region_overlay
                                                * (normalized_patch_scores ** (1-gamma)) * patch_overlay
                                            )
                                    elif level == "local":
                                        if method == "average":
                                            n = 1
                                            score = (
                                                normalized_region_scores * region_overlay * gamma
                                                + normalized_patch_scores * patch_overlay * (1-gamma)
                                            ) / (region_overlay * gamma + patch_overlay * (1-gamma))
                                        elif method == "multiply":
                                            score = (
                                                (normalized_region_scores ** gamma) * region_overlay
                                                * (normalized_patch_scores ** (1-gamma)) * patch_overlay
                                            )
                                    else:
                                        raise ValueError(
                                            "Invalid level. Choose from ['global', 'local']"
                                        )
                                else:
                                    if level == "global":
                                        if method == "average":
                                            n = 2
                                            score = normalized_region_scores * (1-gamma) + normalized_patch_scores * (1-gamma)
                                        elif method == "multiply":
                                            score = normalized_region_scores ** (1-gamma) * normalized_patch_scores ** (1-gamma)
                                    elif level == "local":
                                        if method == "average":
                                            n = 1
                                            score = normalized_region_scores * gamma + normalized_patch_scores * (1-gamma)
                                        elif method == "multiply":
                                            score = normalized_region_scores ** gamma * normalized_patch_scores ** (1-gamma)
                                    else:
                                        raise ValueError(
                                            "Invalid level. Choose from ['global', 'local']"
                                        )

                            else:

                                if level == "global":
                                    if method == "average":
                                        n = 2
                                        score = normalized_region_scores * (1-gamma)
                                    elif method == "multiply":
                                        score = normalized_region_scores ** (1-gamma)
                                elif level == "local":
                                    if method == "average":
                                        n = 1
                                        score = normalized_region_scores * gamma
                                    elif method == "multiply":
                                        score = normalized_region_scores ** gamma
                                else:
                                    raise ValueError(
                                        "Invalid level. Choose from ['global', 'local']"
                                    )

                            if method == "average":
                                score += slide_att_scores * gamma
                                score = score / (n*(1-gamma)+(3-n)*gamma)
                            elif method == "multiply":
                                score = score * (slide_att_scores ** gamma)

                            if threshold != None:

                                if compute_patch_attention:
                                    thresh_factorized_hm_output_dir = Path(
                                        visualization_dir, "thresholded", f"region-head-{j}-patch-head-{i}"
                                    )
                                else:
                                    thresh_factorized_hm_output_dir = Path(
                                        visualization_dir, "thresholded", f"region-head-{j}"
                                    )
                                thresh_factorized_hm_output_dir.mkdir(exist_ok=True, parents=True)

                                att_mask = score.copy()
                                att_mask[att_mask < threshold] = 0
                                att_mask[att_mask >= threshold] = 0.95

                                color_block = (cmap(att_mask) * 255)[:, :, :3].astype(
                                    np.uint8
                                )
                                region_hm = cv2.addWeighted(
                                    color_block,
                                    alpha,
                                    save_region.copy(),
                                    1 - alpha,
                                    0,
                                    save_region.copy(),
                                )
                                region_hm[att_mask == 0] = 0
                                img_inverse = save_region.copy()
                                img_inverse[att_mask == 0.95] = 0
                                region_hm = region_hm + img_inverse
                                img = Image.fromarray(region_hm)
                                img.save(
                                    Path(
                                        thresh_factorized_hm_output_dir,
                                        f"{x}_{y}.png",
                                    )
                                )

                            # save raw factorized attention scores to disk
                            region_attention_head_dir = factorized_attention_dir / f"head_{j}"
                            region_attention_head_dir.mkdir(exist_ok=True, parents=True)
                            np.save(Path(region_attention_head_dir, f"{x}_{y}.npy"), score)

                            color_block = (cmap(score) * 255)[:, :, :3].astype(np.uint8)
                            region_hm = cv2.addWeighted(
                                color_block,
                                alpha,
                                save_region.copy(),
                                1 - alpha,
                                0,
                                save_region.copy(),
                            )
                            if compute_patch_attention:
                                factorized_hm_output_dir = Path(visualization_dir, "regular", f"region-head-{j}-patch-head-{i}")
                            else:
                                factorized_hm_output_dir = Path(visualization_dir, "regular", f"region-head-{j}")
                            factorized_hm_output_dir.mkdir(exist_ok=True, parents=True)
                            img = Image.fromarray(region_hm)
                            img.save(Path(factorized_hm_output_dir, f"{x}_{y}.png",))

    return factorized_attention_dir


def attention_cmap_overlay(attention_map: np.ndarray, canvas: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap, alpha: float) -> Image.Image:
    """
    Create an OpenCV-style heatmap overlay:
    - attention_map: (H, W), values in [0,1]
    - wsi_background: (H, W, 3), RGB uint8
    Returns: blended RGB image (H, W, 3)
    """
    # create colormap (RGB only)
    colored_attention = (cmap(attention_map)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)
    # blend colored attention with canvas
    overlay = cv2.addWeighted(
        colored_attention,
        alpha,
        canvas,
        1 - alpha,
        0,
    )
    overlay_img = Image.fromarray(overlay)
    return overlay_img


def attention_contour_overlay(attention_map: np.ndarray, canvas: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap, threshold: float, alpha: float) -> Image.Image:
    """
    Given a stitched attention map and background canvas, overlays contours of attention regions.

    Parameters:
    - attention_map: (H, W) float32, normalized [0,1]
    - canvas: (H, W, 3) uint8 RGB image
    - threshold: float, threshold on attention
    - blur_sigma: optional smoothing before contour extraction

    Returns:
    - PIL Image with contour overlay
    """
    # threshold
    binary = attention_map > threshold
    binary_closed = binary_closing(binary, iterations=20).astype(np.uint8) * 255

    # find contours
    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rgba = cmap(0.0)
    rgb = tuple(int(255 * c) for c in rgba[:3])
    filled_contour = np.full_like(canvas, rgb, dtype=np.uint8)

    for cnt in contours:
        # Create a mask for the contour
        mask = np.zeros(attention_map.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # Compute mean attention in the contour region
        mean_val = cv2.mean(attention_map, mask=mask)[0]
        rgba = cmap(mean_val)
        rgb = tuple(int(255 * c) for c in rgba[:3])

        # Fill the contour with the color
        cv2.drawContours(filled_contour, [cnt], -1, rgb, thickness=cv2.FILLED)

        # draw contour outline
        outline_color = (255, 255, 255)  # white
        cv2.drawContours(filled_contour, [cnt], -1, outline_color, thickness=1).astype(np.uint8)

    blended = cv2.addWeighted(filled_contour, alpha, canvas, 1 - alpha, 0)
    contour_img = Image.fromarray(blended)
    return contour_img


def stitch_slide_heatmaps(
    wsi_path: Path,
    attention_dir: Path,
    output_dir: Path,
    spacing: float,
    tolerance: float,
    patch_size: int,
    name: str = None,
    suffix: str = None,
    segmentation_parameters: SegmentationParameters | None = None,
    downscale: int = 1,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    segmentation_mask_path: Path = None,
    smoothing: bool = True,
    opacity: float = 0.5,
    threshold: Optional[float] = None,
    restrict_to_tissue: bool = False,
    verbose: bool = True,
):
    """
    Returns region-level heatmaps stitched together at the slide-level.

    Args:
    - wsi_path (Path): path to the whole slide image
    - attention_dir (Path): path to the directory containing unnormalized attention scores as .npy files
    - output_dir (Path): output directory for saving heatmaps
    - spacing (float): pixel spacing (in mpp) at which regions were extracted for that slide
    - name (str): file naming template
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - cmap (matplotlib.colors.LinearSegmentedColormap): colormap for plotting heatmaps
    - opacity (float): set opacity for non-tissue content on stitched heatmap
    - threshold (float): filter out attention scores lower than this value (set to None to disbale thresholding)
    - restrict_to_tissue (bool): whether to restrict highlighted regions to tissue content only
    """
    wsi_object = WholeSlideImage(wsi_path, mask_path=segmentation_mask_path, segment_params=segmentation_parameters)

    vis_level = wsi_object.get_best_level_for_downsample_custom(segmentation_parameters.downsample)
    vis_spacing = wsi_object.get_level_spacing(vis_level)
    wsi_canvas = wsi_object.get_slide(spacing=vis_spacing)
    # x and y axes get inverted when using get_slide method
    width, height, _ = wsi_canvas.shape

    slide_output_dir = Path(output_dir, "visualization", "slide")
    slide_output_dir.mkdir(exist_ok=True, parents=True)

    head_dirs = sorted([x for x in attention_dir.iterdir() if x.is_dir()])
    num_heads = 1
    sub_dir = False
    if len(head_dirs) > 0:
        sub_dir = True
        num_heads = len(head_dirs)

    head_min, head_max = None, None
    min_attention_values = Path(attention_dir, 'min.json')
    max_attention_values = Path(attention_dir, 'max.json')
    if min_attention_values.exists():
        with open(min_attention_values, 'r') as f:
            head_min = json.load(f)
    if max_attention_values.exists():
        with open(max_attention_values, 'r') as f:
            head_max = json.load(f)

    if head_min is None or head_max is None:
        head_min = [np.inf] * num_heads
        head_max = [-np.inf] * num_heads
        with tqdm.tqdm(
            range(num_heads),
            desc="Gathering min/max",
            unit="head",
            leave=True,
        ) as t1:
            for i in t1:
                if not sub_dir:
                    attention_scores = [fp for fp in attention_dir.glob("*.npy")]
                    fname = f"{name}"
                else:
                    hd = head_dirs[i]
                    head_name = hd.stem
                    attention_scores = [fp for fp in hd.glob("*.npy")]
                    fname = f"{name}-{head_name}"
                with tqdm.tqdm(
                    attention_scores,
                    desc=f"Gathering min/max for head [{i+1}/{num_heads}]",
                    unit="region",
                    leave=False,
                    disable=not verbose,
                ) as t2:
                    for fp in t2:
                        attention_array = np.load(fp)
                        head_min[i] = min(head_min[i], attention_array.min())
                        head_max[i] = max(head_max[i], attention_array.max())

    stitched_attention = np.zeros((width,height)) # wsi size at vis_level
    with tqdm.tqdm(
        range(num_heads),
        desc="Stitching heatmaps",
        unit="head",
        leave=True,
    ) as t1:
        for i in t1:
            if not sub_dir:
                attention_scores = [fp for fp in attention_dir.glob("*.npy")]
                fname = f"{name}"
            else:
                hd = head_dirs[i]
                head_name = hd.stem
                attention_scores = [fp for fp in hd.glob("*.npy")]
                fname = f"{name}-{head_name}"
            with tqdm.tqdm(
                attention_scores,
                desc=f"Stitching attention head [{i+1}/{num_heads}]",
                unit="region",
                leave=False,
                disable=not verbose,
            ) as t2:
                for fp in t2:
                    # x, y defined w.r.t level 0
                    x, y = int(fp.stem.split("_")[0]), int(fp.stem.split("_")[1])
                    attention_array = np.load(fp) # (region_size, region_size)
                    w, h = attention_array.shape
                    # head-wise normalized attention array
                    attention_array = (attention_array - head_min[i]) / (head_max[i] - head_min[i] + 1e-8)

                    # need to scale attention_array from spacing level to vis level
                    resize_factor = 1
                    spacing_level, is_within_tolerance = wsi_object.get_best_level_for_spacing(spacing, tolerance)
                    if not is_within_tolerance:
                        natural_spacing = wsi_object.get_level_spacing(spacing_level)
                        resize_factor = spacing / natural_spacing

                    downsample_factor = tuple(dv/ds/resize_factor for dv,ds in zip(wsi_object.level_downsamples[vis_level], wsi_object.level_downsamples[spacing_level]))
                    w_downsampled = int(round(w * downscale / downsample_factor[0], 0))
                    h_downsampled = int(round(h * downscale / downsample_factor[1], 0))

                    attention_array_downsampled = cv2.resize(
                        attention_array,
                        (w_downsampled, h_downsampled),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    # need to scale coordinates from level 0 to vis_level
                    downsample_factor = tuple(dv/ds for dv,ds in zip(wsi_object.level_downsamples[vis_level], wsi_object.level_downsamples[0]))
                    x_downsampled = int(round(x / downsample_factor[0], 0))
                    y_downsampled = int(round(y / downsample_factor[1], 0))

                    stitched_attention[
                        y_downsampled : min(y_downsampled + h_downsampled, width),
                        x_downsampled : min(x_downsampled + w_downsampled, height),
                    ] = attention_array_downsampled[
                        : min(h_downsampled, width - y_downsampled),
                        : min(w_downsampled, height - x_downsampled),
                    ]

                    if restrict_to_tissue:
                        tissue_mask = wsi_object.binary_mask[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                        ]
                        tissue_mask = (tissue_mask > 0).astype(int)
                        stitched_attention[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                        ] = stitched_attention[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                        ] * tissue_mask

            if smoothing:
                scale = spacing / vis_spacing
                sigma = int(scale * patch_size * 0.25)
                if threshold is not None:
                    stitched_attention[stitched_attention < threshold] = 0
                smoothed_attention = gaussian_filter(stitched_attention, sigma=sigma)
                # gamma correction
                # smoothed_attention = np.power(smoothed_attention, 0.5)
                smoothed_attention[smoothed_attention < 0.5] = 0

                overlayed_attention = attention_cmap_overlay(smoothed_attention, wsi_canvas, cmap, opacity)
                # overlayed_attention = attention_contour_overlay(smoothed_attention, wsi_canvas, cmap, threshold=0.5, alpha=opacity)
            else:
                overlayed_attention = attention_cmap_overlay(stitched_attention, wsi_canvas, cmap, opacity)

            if suffix:
                fname = f"{fname}-{suffix}"
            stitched_hm_path = Path(slide_output_dir, f"{fname}.png")
            overlayed_attention.save(stitched_hm_path, dpi=(300, 300))

    return slide_output_dir


def display_stitched_heatmaps(
    wsi_path: Path,
    heatmap_dir: Path,
    output_dir: Path,
    name: str,
    display_patching: bool = False,
    draw_grid: bool = True,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    coordinates_dir: Optional[Path] = None,
    downsample: int = 32,
    font_fp: Path = Path("arial.ttf"),
):
    """
    Display stitched heatmaps from multiple heads together, optionally alongside a visualization of patch extraction results.

    Args:
    - wsi_path (Path): path to the whole slide image
    - heatmap_dir (Path): path to the directory containing heatmaps as .png files
    - output_dir (Path): output directory for saving heatmaps
    - name (str): file naming template
    - display_patching (bool): whether to display patch extraction results alongside stitched heatmaps
    - coordinates_dir (Optional[Path]): if display_patching is True, point to the root folder where extracted coordinates were saved
    - region_size (Optional[int]): if display_patching is True, indicates the size of the extracted regions
    - downsample (int): uses this value to find the closest downsample level in the WSI for slide-level heatmap visualization
    - key (str): if display_patching is True, key used to retrieve extracted regions' coordinates
    - font_fp (Path): path to the font used for matplotlib figure text (e.g. for title & subtitles)
    """
    slide_id = wsi_path.stem

    heatmap_paths = sorted(list(heatmap_dir.glob(f"*.png")))
    heatmaps = {
        fp.stem: Image.open(fp) for fp in heatmap_paths
    }
    w, h = next(iter(heatmaps.values())).size

    if display_patching:

        coordinates_file = coordinates_dir / f"{slide_id}.npy"
        coordinates_arr = np.load(coordinates_file)

        try:
            coordinates = list(zip(coordinates_arr[:,0], coordinates_arr[:,1]))
            region_size_resized = coordinates_arr[0][2]
            region_level = coordinates_arr[0][3]
            resize_factor = coordinates_arr[0][4]
        except Exception as e:
            coordinates = list(zip(coordinates_arr["x"], coordinates_arr["y"]))
            region_size_resized = coordinates_arr["tile_size_resized"][0]
            region_level = coordinates_arr["tile_level"][0]
            resize_factor = coordinates_arr["resize_factor"][0]

        region_size = int(round(region_size_resized / resize_factor,0))

        wsi_object = WholeSlideImage(wsi_path)
        vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
        vis_spacing = wsi_object.get_level_spacing(vis_level)
        slide_canvas = wsi_object.get_slide(spacing=vis_spacing)

        # scale region_size from region_level to vis_level
        downsample_factor = tuple(dv/ds for dv,ds in zip(wsi_object.level_downsamples[region_level], wsi_object.level_downsamples[vis_level]))
        downsampled_region_size = tuple(int(round(region_size * factor, 0)) for factor in downsample_factor)

        patching_im = DrawMapFromCoords(
            slide_canvas,
            wsi_object,
            coordinates,
            downsampled_region_size,
            vis_level,
            draw_grid=draw_grid,
        )
        patching_im.save(
            Path(output_dir, f"tiling.png"),
        )
        data = [(f"{region_size}x{region_size} patching", patching_im)] + [
            (k, v) for k, v in heatmaps.items()
        ]
        heatmaps = OrderedDict(data)

    nhm = len(heatmaps)
    pad = 20

    sm = plt.cm.ScalarMappable(cmap=cmap)
    fig, ax = plt.subplots(dpi=150)
    plt.colorbar(sm, ax=ax)
    ax.remove()
    plt.yticks(fontsize='large')
    color_bar_name = "color_bar.png"
    plt.savefig(color_bar_name, bbox_inches='tight', dpi=150)
    plt.close()
    cbar = Image.open(color_bar_name)
    os.remove(color_bar_name)
    w_cbar, h_cbar = cbar.size

    modes = set([hm.mode for hm in heatmaps.values()])
    mode = "RGB"
    if len(modes) > 1:
        mode = "RGBA"
    canvas = Image.new(
        size=(w * nhm + pad * (nhm + 1) + 2 * pad + w_cbar, h + 2 * pad), mode=mode, color=(255,) * len(mode)
    )

    font = None
    with tqdm.tqdm(
        heatmaps.items(),
        desc="Grouping stitched heatmaps into a single image",
        unit="heatmap",
        leave=True,
    ) as t:
        for i, (txt, hm) in enumerate(t):
            if not font:
                fontsize = 1  # starting font size
                fraction = 0.2  # portion of slide width we want text width to be
                font = ImageFont.truetype(font_fp, fontsize)
                while font.getsize(txt)[0] < fraction * w:
                    # iterate until the text size is just larger than the criteria
                    fontsize += 1
                    font = ImageFont.truetype(font_fp, fontsize)
                # optionally de-increment to be sure it is less than criteria
                fontsize -= 1
                font = ImageFont.truetype(font_fp, fontsize)

            x, y = w * i + pad * (i + 1), pad
            canvas.paste(hm, (x, y))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (x, pad // 2),
                txt,
                (0, 0, 0),
                font=font,
            )

    x, y = w * nhm + pad * (nhm + 1), (h + 2 * pad - h_cbar) // 2
    canvas.paste(cbar, (x, y))

    stitched_hm_path = Path(output_dir, f"{name}.png")
    canvas.save(stitched_hm_path, dpi=(300, 300))