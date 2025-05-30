import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
import wholeslidedata as wsd
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

import source.distributed as distributed
import source.vision_transformer as vits
from source.foundation_models import FoundationModelFactory
from source.model_utils import Attn_Net_Gated
from source.utils import hf_login, update_state_dict
from source.wsi import WholeSlideImage
from source.augmentations import RegionUnfolding


def create_transforms(model, level):
    if level == "global":
        return model.get_transforms()
    elif level == "local":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                RegionUnfolding(model.tile_size),
                model.get_transforms(),
            ]
        )
    else:
        raise ValueError(f"Unknown model level: {level}")


def get_encoder(
        model_options: Dict,
        model_device: torch.device = torch.device("cpu"),
):
    if model_options.name is not None:
        hf_login()
        encoder = FoundationModelFactory(model_options).get_model()
    elif model_options.arch is not None:
        encoder = get_patch_model(
            model_options.pretrained_weights,
            arch=model_options.arch,
            mask_attn=model_options.mask_attn,
            device=model_device,
            verbose=distributed.is_main_process(),
        )
    return encoder


def get_patch_model(
    pretrained_weights: str,
    arch: str = "vit_small",
    mask_attn: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    checkpoint_key = "teacher"
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    patch_model = vits.__dict__[arch](
        patch_size=16,
        num_classes=0,
        mask_attn=mask_attn,
    )
    for p in patch_model.parameters():
        p.requires_grad = False
    patch_model.eval()
    patch_model.to(device)

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
        state_dict, msg = update_state_dict(patch_model.state_dict(), state_dict)
        patch_model.load_state_dict(state_dict, strict=True)
        if verbose:
            print(f"Pretrained weights found at {pretrained_weights}")
            print(msg)

    return patch_model


def get_region_model(
    state_dict,
    input_embed_dim: int,
    arch: str = "vit4k_xs",
    region_size: int = 4096,
    patch_size: int = 256,
    mask_attn: bool = False,
    img_size_pretrained: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    if device is None:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    region_model = vits.__dict__[arch](
        img_size=region_size,
        patch_size=patch_size,
        input_embed_dim=input_embed_dim,
        num_classes=0,
        mask_attn=mask_attn,
        img_size_pretrained=img_size_pretrained,
    )
    for p in region_model.parameters():
        p.requires_grad = False
    region_model.eval()
    region_model.to(device)

    if verbose:
        print("Loading weights for region-level Transformer...")
    state_dict = {k.replace("vit_region.", ""): v for k, v in state_dict.items()}
    state_dict, msg = update_state_dict(region_model.state_dict(), state_dict)
    region_model.load_state_dict(state_dict, strict=True)
    if verbose:
        print(msg)

    return region_model


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


def get_slide_model(
    state_dict,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    slide_model = SlideAgg()
    for p in slide_model.parameters():
        p.requires_grad = False
    slide_model.eval()
    slide_model.to(device)

    if verbose:
        print("Loading weights for slide-level Transformer...")
    state_dict, msg = update_state_dict(slide_model.state_dict(), state_dict)
    slide_model.load_state_dict(state_dict, strict=True)
    if verbose:
        print(msg)
    return slide_model


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
    region_size: int = 4096,
    patch_size: int = 256,
    size: Optional[Tuple[int, int]] = None,
    method: str = 'min-max',
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
):
    n_patch = region_size // patch_size
    if method == 'min-max':
        assert min_val is not None and max_val is not None, "min_val and max_val must be provided for min-max normalization"
        rank = lambda v: (v - min_val) / (max_val - min_val)
    elif method == 'rank':
        rank = lambda v: (rankdata(v, method='min')-1) / len(v)
    color_block = [
        rank(attn.flatten()).reshape(size) for attn in attns
    ]  # [(256, 256)] of length len(attns)
    color_hm = np.concatenate(
        [
            np.concatenate(color_block[i : (i + n_patch)], axis=1)
            for i in range(0, n_patch**2, n_patch)
        ]
    )
    # (16*256, 16*256)
    return color_hm


def normalize_region_scores(
    attn,
    size: Optional[Tuple[int, int]] = None,
    method: str = 'min-max',
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
):
    if method == 'min-max':
        assert min_val is not None and max_val is not None, "min_val and max_val must be provided for min-max normalization"
        rank = lambda v: (v - min_val) / (max_val - min_val)
    elif method == 'rank':
        rank = lambda v: (rankdata(v, method='min')-1) / len(v)
    color_hm = rank(attn.flatten()).reshape(size)  # (4096, 4096)
    return color_hm


def normalize_slide_scores(
    attn,
    method: str = 'min-max',
):
    if method == 'min-max':
        min_val, max_val = attn.min(), attn.max()
        rank = lambda v: (v - min_val) / (max_val - min_val)
    elif method == 'rank':
        rank = lambda v: (rankdata(v, method='min')-1) / len(v)
    color_hm = rank(attn)
    return color_hm


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
        tile = wsi_object.wsi.get_patch(x, y, width, height, spacing=vis_spacing, center=False)

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
    slide_path: str,
    segmentation_mask_path: str,
    x: int,
    y: int,
    region_size: int = 4096,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    spacing: float = 0.5,
    backend: str = 'pyvips',
    downsample: int = 4,
    background_pixel_value: int = 0,
    tissue_pixel_value: int = 1,
    pct_thresh: float = 0.0,
    offset: Optional[int] = None,
):
    # load the slide
    wsi = wsd.WholeSlideImage(Path(slide_path), backend=backend)
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
    # scale patch_size and mini_patch_size
    scaled_patch_size = patch_size // sr
    scaled_mini_patch_size = mini_patch_size // sr

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

    region_mini_patches = []
    for p in region_patches:
        mp = split_into_blocks(p, scaled_mini_patch_size)
        region_mini_patches.append(mp)
    region_mini_patches = np.stack(region_mini_patches)
    tissue = region_mini_patches == tissue_pixel_value
    pct = np.sum(tissue, axis=(-2, -1)) / tissue[0][0].size # (npatch**2, nminipatch**2)
    pct = torch.Tensor(pct)

    mask_mini_patch = (pct > pct_thresh).int()  # (num_patches, nminipatch**2)
    # add the [CLS] token to the mask
    cls_token = mask_mini_patch.new_ones((mask_mini_patch.size(0),1))
    mask_mini_patch = torch.cat((cls_token, mask_mini_patch), dim=1)  # [num_patches, nminipatch**2+1]
    # infer patch-level mask
    pct_patch = torch.sum(pct, axis=-1) / pct.numel()
    mask_patch = (pct_patch > pct_thresh).int().unsqueeze(0)
    # add the [CLS] token to the mask
    cls_token = mask_patch.new_ones((mask_patch.size(0),1))
    mask_patch = torch.cat((cls_token, mask_patch), dim=1)  # [1, num_patches+1]

    return mask_patch, mask_mini_patch


def generate_masks(
    slide_id: str,
    slide_path: str,
    segmentation_mask_path: str,
    region_dir: str,
    region_size: int = 4096,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    spacing: float = 0.5,
    backend: str = 'asap',
    downsample: int = 4,
    region_format: str = "jpg",
    tissue_pixel_value: int = 1,
    pct_thresh: float = 0.0,
):
    # load the slide
    wsi = wsd.WholeSlideImage(Path(slide_path), backend=backend)
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
    # scale patch_size and mini_patch_size
    scaled_patch_size = patch_size // sr
    scaled_mini_patch_size = mini_patch_size // sr
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

        region_mini_patches = []
        for p in region_patches:
            mp = split_into_blocks(p, scaled_mini_patch_size)
            region_mini_patches.append(mp)
        region_mini_patches = np.stack(region_mini_patches)
        tissue = region_mini_patches == tissue_pixel_value
        tissue_pct = np.sum(tissue, axis=(-2, -1)) / tissue[0][0].size
        tissue_pcts.append(tissue_pct)

    pct = np.stack(tissue_pcts)  # (M, npatch**2, nminipatch**2)
    pct = torch.Tensor(pct)

    mask_mini_patch = (pct > pct_thresh).int()  # (M, num_patches, nminipatch**2)
    # add the [CLS] token to the mask
    cls_token = mask_mini_patch.new_ones((mask_mini_patch.size(0),mask_mini_patch.size(1),1))
    mask_mini_patch = torch.cat((cls_token, mask_mini_patch), dim=2)  # [M, num_patches, nminipatch**2+1]
    # infer patch-level mask
    pct_patch = torch.sum(pct, axis=-1) / pct[0].numel()
    mask_patch = (pct_patch > pct_thresh).int()
    # add the [CLS] token to the mask
    cls_token = mask_patch.new_ones((mask_patch.size(0),1))
    mask_patch = torch.cat((cls_token, mask_patch), dim=1)  # [M, num_patches+1]

    return mask_patch, mask_mini_patch


def get_patch_attention_scores(
    patch: Image,
    patch_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    mini_patch_size: int = 16,
    downscale: int = 1,
    mini_patch_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
):
    """
    Forward pass in patch-level Transformer with attention scores saved.

    Args:
    - patch (PIL.Image): input patch
    - patch_model (nn.Module): patch-level Transformer
    - mini_patch_size (int): size of mini-patches used for unrolling input patch
    - downscale (int): how much to downscale the output patch by (e.g. downscale=4 will resize 256x256 regions to 64x64)
    - patch_device (torch.device): device on which patch_model is

    Returns:
    - attention (torch.Tensor): [1, nhead, patch_size/downscale, patch_size/downscale] tensor of attention maps
    """
    patch_size = patch.size[0]
    n_minipatch = patch_size // mini_patch_size

    with torch.no_grad():
        batch = transforms(patch).unsqueeze(0)  # (1, 3, patch_size, patch_size)
        batch = batch.to(patch_device, non_blocking=True)

        attention = patch_model.get_last_selfattention(
            batch,
            mask=mini_patch_attn_mask,
        )  # (1, 6, n_minipatch**2+1, n_minipatch**2+1)
        nh = attention.shape[1]  # number of head
        attention = attention[:, :, 0, 1:].reshape(
            n_minipatch**2, nh, -1
        )  # (1, 6, 1, n_minipatch**2) -> (n_minipatch**2, 6, 1)
        attention = attention.reshape(
            1, nh, n_minipatch, n_minipatch
        )  # (1, 6, n_minipatch, n_minipatch)
        attention = (
            nn.functional.interpolate(
                attention, scale_factor=int(mini_patch_size / downscale), mode="nearest"
            )
            .cpu()
            .numpy()
        )  # (1, 6, patch_size, patch_size) when downscale = 1
        # 'nearest' interpolation guarantees the values in the up-sampled array
        # lie in the same set as the values in the original array

        if downscale != 1:
            batch = nn.functional.interpolate(
                batch, scale_factor=(1 / downscale), mode="nearest"
            )

    return tensorbatch2im(batch), attention


def get_region_attention_scores(
    region: Image,
    patch_model: nn.Module,
    region_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    patch_attn_mask: Optional[torch.Tensor] = None,
    mini_patch_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    compute_patch_attention: bool = True,
):
    """
    Forward pass in hierarchical model with attention scores saved.

    Args:
    - region (PIL.Image): input region
    - patch_model (nn.Module): patch-level Transformer
    - region_model (nn.Module): region-level Transformer
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output regions by (e.g. downscale=4 will resize 4096x4096 regions to 1024x1024)
    - patch_device (torch.device): device on which patch_model is
    - region_device (torch.device): device on which region_model is

    Returns:
    - np.array: [n_patch**2, patch_size/downscale, patch_size/downscale, 3] array sequence of image patch_size-sized patches from the input region.
    - patch_attention (torch.Tensor): [n_patch**2, nhead, patch_size/downscale, patch_size/downscale] tensor sequence of attention maps for patch_size-sized patches.
    - region_attention (torch.Tensor): [nhead, region_size/downscale, region_size/downscale] tensor sequence of attention maps for input region.
    """
    region_size = region.size[0]
    n_patch = region_size // patch_size
    n_minipatch = patch_size // mini_patch_size

    with torch.no_grad():
        patches = transforms(region)  # [n_patch**2, 3, patch_size, patch_size]
        patches = patches.to(patch_device, non_blocking=True)
        if mini_patch_attn_mask is not None:
            mini_patch_attn_mask = mini_patch_attn_mask.to(patch_device, non_blocking=True)
        patch_features = patch_model(patches, mask=mini_patch_attn_mask)  # (n_patch**2, 384)

        patch_attention = None
        if compute_patch_attention:
            patch_attention = patch_model.get_last_selfattention(
                patches,
                mask=mini_patch_attn_mask,
            )  # (n_patch**2, nhead, n_minipatch**2+1, n_minipatch**2+1)
            nh = patch_attention.shape[1]  # number of head
            patch_attention = patch_attention[:, :, 0, 1:].reshape(
                n_patch**2, nh, -1
            )  # (n_patch**2, nhead, n_minipatch**2)
            patch_attention = patch_attention.reshape(
                n_patch**2, nh, n_minipatch, n_minipatch
            )  # (n_patch**2, nhead, n_minipatch, n_minipatch)
            patch_attention = (
                nn.functional.interpolate(
                    patch_attention,
                    scale_factor=int(mini_patch_size / downscale),
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
        region_attention = region_model.get_last_selfattention(
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
    slide_path: Path,
    coordinates_dir: Path,
    patch_model: nn.Module,
    region_model: nn.Module,
    slide_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    patch_size: int = 256,
    downscale: int = 1,
    granular: bool = False,
    offset: int = 1024,
    spacing: Optional[float] = None,
    patch_attn_mask: Optional[torch.Tensor] = None,
    mini_patch_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    slide_device: torch.device = torch.device("cuda:0"),
    main_process: bool = True,
):
    """
    Forward pass in hierarchical model with attention scores saved.

    Args:
    - slide_path (Path): path to the slide
    - coordinates_dir (Path): path to root folder containing tiles coordinates
    - patch_model (nn.Module): patch-level Transformer
    - region_model (nn.Module): region-level Transformer
    - slide_model (nn.Module): slide-level Transformer
    - patch_size (int): size of patches used for unrolling input region
    - downscale (int): how much to downscale the output regions by (e.g. downscale=4 will resize 4096x4096 regions to 1024x1024)
    - patch_device (torch.device): device on which patch_model is
    - region_device (torch.device): device on which region_model is
    - main_process (bool): controls tqdm display when running with multiple gpus

    Returns:
    - np.array: [n_patch**2, patch_size/downscale, patch_size/downscale, 3] array sequence of image patch_size-sized patches from the input region.
    - patch_attention (torch.Tensor): [n_patch**2, nhead, patch_size/downscale, patch_size/downscale] tensor sequence of attention maps for patch_size-sized patches.
    - region_attention (torch.Tensor): [nhead, region_size/downscale, region_size/downscale] tensor sequence of attention maps for input region.
    """
    wsi = wsd.WholeSlideImage(slide_path, backend="asap")

    slide_id = slide_path.stem.replace(" ", "_")
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
    region_size = int(region_size_resized / resize_factor)

    if granular:

        assert slide_path
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
                            disable=not main_process,
                        ) as t1:

                            for k, (x,y) in enumerate(t1):

                                if offset_x == offset_y == 0:
                                    coords.append((x, y))

                                new_x = max(0, min(x + ix * offset_x * offset_, w))
                                new_y = max(0, min(y + iy * offset_y * offset_, h))

                                # if new_x == 0:
                                #     if reached_left_border:
                                #         break
                                #     elif offset_y == (ncomp - 1):
                                #         reached_left_border = True
                                # if new_x == w - region_size:
                                #     continue
                                # if new_y == 0:
                                #     continue
                                # if new_y == h - region_size:
                                #     continue

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
                                    mpm = None
                                    if mini_patch_attn_mask is not None:
                                        mpm = mini_patch_attn_mask[k].to(patch_device, non_blocking=True)
                                    patch_features = patch_model(patches, mask=mpm)  # (n_patch**2, 384)

                                    regions = (
                                        patch_features.unfold(0, n_patch, n_patch)
                                        .transpose(0, 1)
                                        .unsqueeze(dim=0)
                                    )  # (1, 384, n_patch, n_patch)
                                    regions = regions.to(region_device, non_blocking=True)
                                    pm = None
                                    if patch_attn_mask is not None:
                                        pm = patch_attn_mask[k].unsqueeze(0).to(region_device, non_blocking=True)
                                    region_features = region_model(regions, mask=pm)  # (1, 192)

                                    features.append(region_features)

                        with torch.no_grad():

                            feature_seq = torch.stack(features, dim=0).squeeze(1)  # (M, 192)
                            feature_seq = feature_seq.to(slide_device, non_blocking=True)
                            slide_attention = slide_model(feature_seq, return_attention=True)
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
            disable=not main_process,
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
                    mpm = None
                    if mini_patch_attn_mask is not None:
                        mpm = mini_patch_attn_mask[k].to(patch_device, non_blocking=True)
                    patch_features = patch_model(patches, mask=mpm)  # (n_patch**2, 384)

                    regions = (
                        patch_features.unfold(0, n_patch, n_patch)
                        .transpose(0, 1)
                        .unsqueeze(dim=0)
                    )  # (1, 384, n_patch, n_patch)
                    regions = regions.to(region_device, non_blocking=True)
                    pm = None
                    if patch_attn_mask is not None:
                        pm = patch_attn_mask[k].unsqueeze(0).to(region_device, non_blocking=True)
                    region_features = region_model(regions, mask=pm)  # (1, 192)

                    features.append(region_features)

        with torch.no_grad():

            feature_seq = torch.stack(features, dim=0).squeeze(1)  # (M, 192)
            feature_seq = feature_seq.to(slide_device, non_blocking=True)
            slide_attention = slide_model(feature_seq, return_attention=True)
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


def create_patch_heatmaps_indiv(
    patch: Image,
    patch_model: nn.Module,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    threshold: float = 0.5,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    granular: bool = False,
    offset: int = 16,
    downscale: int = 1,
    patch_device: torch.device = torch.device("cuda:0"),
):
    """
    Creates patch heatmaps (saved individually).

    Args:
    - patch (PIL.Image): input patch
    - patch_model (nn.Module): patch-level Transformer
    - output_dir (Path): output directory for saving heatmaps
    - patch_size (int): size of input patch
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - threshold (float): filter out mini-patches with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - granular (bool): create additional offset patches to get more granular heatmaps
    - offset (int): if granular is True, uses this value to offset patches
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - patch_device (torch.device): device on which patch_model is
    """
    patch1 = patch.copy()
    _, att = get_patch_attention_scores(
        patch1, patch_model, mini_patch_size=mini_patch_size, downscale=downscale, patch_device=patch_device
    )
    offset_ = offset

    if granular:
        offset = int(offset_ * patch_size / 256)
        patch2 = add_margin(
            patch.crop((offset, offset, patch_size, patch_size)),
            top=0,
            left=0,
            bottom=offset,
            right=offset,
            color=(255, 255, 255),
        )
        _, att_2 = get_patch_attention_scores(
            patch2,
            patch_model,
            mini_patch_size=mini_patch_size,
            downscale=downscale,
            patch_device=patch_device,
        )

    save_region = np.array(patch.copy())
    nhead_patch = patch_model.num_heads

    with tqdm.tqdm(
        range(nhead_patch),
        desc="Patch-level Transformer heatmaps",
        unit="head",
        leave=True,
    ) as t:
        for i in t:
            att_scores = normalize_patch_scores(att[:, i, :, :], size=(patch_size,) * 2)

            if granular:
                att_scores_2 = normalize_patch_scores(
                    att_2[:, i, :, :], size=(patch_size,) * 2
                )
                att_scores *= 100
                att_scores_2 *= 100
                new_att_scores_2 = np.zeros_like(att_scores_2)
                new_att_scores_2[offset:patch_size, offset:patch_size] = att_scores_2[
                    : (patch_size - offset), : (patch_size - offset)
                ]
                patch_overlay = np.ones_like(att_scores_2) * 100
                patch_overlay[offset:patch_size, offset:patch_size] += 100
                att_scores = (att_scores + new_att_scores_2) / patch_overlay

            if threshold != None:
                att_mask = att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask >= threshold] = 0.95

                color_block = (cmap(att_mask) * 255)[:, :, :3].astype(np.uint8)
                patch_hm = cv2.addWeighted(
                    color_block,
                    alpha,
                    save_region.copy(),
                    1 - alpha,
                    0,
                    save_region.copy(),
                )
                patch_hm[att_mask == 0] = 0
                img_inverse = save_region.copy()
                img_inverse[att_mask == 0.95] = 0
                patch_hm = patch_hm + img_inverse
                img_thresh = Image.fromarray(patch_hm)
                img_thresh.save(
                    Path(output_dir, f"{patch_size}_head_{i}_thresh.png")
                )

            color_block = (cmap(att_scores) * 255)[:, :, :3].astype(np.uint8)
            patch_hm = cv2.addWeighted(
                color_block, alpha, save_region.copy(), 1 - alpha, 0, save_region.copy()
            )
            img = Image.fromarray(patch_hm)
            img.save(Path(output_dir, f"{patch_size}_head_{i}.png"))


def create_patch_heatmaps_concat(
    patch: Image,
    patch_model: nn.Module,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    fname: str = "patch",
    threshold: float = 0.5,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    granular: bool = False,
    offset: int = 16,
    downscale: int = 1,
    patch_device: torch.device = torch.device("cuda:0"),
):
    """
    Creates patch heatmaps (concatenated for easy comparison)

    Args:
    - patch (PIL.Image): input patch
    - patch_model (nn.Module): patch-level Transformer
    - output_dir (Path): output directory for saving heatmaps
    - patch_size (int): size of input patch
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - fname (str): file naming template
    - threshold (float): filter out mini-patches with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - granular (bool): create additional offset patches to get more granular heatmaps
    - offset (int): if granular is True, uses this value to offset patches
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - patch_device (torch.device): device on which patch_model is
    """
    patch1 = patch.copy()
    _, att = get_patch_attention_scores(
        patch1, patch_model, mini_patch_size=mini_patch_size, downscale=downscale, patch_device=patch_device
    )
    offset_ = offset

    if granular:
        offset = int(offset_ * patch_size / 256)
        patch2 = add_margin(
            patch.crop((offset, offset, patch_size, patch_size)),
            top=0,
            left=0,
            bottom=offset,
            right=offset,
            color=(255, 255, 255),
        )
        _, att_2 = get_patch_attention_scores(
            patch2,
            patch_model,
            mini_patch_size=mini_patch_size,
            downscale=downscale,
            patch_device=patch_device,
        )

    save_region = np.array(patch.copy())
    nhead_patch = patch_model.num_heads

    hms, hms_thresh = [], []

    with tqdm.tqdm(
        range(nhead_patch),
        desc="Patch-level Transformer heatmaps",
        unit="head",
        leave=True,
    ) as t:
        for i in t:
            att_scores = normalize_patch_scores(att[:, i, :, :], size=(patch_size,) * 2)

            if granular:
                att_scores_2 = normalize_patch_scores(
                    att_2[:, i, :, :], size=(patch_size,) * 2
                )
                att_scores *= 100
                att_scores_2 *= 100
                new_att_scores_2 = np.zeros_like(att_scores_2)
                new_att_scores_2[offset:patch_size, offset:patch_size] = att_scores_2[
                    : (patch_size - offset), : (patch_size - offset)
                ]
                patch_overlay = np.ones_like(att_scores_2) * 100
                patch_overlay[offset:patch_size, offset:patch_size] += 100
                att_scores = (att_scores + new_att_scores_2) / patch_overlay

            if threshold != None:
                att_mask = att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask >= threshold] = 0.95

                color_block = (cmap(att_mask) * 255)[:, :, :3].astype(np.uint8)
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
                hms_thresh.append(region_hm)

            color_block = (cmap(att_scores) * 255)[:, :, :3].astype(np.uint8)
            region_hm = cv2.addWeighted(
                color_block, alpha, save_region.copy(), 1 - alpha, 0, save_region.copy()
            )
            hms.append(region_hm)

    hms = [Image.fromarray(img) for img in hms]
    hms_thresh = [Image.fromarray(img) for img in hms_thresh]

    concat_img = getConcatImage(
        [getConcatImage(hms[0:3]), getConcatImage(hms[3:6])], how="vertical"
    )

    concat_img.save(Path(output_dir, f"{fname}_{patch_size}_hm.png"))

    concat_img_thresh = getConcatImage(
        [getConcatImage(hms_thresh[0:3]), getConcatImage(hms_thresh[3:6])],
        how="vertical",
    )
    concat_img_thresh.save(Path(output_dir, f"{fname}_{patch_size}_thresh.png"))



    # """
    # Forward pass in hierarchical model with attention scores saved.

    # Args:
    # - region (PIL.Image): input region
    # - patch_model (torch.nn): patch-level ViT
    # - region_model (torch.nn): region-level Transformer
    # - patch_size (int): size of patches used for unrolling input region
    # - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    # - downscale (int): how much to downscale the output image by (e.g. downscale=4 & input image size 4096x4096 will give an output image of size 1024x1024)

    # Returns:
    # - np.array: [n_patch**2, patch_size/downscale, patch_size/downscale, 3] array sequence of image patch_size-sized patches from the input region.
    # - patch_attention (torch.Tensor): [n_patch**2, nhead, patch_size/downscale, patch_size/downscale] tensor sequence of attention maps for patch_size-sized patches.
    # - region_attention (torch.Tensor): [nhead, region_size/downscale, region_size/downscale] tensor sequence of attention maps for input region.
    # """
    # t = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    # )
    # region_size = region.size[0]
    # n_patch = region_size // patch_size
    # n_minipatch = patch_size // mini_patch_size

    # with torch.no_grad():
    #     patches = (
    #         t(region)
    #         .unsqueeze(0)
    #         .unfold(2, patch_size, patch_size)
    #         .unfold(3, patch_size, patch_size)
    #     )  # (1, 3, region_size, region_size) -> (1, 3, n_patch, n_patch, patch_size, patch_size)
    #     patches = rearrange(
    #         patches, "b c p1 p2 w h -> (b p1 p2) c w h"
    #     )  # (n_patch**2, 3, patch_size, patch_size)
    #     patches = patches.to(patch_device, non_blocking=True)
    #     patch_features = patch_model(patches)  # (n_patch**2, 384)

    #     patch_attention = patch_model.get_last_selfattention(
    #         patches
    #     )  # (n_patch**2, nhead, n_minipatch**2+1, n_minipatch**2+1)
    #     nh = patch_attention.shape[1]  # number of head
    #     patch_attention = patch_attention[:, :, 0, 1:].reshape(
    #         n_patch**2, nh, -1
    #     )  # (n_patch**2, nhead, n_minipatch**2)
    #     patch_attention = patch_attention.reshape(
    #         n_patch**2, nh, n_minipatch, n_minipatch
    #     )  # (n_patch**2, nhead, n_minipatch, n_minipatch)
    #     patch_attention = (
    #         nn.functional.interpolate(
    #             patch_attention,
    #             scale_factor=int(mini_patch_size / downscale),
    #             mode="nearest",
    #         )
    #         .cpu()
    #         .numpy()
    #     )  # (n_patch**2, nhead, patch_size, patch_size) when downscale = 1
    #     # 'nearest' interpolation guarantees the values in the up-sampled array
    #     # lie in the same set as the values in the original array

    #     region_features = (
    #         patch_features.unfold(0, n_patch, n_patch).transpose(0, 1).unsqueeze(dim=0)
    #     )  # (1, 384, n_patch, n_patch)
    #     region_attention = region_model.get_last_selfattention(
    #         region_features.detach().to(region_device)
    #     )  # (1, nhead, n_patch**2+1, n_patch**2+1)
    #     nh = region_attention.shape[1]  # number of head
    #     region_attention = region_attention[0, :, 0, 1:].reshape(
    #         nh, -1
    #     )  # (nhead, 1, n_patch**2) -> (nhead, n_patch**2)
    #     region_attention = region_attention.reshape(
    #         nh, n_patch, n_patch
    #     )  # (nhead, n_patch, n_patch)
    #     region_attention = (
    #         nn.functional.interpolate(
    #             region_attention.unsqueeze(0),
    #             scale_factor=int(patch_size / downscale),
    #             mode="nearest",
    #         )[0]
    #         .cpu()
    #         .numpy()
    #     )  # (nhead, region_size, region_size) when downscale = 1

    #     if downscale != 1:
    #         patches = nn.functional.interpolate(
    #             patches, scale_factor=(1 / downscale), mode="nearest"
    #         )

    # return tensorbatch2im(patches), patch_attention, region_attention


def create_region_heatmaps_indiv(
    region: Image,
    patch_model: nn.Module,
    region_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    fname: str = "region",
    downscale: int = 1,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    threshold: Optional[float] = None,
    granular: bool = False,
    offset: int = 128,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
):
    """
    Creates following region heatmaps: patch-level ViT, region-level ViT & hierarchical heatmaps

    Args:
    - region (PIL.Image): input region
    - patch_model (nn.Module): patch-level Transformer
    - region_model (nn.Module): region-level Transformer
    - output_dir (Path): output directory for saving heatmaps
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - fname (str): file naming template
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - threshold (float): filter out patches with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - granular (bool): create additional offset patches to get more granular heatmaps
    - offset (int): if granular is True, uses this value to offset patches
    - patch_device (torch.device): device on which patch_model is
    - region_device (torch.device): device on which region_model is
    """
    region_size = region.size[0]
    n_patch = region_size // patch_size

    nhead_patch = patch_model.num_heads
    nhead_region = region_model.num_heads
    offset_ = offset

    _, patch_att, region_att = get_region_attention_scores(
        region,
        patch_model,
        region_model,
        transforms=transforms,
        patch_size=patch_size,
        mini_patch_size=mini_patch_size,
        downscale=downscale,
        patch_device=patch_device,
        region_device=region_device,
    )  # (n_patch**2, nhead, patch_size, patch_size) when downscale = 1

    if granular:
        offset = int(offset_ * region_size / 4096)
        region2 = add_margin(
            region.crop((offset, offset, region_size, region_size)),
            top=0,
            left=0,
            bottom=offset,
            right=offset,
            color=(255, 255, 255),
        )
        region3 = add_margin(
            region.crop((offset * 2, offset * 2, region_size, region_size)),
            top=0,
            left=0,
            bottom=offset * 2,
            right=offset * 2,
            color=(255, 255, 255),
        )
        region4 = add_margin(
            region.crop((offset * 3, offset * 3, region_size, region_size)),
            top=0,
            left=0,
            bottom=offset * 3,
            right=offset * 3,
            color=(255, 255, 255),
        )

        _, patch_att_2, region_att_2 = get_region_attention_scores(
            region2,
            patch_model,
            region_model,
            transforms=transforms,
            patch_size=patch_size,
            mini_patch_size=mini_patch_size,
            downscale=downscale,
            patch_device=patch_device,
            region_device=region_device,
        )
        _, _, region_att_3 = get_region_attention_scores(
            region3,
            patch_model,
            region_model,
            transforms=transforms,
            patch_size=patch_size,
            mini_patch_size=mini_patch_size,
            downscale=downscale,
            patch_device=patch_device,
            region_device=region_device,
            compute_patch_attention=False,
        )
        _, _, region_att_4 = get_region_attention_scores(
            region4,
            patch_model,
            region_model,
            transforms=transforms,
            patch_size=patch_size,
            mini_patch_size=mini_patch_size,
            downscale=downscale,
            patch_device=patch_device,
            region_device=region_device,
            compute_patch_attention=False,
        )

        offset_2 = offset // downscale
        offset_3 = (offset * 2) // downscale
        offset_4 = (offset * 3) // downscale

    s = region_size // downscale
    # given region is an RGB image, so the default filter for resizing is Resampling.BICUBIC
    # which is fine as we're resizing the image here, not attention scores
    save_region = np.array(region.resize((s, s)))

    patch_output_dir = Path(output_dir, f"{fname}_{patch_size}")
    patch_output_dir.mkdir(exist_ok=True, parents=True)

    with tqdm.tqdm(
        range(nhead_patch),
        desc="Patch-level Transformer heatmaps",
        unit="head",
        leave=True,
    ) as t:
        for i in t:
            patch_att_scores = concat_patch_scores(
                patch_att[:, i, :, :],
                region_size=region_size,
                patch_size=patch_size,
                size=(s // n_patch,) * 2,
            )

            if granular:
                patch_att_scores_2 = concat_patch_scores(
                    patch_att_2[:, i, :, :],
                    region_size=region_size,
                    patch_size=patch_size,
                    size=(s // n_patch,) * 2,
                )
                patch_att_scores *= 100
                patch_att_scores_2 *= 100
                new_patch_att_scores_2 = np.zeros_like(patch_att_scores_2)
                new_patch_att_scores_2[offset_2:s, offset_2:s] = patch_att_scores_2[
                    : (s - offset_2), : (s - offset_2)
                ]
                patch_overlay = np.ones_like(patch_att_scores_2) * 100
                patch_overlay[offset_2:s, offset_2:s] += 100
                patch_att_scores = (
                    patch_att_scores + new_patch_att_scores_2
                ) / patch_overlay

            if threshold != None:
                thresh_output_dir = Path(output_dir, f"{fname}_{patch_size}_thresh")
                thresh_output_dir.mkdir(exist_ok=True, parents=True)

                att_mask = patch_att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask >= threshold] = 0.95

                patch_color_block = (cmap(att_mask) * 255)[:, :, :3].astype(np.uint8)
                patch_hm = cv2.addWeighted(
                    patch_color_block,
                    alpha,
                    save_region.copy(),
                    1 - alpha,
                    0,
                    save_region.copy(),
                )
                patch_hm[att_mask == 0] = 0
                img_inverse = save_region.copy()
                img_inverse[att_mask == 0.95] = 0
                patch_hm = patch_hm + img_inverse
                img = Image.fromarray(patch_hm)
                img.save(Path(thresh_output_dir, f"head_{i}.png"))

            patch_color_block = (cmap(patch_att_scores) * 255)[:, :, :3].astype(
                np.uint8
            )
            patch_hm = cv2.addWeighted(
                patch_color_block,
                alpha,
                save_region.copy(),
                1 - alpha,
                0,
                save_region.copy(),
            )
            img = Image.fromarray(patch_hm)
            img.save(Path(patch_output_dir, f"head_{i}.png"))

    region_output_dir = Path(output_dir, f"{fname}_{region_size}")
    region_output_dir.mkdir(exist_ok=True, parents=True)

    with tqdm.tqdm(
        range(nhead_region),
        desc="Region-level Transformer heatmaps",
        unit="head",
        leave=True,
    ) as t:
        for j in t:
            region_att_scores = normalize_region_scores(region_att[j], size=(s,) * 2, min_val=region_att[j].min(), max_val=region_att[j].max())

            if granular:
                region_att_scores_2 = normalize_region_scores(
                    region_att_2[j], size=(s,) * 2, min_val=region_att_2[j].min(), max_val=region_att_2[j].max()
                )
                region_att_scores_3 = normalize_region_scores(
                    region_att_3[j], size=(s,) * 2, min_val=region_att_3[j].min(), max_val=region_att_3[j].max()
                )
                region_att_scores_4 = normalize_region_scores(
                    region_att_4[j], size=(s,) * 2, min_val=region_att_4[j].min(), max_val=region_att_4[j].max()
                )
                region_att_scores *= 100
                region_att_scores_2 *= 100
                region_att_scores_3 *= 100
                region_att_scores_4 *= 100
                new_region_att_scores_2 = np.zeros_like(region_att_scores_2)
                new_region_att_scores_2[offset_2:s, offset_2:s] = region_att_scores_2[
                    : (s - offset_2), : (s - offset_2)
                ]
                new_region_att_scores_3 = np.zeros_like(region_att_scores_3)
                new_region_att_scores_3[offset_3:s, offset_3:s] = region_att_scores_3[
                    : (s - offset_3), : (s - offset_3)
                ]
                new_region_att_scores_4 = np.zeros_like(region_att_scores_4)
                new_region_att_scores_4[offset_4:s, offset_4:s] = region_att_scores_4[
                    : (s - offset_4), : (s - offset_4)
                ]
                region_overlay = np.ones_like(new_region_att_scores_2) * 100
                region_overlay[offset_2:s, offset_2:s] += 100
                region_overlay[offset_3:s, offset_3:s] += 100
                region_overlay[offset_4:s, offset_4:s] += 100
                region_att_scores = (
                    region_att_scores
                    + new_region_att_scores_2
                    + new_region_att_scores_3
                    + new_region_att_scores_4
                ) / region_overlay

            if threshold != None:
                thresh_region_output_dir = Path(
                    output_dir, f"{fname}_{region_size}_thresh"
                )
                thresh_region_output_dir.mkdir(exist_ok=True, parents=True)

                att_mask = region_att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask >= threshold] = 0.95

                region_color_block = (cmap(att_mask) * 255)[:, :, :3].astype(np.uint8)
                region_hm = cv2.addWeighted(
                    region_color_block,
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
                img.save(Path(thresh_region_output_dir, f"head_{j}.png"))

            region_color_block = (cmap(region_att_scores) * 255)[:, :, :3].astype(
                np.uint8
            )
            region_hm = cv2.addWeighted(
                region_color_block,
                alpha,
                save_region.copy(),
                1 - alpha,
                0,
                save_region.copy(),
            )
            img = Image.fromarray(region_hm)
            img.save(Path(region_output_dir, f"head_{j}.png"))

    hierarchical_output_dir = Path(output_dir, f"{fname}_{region_size}_{patch_size}")
    hierarchical_output_dir.mkdir(exist_ok=True, parents=True)

    with tqdm.tqdm(
        range(nhead_region),
        desc="Hierarchical heatmaps",
        unit="head",
        leave=True,
    ) as t1:
        for j in t1:
            region_att_scores = normalize_region_scores(region_att[j], size=(s,) * 2, min_val=region_att[j].min(), max_val=region_att[j].max())

            if granular:
                region_att_scores_2 = normalize_region_scores(
                    region_att_2[j], size=(s,) * 2, min_val=region_att_2[j].min(), max_val=region_att_2[j].max()
                )
                region_att_scores_3 = normalize_region_scores(
                    region_att_3[j], size=(s,) * 2, min_val=region_att_3[j].min(), max_val=region_att_3[j].max()
                )
                region_att_scores_4 = normalize_region_scores(
                    region_att_4[j], size=(s,) * 2, min_val=region_att_4[j].min(), max_val=region_att_4[j].max()
                )
                region_att_scores *= 100
                region_att_scores_2 *= 100
                region_att_scores_3 *= 100
                region_att_scores_4 *= 100
                new_region_att_scores_2 = np.zeros_like(region_att_scores_2)
                new_region_att_scores_2[offset_2:s, offset_2:s] = region_att_scores_2[
                    : (s - offset_2), : (s - offset_2)
                ]
                new_region_att_scores_3 = np.zeros_like(region_att_scores_3)
                new_region_att_scores_3[offset_3:s, offset_3:s] = region_att_scores_3[
                    : (s - offset_3), : (s - offset_3)
                ]
                new_region_att_scores_4 = np.zeros_like(region_att_scores_4)
                new_region_att_scores_4[offset_4:s, offset_4:s] = region_att_scores_4[
                    : (s - offset_4), : (s - offset_4)
                ]
                region_overlay = np.ones_like(new_region_att_scores_2) * 100
                region_overlay[offset_2:s, offset_2:s] += 100
                region_overlay[offset_3:s, offset_3:s] += 100
                region_overlay[offset_4:s, offset_4:s] += 100
                region_att_scores = (
                    region_att_scores
                    + new_region_att_scores_2
                    + new_region_att_scores_3
                    + new_region_att_scores_4
                ) / region_overlay

            with tqdm.tqdm(
                range(nhead_patch),
                desc=f"Region head [{j+1}/{nhead_region}]",
                unit="head",
                leave=False,
            ) as t2:
                for i in t2:
                    patch_att_scores = concat_patch_scores(
                        patch_att[:, i, :, :],
                        region_size=region_size,
                        patch_size=patch_size,
                        size=(s // n_patch,) * 2,
                    )

                    if granular:
                        patch_att_scores_2 = concat_patch_scores(
                            patch_att_2[:, i, :, :],
                            region_size=region_size,
                            patch_size=patch_size,
                            size=(s // n_patch,) * 2,
                        )
                        patch_att_scores *= 100
                        patch_att_scores_2 *= 100
                        new_patch_att_scores_2 = np.zeros_like(patch_att_scores_2)
                        new_patch_att_scores_2[
                            offset_2:s, offset_2:s
                        ] = patch_att_scores_2[: (s - offset_2), : (s - offset_2)]
                        # TODO: why do they introduce a factor 2 here?
                        patch_overlay = np.ones_like(patch_att_scores_2) * 100 * 2
                        patch_overlay[offset_2:s, offset_2:s] += 100 * 2
                        patch_att_scores = (
                            (patch_att_scores + new_patch_att_scores_2)
                            * 2
                            / patch_overlay
                        )

                    if granular:
                        score = (
                            region_att_scores * region_overlay
                            + patch_att_scores * patch_overlay
                        ) / (region_overlay + patch_overlay)
                    else:
                        score = (region_att_scores + patch_att_scores) / 2

                    if threshold != None:
                        thresh_hierarchical_output_dir = Path(
                            output_dir, f"{fname}_{region_size}_{patch_size}_thresh"
                        )
                        thresh_hierarchical_output_dir.mkdir(
                            exist_ok=True, parents=True
                        )

                        att_mask = score.copy()
                        att_mask[att_mask < threshold] = 0
                        att_mask[att_mask >= threshold] = 0.95

                        color_block = (cmap(att_mask) * 255)[:, :, :3].astype(np.uint8)
                        hm = cv2.addWeighted(
                            color_block,
                            alpha,
                            save_region.copy(),
                            1 - alpha,
                            0,
                            save_region.copy(),
                        )
                        hm[att_mask == 0] = 0
                        img_inverse = save_region.copy()
                        img_inverse[att_mask == 0.95] = 0
                        hm = hm + img_inverse
                        img = Image.fromarray(hm)
                        img.save(
                            Path(
                                thresh_hierarchical_output_dir,
                                f"rhead_{j}_phead_{i}.png",
                            )
                        )

                    color_block = (cmap(score) * 255)[:, :, :3].astype(np.uint8)
                    hm = cv2.addWeighted(
                        color_block,
                        alpha,
                        save_region.copy(),
                        1 - alpha,
                        0,
                        save_region.copy(),
                    )
                    img = Image.fromarray(hm)
                    img.save(
                        Path(
                            hierarchical_output_dir,
                            f"rhead_{j}_phead_{i}.png",
                        )
                    )


def create_region_heatmaps_concat(
    region: Image,
    patch_model: nn.Module,
    region_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    granular: bool = False,
    offset: int = 128,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
):
    """
    Creates region heatmaps (concatenated for easy comparison)

    Args:
    - region (PIL.Image): input region
    - patch_model (nn.Module): patch-level Transformer
    - region_model (nn.Module): region-level Transformer
    - output_dir (Path): output directory for saving heatmaps
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - granular (bool): create additional offset patches to get more granular heatmaps
    - offset (int): if granular is True, uses this value to offset patches
    - patch_device (torch.device): device on which patch_model is
    - region_device (torch.device): device on which region_model is
    """
    region_size = region.size[0]
    n_patch = region_size // patch_size

    nhead_patch = patch_model.num_heads
    nhead_region = region_model.num_heads
    offset_ = offset

    _, patch_att, region_att = get_region_attention_scores(
        region,
        patch_model,
        region_model,
        transforms=transforms,
        patch_size=patch_size,
        mini_patch_size=mini_patch_size,
        downscale=downscale,
        patch_device=patch_device,
        region_device=region_device,
    )  # (256, 6, 128, 128), (6, 2048, 2048) when downscale = 2

    if granular:
        offset = int(offset_ * region_size / 4096)
        region2 = add_margin(
            region.crop((offset, offset, region_size, region_size)),
            top=0,
            left=0,
            bottom=offset,
            right=offset,
            color=(255, 255, 255),
        )
        region3 = add_margin(
            region.crop((offset * 2, offset * 2, region_size, region_size)),
            top=0,
            left=0,
            bottom=offset * 2,
            right=offset * 2,
            color=(255, 255, 255),
        )
        region4 = add_margin(
            region.crop((offset * 3, offset * 3, region_size, region_size)),
            top=0,
            left=0,
            bottom=offset * 3,
            right=offset * 3,
            color=(255, 255, 255),
        )

        _, patch_att_2, region_att_2 = get_region_attention_scores(
            region2,
            patch_model,
            region_model,
            transforms=transforms,
            patch_size=patch_size,
            mini_patch_size=mini_patch_size,
            downscale=downscale,
            patch_device=patch_device,
            region_device=region_device,
        )
        _, _, region_att_3 = get_region_attention_scores(
            region3,
            patch_model,
            region_model,
            transforms=transforms,
            patch_size=patch_size,
            mini_patch_size=mini_patch_size,
            downscale=downscale,
            patch_device=patch_device,
            region_device=region_device,
            compute_patch_attention=False,
        )
        _, _, region_att_4 = get_region_attention_scores(
            region4,
            patch_model,
            region_model,
            transforms=transforms,
            patch_size=patch_size,
            mini_patch_size=mini_patch_size,
            downscale=downscale,
            patch_device=patch_device,
            region_device=region_device,
            compute_patch_attention=False,
        )

        offset_2 = offset // downscale
        offset_3 = (offset * 2) // downscale
        offset_4 = (offset * 3) // downscale

    s = region_size // downscale  # 2048 for downscale = 2, region_size = 4096
    # given region is an RGB image, so the default filter for resizing is Resampling.BICUBIC
    # which is fine as we're resizing the image here, not attention scores
    save_region = np.array(
        region.resize((s, s))
    )  # (2048, 2048) for downscale = 2, region_size = 4096

    with tqdm.tqdm(
        range(nhead_region),
        desc="Hierarchical heatmaps",
        unit="head",
        leave=True,
    ) as t1:
        for j in t1:
            region_att_scores_1 = normalize_region_scores(
                region_att[j], size=(s,) * 2, min_val=region_att[j].min(), max_val=region_att[j].max()
            )  # (2048, 2048) for downscale = 2

            if granular:
                region_att_scores_2 = normalize_region_scores(
                    region_att_2[j], size=(s,) * 2, min_val=region_att_2[j].min(), max_val=region_att_2[j].max()
                )
                region_att_scores_3 = normalize_region_scores(
                    region_att_3[j], size=(s,) * 2, min_val=region_att_3[j].min(), max_val=region_att_3[j].max()
                )
                region_att_scores_4 = normalize_region_scores(
                    region_att_4[j], size=(s,) * 2, min_val=region_att_4[j].min(), max_val=region_att_4[j].max()
                )
                region_att_scores_1 *= 100
                region_att_scores_2 *= 100
                region_att_scores_3 *= 100
                region_att_scores_4 *= 100
                new_region_att_scores_2 = np.zeros_like(region_att_scores_2)
                new_region_att_scores_2[offset_2:s, offset_2:s] = region_att_scores_2[
                    : (s - offset_2), : (s - offset_2)
                ]
                new_region_att_scores_3 = np.zeros_like(region_att_scores_3)
                new_region_att_scores_3[offset_3:s, offset_3:s] = region_att_scores_3[
                    : (s - offset_3), : (s - offset_3)
                ]
                new_region_att_scores_4 = np.zeros_like(region_att_scores_4)
                new_region_att_scores_4[offset_4:s, offset_4:s] = region_att_scores_4[
                    : (s - offset_4), : (s - offset_4)
                ]
                region_overlay = np.ones_like(new_region_att_scores_2) * 100
                region_overlay[offset_2:s, offset_2:s] += 100
                region_overlay[offset_3:s, offset_3:s] += 100
                region_overlay[offset_4:s, offset_4:s] += 100
                region_att_scores = (
                    region_att_scores_1
                    + new_region_att_scores_2
                    + new_region_att_scores_3
                    + new_region_att_scores_4
                ) / region_overlay

            # TODO: why do they run cmap on region_att_scores_1/100 and not on region_att_scores?
            region_color_block = (cmap(region_att_scores_1 / 100) * 255)[
                :, :, :3
            ].astype(np.uint8)
            region_hm = cv2.addWeighted(
                region_color_block,
                alpha,
                save_region.copy(),
                1 - alpha,
                0,
                save_region.copy(),
            )  # (2048, 2048) for downscale = 2, region_size = 4096

            with tqdm.tqdm(
                range(nhead_patch),
                desc=f"Region head [{j+1}/{nhead_region}]",
                unit="head",
                leave=False,
            ) as t2:
                for i in t2:
                    patch_att_scores = concat_patch_scores(
                        patch_att[:, i, :, :],
                        region_size=region_size,
                        patch_size=patch_size,
                        size=(s // n_patch,) * 2,
                    )  # (2048, 2048) for downscale = 2

                    if granular:
                        patch_att_scores_2 = concat_patch_scores(
                            patch_att_2[:, i, :, :],
                            region_size=region_size,
                            patch_size=patch_size,
                            size=(s // n_patch,) * 2,
                        )
                        patch_att_scores *= 100
                        patch_att_scores_2 *= 100
                        new_patch_att_scores_2 = np.zeros_like(patch_att_scores_2)
                        new_patch_att_scores_2[
                            offset_2:s, offset_2:s
                        ] = patch_att_scores_2[: (s - offset_2), : (s - offset_2)]
                        # TODO: why do they introduce a factor 2 here?
                        patch_overlay = np.ones_like(patch_att_scores_2) * 100 * 2
                        patch_overlay[offset_2:s, offset_2:s] += 100 * 2
                        patch_att_scores = (
                            (patch_att_scores + new_patch_att_scores_2)
                            * 2
                            / patch_overlay
                        )

                    patch_color_block = (cmap(patch_att_scores) * 255)[:, :, :3].astype(
                        np.uint8
                    )
                    patch_hm = cv2.addWeighted(
                        patch_color_block,
                        alpha,
                        save_region.copy(),
                        1 - alpha,
                        0,
                        save_region.copy(),
                    )  # (2048, 2048) for downscale = 2

                    if granular:
                        score = (
                            region_att_scores * region_overlay
                            + patch_att_scores * patch_overlay
                        ) / (region_overlay + patch_overlay)
                    else:
                        score = (region_att_scores_1 + patch_att_scores) / 2

                    color_block = (cmap(score) * 255)[:, :, :3].astype(np.uint8)
                    hierarchical_region_hm = cv2.addWeighted(
                        color_block,
                        alpha,
                        save_region.copy(),
                        1 - alpha,
                        0,
                        save_region.copy(),
                    )  # (2048, 2048) for downscale = 2

                    pad = 100
                    canvas = Image.new(
                        "RGB", (s * 2 + pad * 3,) * 2, (255,) * 3
                    )  # (2 * region_size // downscale + 100, 2 * region_size // downscale + 100, 3) ; (4096, 4096, 3) for downscale = 2, region_size = 4096
                    draw = ImageDraw.Draw(canvas)
                    draw.text(
                        (s * 0.5 + pad, pad // 4),
                        f"patch-level Transformer (Head: {i})",
                        (0, 0, 0),
                    )
                    canvas = canvas.rotate(90)
                    draw = ImageDraw.Draw(canvas)
                    draw.text(
                        (s * 1.5 + pad * 2, pad // 4),
                        f"region-level Transformer (Head: {j})",
                        (0, 0, 0),
                    )
                    canvas.paste(
                        Image.fromarray(save_region), (pad, pad)
                    )  # (2048, 2048) for downscale = 2, region_size = 4096 ; (100, 100)
                    canvas.paste(
                        Image.fromarray(region_hm), (s + 2 * pad, pad)
                    )  # (2048, 2048) for downscale = 2, region_size = 4096 ; (2048+100, 100)
                    canvas.paste(
                        Image.fromarray(patch_hm), (pad, s + 2 * pad)
                    )  # (2048, 2048) for downscale = 2, region_size = 4096 ; (100, 2048+100)
                    canvas.paste(
                        Image.fromarray(hierarchical_region_hm),
                        (s + 2 * pad, s + 2 * pad),
                    )  # (2048, 2048) for downscale = 2, region_size = 4096 ; (2048+100, 2048+100)
                    canvas.save(Path(output_dir, f"rhead_{j}_phead_{i}.png"))


def get_slide_heatmaps_patch_level(
    slide_path: Path,
    coordinates_dir: Path,
    patch_model: nn.Module,
    region_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    threshold: Optional[float] = None,
    highlight: Optional[float] = None,
    opacity: float = 0.3,
    granular: bool = False,
    offset: int = 128,
    segmentation_mask_path: Optional[str] = None,
    spacing: Optional[float] = None,
    downsample: Optional[int] = None,
    background_pixel_value: Optional[int] = None,
    tissue_pixel_value: Optional[int] = None,
    patch_attn_mask: Optional[torch.Tensor] = None,
    mini_patch_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    main_process: bool = True,
):
    """
    Returns heatmaps of patch-level Transformer attention for each region extracted in the slide.
    These can later be stitched together for slide-level heatmap visualization.

    Args:
    - slide_path (Path): path to slide
    - coordinates_dir (Path): path to root folder containing region coordinates .npy files
    - patch_model (nn.Module): patch-level Transformer
    - region_model (nn.Module): region-level Transformer
    - output_dir (Path): output directory for saving heatmaps
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - threshold (float): filter out regions with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - highlight (float): filter out regions with attention scores lower than this value (set to None to disbale heatmap highlighting)
    - opacity (float): if highlight, set opacity for non-highlighted regions on stitched heatmap
    - granular (bool): create additional offset regions to get more granular heatmaps
    - offset (int): if granular is True, uses this value to offset regions
    - patch_device (torch.device): device on which patch_model is
    - region_device (torch.device): device on which region_model is
    - main_process (bool): controls tqdm display when running with multiple gpus
    """
    slide_id = slide_path.stem.replace(" ", "_")
    wsi = wsd.WholeSlideImage(slide_path, backend="asap")

    patch_output_dir = Path(output_dir, "patch", f"{patch_size}")
    patch_output_dir.mkdir(exist_ok=True, parents=True)

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
    region_size = int(region_size_resized / resize_factor)

    nhead_patch = patch_model.num_heads
    offset_ = offset

    mask_attn_patch = (mini_patch_attn_mask is not None)
    mask_attn_region = (patch_attn_mask is not None)
    mask_attention = mask_attn_patch or mask_attn_region

    with tqdm.tqdm(
        coordinates,
        desc="Gathering attention scores",
        unit="region",
        leave=True,
        disable=not main_process,
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

            pm, mpm = None, None
            if mask_attn_patch:
                mpm = mini_patch_attn_mask[k]
            if mask_attn_region:
                pm = patch_attn_mask[k].unsqueeze(0)

            _, patch_att, _ = get_region_attention_scores(
                region,
                patch_model,
                region_model,
                transforms=transforms,
                patch_size=patch_size,
                mini_patch_size=mini_patch_size,
                downscale=downscale,
                patch_attn_mask=pm,
                mini_patch_attn_mask=mpm,
                patch_device=patch_device,
                region_device=region_device,
            )

            if granular:
                offset = int(offset_ * region_size / 4096)
                region2 = add_margin(
                    region.crop((offset, offset, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset,
                    right=offset,
                    color=(255, 255, 255),
                )
                pm2, mpm2 = None, None
                if mask_attention:
                    pm2, mpm2 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset,
                    )
                    if not mask_attn_patch:
                        mpm2 = None
                    if not mask_attn_region:
                        pm2 = None
                _, patch_att_2, _ = get_region_attention_scores(
                    region2,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm2,
                    mini_patch_attn_mask=mpm2,
                    patch_device=patch_device,
                    region_device=region_device,
                )
                offset_2 = offset // downscale

            s = region_size // downscale
            # given region is an RGB image, so the default filter for resizing is Resampling.BICUBIC
            # which is fine as we're resizing the image here, not attention scores
            save_region = np.array(region.resize((s, s)))

            with tqdm.tqdm(
                range(nhead_patch),
                desc=f"Processing region [{k+1}/{nregions}]",
                unit="head",
                leave=False,
                disable=not main_process,
            ) as t2:
                for i in t2:

                    patch_att_scores = concat_patch_scores(
                        patch_att[:, i, :, :],
                        region_size=region_size,
                        patch_size=patch_size,
                        size=(s // n_patch,) * 2,
                    )

                    if granular:
                        patch_att_scores_2 = concat_patch_scores(
                            patch_att_2[:, i, :, :],
                            region_size=region_size,
                            patch_size=patch_size,
                            size=(s // n_patch,) * 2,
                        )
                        patch_att_scores *= 100
                        patch_att_scores_2 *= 100
                        new_patch_att_scores_2 = np.zeros_like(patch_att_scores_2)
                        new_patch_att_scores_2[
                            offset_2:s, offset_2:s
                        ] = patch_att_scores_2[: (s - offset_2), : (s - offset_2)]
                        patch_overlay = np.ones_like(patch_att_scores_2) * 100
                        patch_overlay[offset_2:s, offset_2:s] += 100
                        patch_att_scores = (
                            patch_att_scores + new_patch_att_scores_2
                        ) / patch_overlay

                    if threshold != None:
                        thresh_patch_hm_output_dir = Path(
                            patch_output_dir, "thresholded", f"head_{i}"
                        )
                        thresh_patch_hm_output_dir.mkdir(exist_ok=True, parents=True)

                        att_mask = patch_att_scores.copy()
                        att_mask[att_mask < threshold] = 0
                        att_mask[att_mask >= threshold] = 0.95

                        patch_color_block = (cmap(att_mask) * 255)[:, :, :3].astype(
                            np.uint8
                        )
                        patch_hm = cv2.addWeighted(
                            patch_color_block,
                            alpha,
                            save_region.copy(),
                            1 - alpha,
                            0,
                            save_region.copy(),
                        )
                        patch_hm[att_mask == 0] = 0
                        img_inverse = save_region.copy()
                        img_inverse[att_mask == 0.95] = 0
                        patch_hm = patch_hm + img_inverse
                        img = Image.fromarray(patch_hm)
                        img.save(Path(thresh_patch_hm_output_dir, f"{x}_{y}.png"))

                    if highlight != None:
                        highlight_patch_hm_output_dir = Path(
                            patch_output_dir, "highlighted", f"head_{i}"
                        )
                        highlight_patch_hm_output_dir.mkdir(exist_ok=True, parents=True)
                        save_region = np.array(region.resize((s, s)))
                        rgba_region = np.dstack((save_region, np.zeros((s,s), dtype=np.uint8)+255))
                        att_mask = patch_att_scores.copy()
                        att_mask[att_mask < highlight] = 0
                        att_mask[att_mask >= highlight] = 1
                        highlighted_hm = rgba_region * (att_mask >= highlight)[..., np.newaxis]
                        m_black = (highlighted_hm[:, :, 0:3] == [0,0,0]).all(2)
                        transparent_region = np.dstack((save_region, np.zeros((s,s), dtype=np.uint8)+int(255*opacity)))
                        highlighted_hm[m_black] = transparent_region[m_black]
                        img = Image.fromarray(highlighted_hm, mode='RGBA')
                        img.save(Path(highlight_patch_hm_output_dir, f"{x}_{y}.png"))

                    patch_color_block = (cmap(patch_att_scores) * 255)[:, :, :3].astype(
                        np.uint8
                    )
                    patch_hm = cv2.addWeighted(
                        patch_color_block,
                        alpha,
                        save_region.copy(),
                        1 - alpha,
                        0,
                        save_region.copy(),
                    )
                    patch_hm_output_dir = Path(patch_output_dir, "regular", f"head_{i}")
                    patch_hm_output_dir.mkdir(exist_ok=True, parents=True)
                    img = Image.fromarray(patch_hm)
                    img.save(Path(patch_hm_output_dir, f"{x}_{y}.png"))

    return patch_output_dir


def get_slide_heatmaps_region_level(
    slide_path: Path,
    coordinates_dir: Path,
    patch_model: nn.Module,
    region_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    threshold: Optional[float] = None,
    highlight: Optional[float] = None,
    opacity: float = 0.3,
    granular: bool = False,
    offset: int = 128,
    gaussian_smoothing: bool = False,
    segmentation_mask_path: Optional[str] = None,
    spacing: Optional[float] = None,
    downsample: Optional[int] = None,
    background_pixel_value: Optional[int] = None,
    tissue_pixel_value: Optional[int] = None,
    restrict_to_tissue: bool = False,
    patch_attn_mask: Optional[torch.Tensor] = None,
    mini_patch_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    main_process: bool = True,
):
    """
    Returns heatmaps of region-level Transformer attention for each region extracted in the slide.
    These can later be stitched together for slide-level heatmap visualization.

    Args:
    - slide_path (Path): path to the slide
    - coordinates_dir (Path): path to root folder containing region coordinates in .npy files
    - patch_model (nn.Module): patch-level Transformer
    - region_model (nn.Module): region-level Transformer
    - output_dir (Path): output directory for saving heatmaps
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - threshold (float): filter out regions with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - highlight (float): filter out regions with attention scores lower than this value (set to None to disbale heatmap highlighting)
    - opacity (float): if highlight, set opacity for non-highlighted regions on stitched heatmap
    - region_fmt (str): file format used for extracted regions
    - granular (bool): create additional offset regions to get more granular heatmaps
    - offset (int): if granular is True, uses this value to offset regions
    - restrict_to_tissue (bool): whether to restrict heatmaps to tissue content
    - patch_device (torch.device): device on which patch_model is
    - region_device (torch.device): device on which region_model is
    - main_process (bool): controls tqdm display when running with multiple gpus
    """
    slide_id = slide_path.stem.replace(" ", "_")
    wsi_object = WholeSlideImage(slide_path, segmentation_mask_path, downsample=downsample)

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

    region_spacing = wsi_object.spacings[region_level]
    region_size = int(region_size_resized / resize_factor)

    region_output_dir = Path(output_dir, "region", f"{region_size}")
    region_output_dir.mkdir(exist_ok=True, parents=True)

    nhead_region = region_model.num_heads
    offset_ = offset

    mask_attn_patch = (mini_patch_attn_mask is not None)
    mask_attn_region = (patch_attn_mask is not None)
    mask_attention = mask_attn_patch or mask_attn_region

    # do a first pass on the slide to get the attention scores
    # and find the min/max values for normalization

    head_min = [float('inf')] * nhead_region
    head_max = [float('-inf')] * nhead_region

    with tqdm.tqdm(
        coordinates,
        desc="First pass over regions to grab min/max attention scores for normalization",
        unit="region",
        leave=True,
        disable=not main_process,
    ) as t1:
        for k, (x,y) in enumerate(t1):
            region_arr = wsi_object.wsi.get_patch(
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

            pm, mpm = None, None
            if mask_attn_patch:
                mpm = mini_patch_attn_mask[k]
            if mask_attn_region:
                pm = patch_attn_mask[k].unsqueeze(0)

            _, _, region_att = get_region_attention_scores(
                region,
                patch_model,
                region_model,
                transforms=transforms,
                patch_size=patch_size,
                mini_patch_size=mini_patch_size,
                downscale=downscale,
                patch_attn_mask=pm,
                mini_patch_attn_mask=mpm,
                patch_device=patch_device,
                region_device=region_device,
                compute_patch_attention=False,
            )

            for i in range(nhead_region):
                head_min[i] = min(head_min[i], region_att[i].min())
                head_max[i] = max(head_max[i], region_att[i].max())

            if granular:
                offset = int(offset_ * region_size / 4096)
                region2 = add_margin(
                    region.crop((offset, offset, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset,
                    right=offset,
                    color=(255, 255, 255),
                )
                pm2, mpm2 = None, None
                if mask_attention:
                    pm2, mpm2 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset,
                    )
                    if not mask_attn_patch:
                        mpm2 = None
                    if not mask_attn_region:
                        pm2 = None
                region3 = add_margin(
                    region.crop((offset * 2, offset * 2, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset * 2,
                    right=offset * 2,
                    color=(255, 255, 255),
                )
                pm3, mpm3 = None, None
                if mask_attention:
                    pm3, mpm3 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset*2,
                    )
                    if not mask_attn_patch:
                        mpm3 = None
                    if not mask_attn_region:
                        pm3 = None
                region4 = add_margin(
                    region.crop((offset * 3, offset * 3, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset * 3,
                    right=offset * 3,
                    color=(255, 255, 255),
                )
                pm4, mpm4 = None, None
                if mask_attention:
                    pm4, mpm4 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset*3,
                    )
                    if not mask_attn_patch:
                        mpm4 = None
                    if not mask_attn_region:
                        pm4 = None

                _, _, region_att_2 = get_region_attention_scores(
                    region2,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm2,
                    mini_patch_attn_mask=mpm2,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=False,
                )
                _, _, region_att_3 = get_region_attention_scores(
                    region3,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm3,
                    mini_patch_attn_mask=mpm3,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=False,
                )
                _, _, region_att_4 = get_region_attention_scores(
                    region4,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm4,
                    mini_patch_attn_mask=mpm4,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=False,
                )

                for i in range(nhead_region):
                    head_min[i] = min(
                        head_min[i],
                        region_att_2[i].min(),
                        region_att_3[i].min(),
                        region_att_4[i].min(),
                    )
                    head_max[i] = max(
                        head_max[i],
                        region_att_2[i].max(),
                        region_att_3[i].max(),
                        region_att_4[i].max(),
                    )

    ## second pass to get normalized attention scores

    with tqdm.tqdm(
        coordinates,
        desc="Second pass to get normalized attention scores",
        unit="region",
        leave=True,
        disable=not main_process,
    ) as t1:
        for k, (x,y) in enumerate(t1):
            region_arr = wsi_object.wsi.get_patch(
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

            pm, mpm = None, None
            if mask_attn_patch:
                mpm = mini_patch_attn_mask[k]
            if mask_attn_region:
                pm = patch_attn_mask[k].unsqueeze(0)

            _, _, region_att = get_region_attention_scores(
                region,
                patch_model,
                region_model,
                transforms=transforms,
                patch_size=patch_size,
                mini_patch_size=mini_patch_size,
                downscale=downscale,
                patch_attn_mask=pm,
                mini_patch_attn_mask=mpm,
                patch_device=patch_device,
                region_device=region_device,
                compute_patch_attention=False,
            )

            if granular:
                offset = int(offset_ * region_size / 4096)
                region2 = add_margin(
                    region.crop((offset, offset, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset,
                    right=offset,
                    color=(255, 255, 255),
                )
                pm2, mpm2 = None, None
                if mask_attention:
                    pm2, mpm2 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset,
                    )
                    if not mask_attn_patch:
                        mpm2 = None
                    if not mask_attn_region:
                        pm2 = None
                region3 = add_margin(
                    region.crop((offset * 2, offset * 2, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset * 2,
                    right=offset * 2,
                    color=(255, 255, 255),
                )
                pm3, mpm3 = None, None
                if mask_attention:
                    pm3, mpm3 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset*2,
                    )
                    if not mask_attn_patch:
                        mpm3 = None
                    if not mask_attn_region:
                        pm3 = None
                region4 = add_margin(
                    region.crop((offset * 3, offset * 3, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset * 3,
                    right=offset * 3,
                    color=(255, 255, 255),
                )
                pm4, mpm4 = None, None
                if mask_attention:
                    pm4, mpm4 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset*3,
                    )
                    if not mask_attn_patch:
                        mpm4 = None
                    if not mask_attn_region:
                        pm4 = None

                _, _, region_att_2 = get_region_attention_scores(
                    region2,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm2,
                    mini_patch_attn_mask=mpm2,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=False,
                )
                _, _, region_att_3 = get_region_attention_scores(
                    region3,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm3,
                    mini_patch_attn_mask=mpm3,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=False,
                )
                _, _, region_att_4 = get_region_attention_scores(
                    region4,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm4,
                    mini_patch_attn_mask=mpm4,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=False,
                )

                offset_2 = offset // downscale
                offset_3 = (offset * 2) // downscale
                offset_4 = (offset * 3) // downscale

            s = region_size // downscale
            # given region is an RGB image, the default filter for resizing is Resampling.BICUBIC
            # which is fine as we're resizing the image here, not attention scores
            save_region = np.array(region.resize((s, s)))

            with tqdm.tqdm(
                range(nhead_region),
                desc=f"Processing region [{k+1}/{nregions}]",
                unit="head",
                leave=False,
                disable=not main_process,
            ) as t2:
                for j in t2:

                    region_att_scores = normalize_region_scores(
                        region_att[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                    )

                    if granular:
                        region_att_scores_2 = normalize_region_scores(
                            region_att_2[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                        )
                        region_att_scores_3 = normalize_region_scores(
                            region_att_3[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                        )
                        region_att_scores_4 = normalize_region_scores(
                            region_att_4[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                        )
                        region_att_scores *= 100
                        region_att_scores_2 *= 100
                        region_att_scores_3 *= 100
                        region_att_scores_4 *= 100
                        new_region_att_scores_2 = np.zeros_like(region_att_scores_2)
                        new_region_att_scores_2[
                            offset_2:s, offset_2:s
                        ] = region_att_scores_2[: (s - offset_2), : (s - offset_2)]
                        new_region_att_scores_3 = np.zeros_like(region_att_scores_3)
                        new_region_att_scores_3[
                            offset_3:s, offset_3:s
                        ] = region_att_scores_3[: (s - offset_3), : (s - offset_3)]
                        new_region_att_scores_4 = np.zeros_like(region_att_scores_4)
                        new_region_att_scores_4[
                            offset_4:s, offset_4:s
                        ] = region_att_scores_4[: (s - offset_4), : (s - offset_4)]
                        region_overlay = np.ones_like(new_region_att_scores_2) * 100
                        region_overlay[offset_2:s, offset_2:s] += 100
                        region_overlay[offset_3:s, offset_3:s] += 100
                        region_overlay[offset_4:s, offset_4:s] += 100
                        region_att_scores = (
                            region_att_scores
                            + new_region_att_scores_2
                            + new_region_att_scores_3
                            + new_region_att_scores_4
                        ) / region_overlay

                    if gaussian_smoothing:

                        overlap_regions, overlap_coords = create_overlap_regions((x,y), coordinates, Path(region_dir, slide_id, "imgs"), region_size)
                        n_overlap_regions = len(overlap_regions)

                        region_overlay = np.ones_like(region_att_scores) * 100

                        with tqdm.tqdm(
                            overlap_regions,
                            desc=f"Smoothing region [{k+1}/{nregions}]",
                            unit="region",
                            leave=False,
                            disable=(not main_process) or (len(overlap_regions) == 0),
                        ) as t2:
                            for m, region_ov in enumerate(t2):

                                if m == 0:
                                    region_att_scores *= 100

                                x_ov, y_ov = overlap_coords[m]

                                _, _, region_ov_att = get_region_attention_scores(
                                    region_ov,
                                    patch_model,
                                    region_model,
                                    transforms=transforms,
                                    patch_size=patch_size,
                                    mini_patch_size=mini_patch_size,
                                    downscale=downscale,
                                    patch_device=patch_device,
                                    region_device=region_device,
                                    compute_patch_attention=False,
                                )

                                if granular:
                                    offset = int(offset_ * region_size / 4096)
                                    region_ov2 = add_margin(
                                        region_ov.crop((offset, offset, region_size, region_size)),
                                        top=0,
                                        left=0,
                                        bottom=offset,
                                        right=offset,
                                        color=(255, 255, 255),
                                    )
                                    region_ov3 = add_margin(
                                        region_ov.crop((offset * 2, offset * 2, region_size, region_size)),
                                        top=0,
                                        left=0,
                                        bottom=offset * 2,
                                        right=offset * 2,
                                        color=(255, 255, 255),
                                    )
                                    region_ov4 = add_margin(
                                        region_ov.crop((offset * 3, offset * 3, region_size, region_size)),
                                        top=0,
                                        left=0,
                                        bottom=offset * 3,
                                        right=offset * 3,
                                        color=(255, 255, 255),
                                    )

                                    _, _, region_ov_att_2 = get_region_attention_scores(
                                        region_ov2,
                                        patch_model,
                                        region_model,
                                        transforms=transforms,
                                        patch_size=patch_size,
                                        mini_patch_size=mini_patch_size,
                                        downscale=downscale,
                                        patch_device=patch_device,
                                        region_device=region_device,
                                        compute_patch_attention=False,
                                    )
                                    _, _, region_ov_att_3 = get_region_attention_scores(
                                        region_ov3,
                                        patch_model,
                                        region_model,
                                        transforms=transforms,
                                        patch_size=patch_size,
                                        mini_patch_size=mini_patch_size,
                                        downscale=downscale,
                                        patch_device=patch_device,
                                        region_device=region_device,
                                        compute_patch_attention=False,
                                    )
                                    _, _, region_ov_att_4 = get_region_attention_scores(
                                        region_ov4,
                                        patch_model,
                                        region_model,
                                        transforms=transforms,
                                        patch_size=patch_size,
                                        mini_patch_size=mini_patch_size,
                                        downscale=downscale,
                                        patch_device=patch_device,
                                        region_device=region_device,
                                        compute_patch_attention=False,
                                    )

                                    offset_2 = offset // downscale
                                    offset_3 = (offset * 2) // downscale
                                    offset_4 = (offset * 3) // downscale

                                region_ov_att_scores = normalize_region_scores(
                                    region_ov_att[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                                )

                                if granular:
                                    region_ov_att_scores_2 = normalize_region_scores(
                                        region_ov_att_2[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                                    )
                                    region_ov_att_scores_3 = normalize_region_scores(
                                        region_ov_att_3[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                                    )
                                    region_ov_att_scores_4 = normalize_region_scores(
                                        region_ov_att_4[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                                    )
                                    region_ov_att_scores *= 100
                                    region_ov_att_scores_2 *= 100
                                    region_ov_att_scores_3 *= 100
                                    region_ov_att_scores_4 *= 100
                                    new_region_ov_att_scores_2 = np.zeros_like(region_ov_att_scores_2)
                                    new_region_ov_att_scores_2[
                                        offset_2:s, offset_2:s
                                    ] = region_ov_att_scores_2[: (s - offset_2), : (s - offset_2)]
                                    new_region_ov_att_scores_3 = np.zeros_like(region_ov_att_scores_3)
                                    new_region_ov_att_scores_3[
                                        offset_3:s, offset_3:s
                                    ] = region_ov_att_scores_3[: (s - offset_3), : (s - offset_3)]
                                    new_region_ov_att_scores_4 = np.zeros_like(region_ov_att_scores_4)
                                    new_region_ov_att_scores_4[
                                        offset_4:s, offset_4:s
                                    ] = region_ov_att_scores_4[: (s - offset_4), : (s - offset_4)]
                                    region_ov_overlay = np.ones_like(new_region_ov_att_scores_2) * 100
                                    region_ov_overlay[offset_2:s, offset_2:s] += 100
                                    region_ov_overlay[offset_3:s, offset_3:s] += 100
                                    region_ov_overlay[offset_4:s, offset_4:s] += 100
                                    region_ov_att_scores = (
                                        region_ov_att_scores
                                        + new_region_ov_att_scores_2
                                        + new_region_ov_att_scores_3
                                        + new_region_ov_att_scores_4
                                    ) / region_ov_overlay

                                new_region_ov_att_scores = np.zeros_like(region_ov_att_scores)
                                region_ov_att_scores *= 100
                                new_region_ov_att_scores[
                                    (x_ov-x):s, (y_ov-y):s
                                ] = region_ov_att_scores[:(s-(x_ov-x)), :(s-(y_ov-y))]
                                region_overlay[x_ov:x+s, y_ov:y+s] += 100
                                region_att_scores = region_att_scores + new_region_ov_att_scores
                                if (n_overlap_regions == 1) or (n_overlap_regions == 2 and m == 1):
                                    region_att_scores = region_att_scores / region_overlay

                    region_color_block = (cmap(region_att_scores) * 255)[
                        :, :, :3
                    ].astype(np.uint8)
                    region_hm = cv2.addWeighted(
                        region_color_block,
                        alpha,
                        save_region.copy(),
                        1 - alpha,
                        0,
                        save_region.copy(),
                    )

                    region_hm_output_dir = Path(
                        region_output_dir, "regular", f"head_{j}"
                    )
                    region_hm_output_dir.mkdir(exist_ok=True, parents=True)
                    if restrict_to_tissue:
                        seg_level, seg_spacing = wsi_object.seg_level, wsi_object.seg_spacing
                        height, width = wsi_object.wsi.get_shape_from_spacing(seg_spacing)
                        # need to scale coordinates from level 0 to seg_level
                        spacing_level = wsi_object.get_best_level_for_spacing(spacing)
                        downsample_factor = tuple(dv/ds for dv,ds in zip(wsi_object.level_downsamples[seg_level], wsi_object.level_downsamples[0]))
                        x_downsampled = int(round(x * 1 / downsample_factor[0], 0))
                        y_downsampled = int(round(y * 1 / downsample_factor[1], 0))
                        # need to scale region from spacing level to seg_level
                        downsample_factor = tuple(dv/ds for dv,ds in zip(wsi_object.level_downsamples[seg_level], wsi_object.level_downsamples[spacing_level]))
                        w_downsampled = int(round(s * downscale * 1 / downsample_factor[0], 0))
                        h_downsampled = int(round(s * downscale * 1 / downsample_factor[1], 0))
                        tissue_mask = wsi_object.binary_mask[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                        ]
                        # take care of non tissue content
                        tissue_mask_neg = (tissue_mask == 0).astype(int)
                        tissue_mask_neg_scaled = (tissue_mask_neg * 255).astype(np.uint8)
                        tissue_mask_neg_pil = Image.fromarray(tissue_mask_neg_scaled)
                        tissue_mask_neg_pil_resized = tissue_mask_neg_pil.resize((s, s), resample=Image.NEAREST)
                        tissue_mask_neg_resized = np.asarray(tissue_mask_neg_pil_resized)
                        tissue_mask_neg_resized = (tissue_mask_neg_resized > 0).astype('uint8')
                        tissue_mask_neg_resized = tissue_mask_neg_resized[..., np.newaxis]
                        canvas = wsi_object.wsi.get_patch(x=x, y=y, width=s, height=s, spacing=spacing, center=False)
                        canvas_masked = canvas * tissue_mask_neg_resized
                        # take care of tissue content
                        tissue_mask_pos = (tissue_mask > 0).astype(int)
                        tissue_mask_pos_scaled = (tissue_mask_pos * 255).astype(np.uint8)
                        tissue_mask_pos_pil = Image.fromarray(tissue_mask_pos_scaled)
                        tissue_mask_pos_pil_resized = tissue_mask_pos_pil.resize((s, s), resample=Image.NEAREST)
                        tissue_mask_pos_resized = np.asarray(tissue_mask_pos_pil_resized)
                        tissue_mask_pos_resized = (tissue_mask_pos_resized > 0).astype('uint8')
                        tissue_mask_pos_resized = tissue_mask_pos_resized[..., np.newaxis]
                        region_hm_restricted = region_hm * tissue_mask_pos_resized
                        # combine tissue and non-tissue content
                        region_hm_combined = region_hm_restricted + canvas_masked
                        img = Image.fromarray(region_hm_combined)
                    else:
                        img = Image.fromarray(region_hm)
                    img.save(Path(region_hm_output_dir, f"{x}_{y}.png"))

                    if threshold != None:
                        thresh_region_hm_output_dir = Path(
                            region_output_dir, "thresholded", f"head_{j}"
                        )
                        thresh_region_hm_output_dir.mkdir(exist_ok=True, parents=True)

                        att_mask = region_att_scores.copy()
                        att_mask[att_mask < threshold] = 0
                        att_mask[att_mask >= threshold] = 0.95

                        color_block = (cmap(att_mask) * 255)[:, :, :3].astype(np.uint8)
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
                        img.save(Path(thresh_region_hm_output_dir, f"{x}_{y}.png"))

                    if highlight != None:
                        highlight_region_hm_output_dir = Path(
                            region_output_dir, "highlighted", f"head_{j}"
                        )
                        highlight_region_hm_output_dir.mkdir(exist_ok=True, parents=True)
                        save_region = np.array(region.resize((s, s)))
                        rgba_region = np.dstack((save_region, np.zeros((s,s), dtype=np.uint8)+255))
                        att_mask = region_att_scores.copy()
                        att_mask[att_mask < highlight] = 0
                        att_mask[att_mask >= highlight] = 1
                        highlighted_hm = rgba_region * (att_mask >= highlight)[..., np.newaxis]
                        m_black = (highlighted_hm[:, :, 0:3] == [0,0,0]).all(2)
                        transparent_region = np.dstack((save_region, np.zeros((s,s), dtype=np.uint8)+int(255*opacity)))
                        highlighted_hm[m_black] = transparent_region[m_black]
                        img = Image.fromarray(highlighted_hm, mode='RGBA')
                        img.save(Path(highlight_region_hm_output_dir, f"{x}_{y}.png"))

    return region_output_dir

def get_slide_heatmaps_slide_level(
    slide_path: Path,
    coordinates_dir: Path,
    patch_model: nn.Module,
    region_model: nn.Module,
    slide_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    output_dir: Path,
    patch_size: int = 256,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    threshold: Optional[float] = None,
    highlight: Optional[float] = None,
    opacity: float = 0.3,
    granular: bool = False,
    offset: int = 1024,
    spacing: Optional[float] = None,
    gaussian_smoothing: bool = False,
    gaussian_offset: int = 128,
    patch_attn_mask: Optional[torch.Tensor] = None,
    mini_patch_attn_mask: Optional[torch.Tensor] = None,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    slide_device: torch.device = torch.device("cuda:0"),
    main_process: bool = True,
):
    """
    Returns heatmaps of slide-level Transformer attention for each region extracted in the slide.
    These can later be stitched together for slide-level heatmap visualization.

    Args:
    - slide_path (Path): path to the slide
    - coordinates_dir (Path): path to root folder containing region coordinates in .npy files
    - patch_model (nn.Module): patch-level Transformer
    - region_model (nn.Module): region-level Transformer
    - slide_model (nn.Module): slide-level Transformer
    - patch_size (int): size of patches used for unrolling input region
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - threshold (float): filter out regions with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - highlight (float): filter out regions with attention scores lower than this value (set to None to disbale heatmap highlighting)
    - opacity (float): if highlight, set opacity for non-highlighted regions on stitched heatmap
    - patch_device (torch.device): device on which patch_model is
    - region_device (torch.device): device on which region_model is
    - main_process (bool): controls tqdm display when running with multiple gpus
    """
    slide_id = slide_path.stem.replace(" ", "_")

    att, coords = get_slide_attention_scores(
        slide_path,
        coordinates_dir,
        patch_model,
        region_model,
        slide_model,
        transforms=transforms,
        patch_size=patch_size,
        downscale=downscale,
        granular=granular,
        offset=offset,
        spacing=spacing,
        patch_attn_mask=patch_attn_mask,
        mini_patch_attn_mask=mini_patch_attn_mask,
        patch_device=patch_device,
        region_device=region_device,
        slide_device=slide_device,
    )  # (M, region_size, region_size), (M)

    wsi = wsd.WholeSlideImage(slide_path, backend="asap")

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
    region_size = int(region_size_resized / resize_factor)

    slide_output_dir = Path(output_dir, "slide", f"{region_size}")
    slide_output_dir.mkdir(exist_ok=True, parents=True)

    with tqdm.tqdm(
        coordinates,
        desc="Gathering attention scores",
        unit="region",
        leave=True,
        disable=not main_process,
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
            g_offset = gaussian_offset // downscale

            if threshold != None:
                thresh_slide_hm_output_dir = slide_output_dir / "thresholded"
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
                highlight_slide_hm_output_dir = slide_output_dir / "highlighted"
                highlight_slide_hm_output_dir.mkdir(exist_ok=True, parents=True)
                save_region = np.array(region.resize((s, s)))
                rgba_region = np.dstack((save_region, np.zeros((s,s), dtype=np.uint8)+255))
                att_mask = att[k].copy()
                att_mask[att_mask < highlight] = 0
                att_mask[att_mask >= highlight] = 1
                highlighted_hm = rgba_region * (att_mask >= highlight)[..., np.newaxis]
                img = Image.fromarray(highlighted_hm)
                img.save(Path(highlight_slide_hm_output_dir, f"{x}_{y}.png"))

            if gaussian_smoothing:
                neighbors, descriptors = find_neighboring_regions((x,y), coords, region_size)
                if neighbors:
                    for n, d in zip(neighbors,descriptors):
                        n_idx = neighbors.index(n)
                        n_att = att[n_idx]
                        if d == "top":
                            smoothed_concat_attn = np.zeros((s*2, s))
                            concat_att = np.concatenate([n_att, att[k]], axis=0)
                            smoothed_concat_attn[s-g_offset:s+g_offset, :] = gaussian_filter(concat_att[s-g_offset:s+g_offset, :], sigma=offset//2, axes=0)
                            att[k, :g_offset, :] = smoothed_concat_attn[s:s+g_offset, :]
                        elif d == "bot":
                            smoothed_concat_attn = np.zeros((s*2, s))
                            concat_att = np.concatenate([att[k], n_att], axis=0)
                            smoothed_concat_attn[s-g_offset:s+g_offset, :] = gaussian_filter(concat_att[s-g_offset:s+g_offset, :], sigma=offset//2, axes=0)
                            att[k, s-g_offset:s, :] = smoothed_concat_attn[s-g_offset:s, :]
                        elif d == "left":
                            smoothed_concat_attn = np.zeros((s, s*2))
                            concat_att = np.concatenate([n_att, att[k]], axis=1)
                            smoothed_concat_attn[:, s-g_offset:s+g_offset] = gaussian_filter(concat_att[:, s-g_offset:s+g_offset], sigma=offset//2, axes=1)
                            att[k, :, :g_offset] = smoothed_concat_attn[:, s:s+g_offset]
                        elif d == "right":
                            smoothed_concat_attn = np.zeros((s, s*2))
                            concat_att = np.concatenate([att[k], n_att], axis=1)
                            smoothed_concat_attn[:, s-g_offset:s+g_offset] = gaussian_filter(concat_att[:, s-g_offset:s+g_offset], sigma=offset//2, axes=1)
                            att[k, :, s-g_offset:s] = smoothed_concat_attn[:, s-g_offset:s]

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
            slide_hm_output_dir = slide_output_dir / "regular"
            slide_hm_output_dir.mkdir(exist_ok=True, parents=True)
            img = Image.fromarray(hm)
            img.save(Path(slide_hm_output_dir, f"{x}_{y}.png"))

    return slide_output_dir


def get_slide_blended_heatmaps(
    slide_path: Path,
    coordinates_dir: Path,
    patch_model: nn.Module,
    region_model: nn.Module,
    slide_model: nn.Module,
    transforms: torchvision.transforms.Compose,
    level: str,
    output_dir: Path,
    method: str = "multiply",
    gamma: float = 0.5,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    threshold: Optional[float] = None,
    smoothing: Optional[Dict] = None,
    segmentation_mask_path: Optional[str] = None,
    spacing: Optional[float] = None,
    downsample: Optional[int] = None,
    background_pixel_value: Optional[int] = None,
    tissue_pixel_value: Optional[int] = None,
    patch_attn_mask: Optional[torch.Tensor] = None,
    mini_patch_attn_mask: Optional[torch.Tensor] = None,
    compute_patch_attention: bool = True,
    patch_device: torch.device = torch.device("cuda:0"),
    region_device: torch.device = torch.device("cuda:0"),
    slide_device: torch.device = torch.device("cuda:0"),
    main_process: bool = True,
):
    """
    Returns blended heatmaps (patch-level, region-level & slide-level Transformer heatmaps blended together) for each region extracted in the slide.
    These can later be stitched together for slide-level heatmap visualization.

    Args:
    - slide_path (Path): path to the slide
    - coordinates_dir (Path): path to root folder containing region coordinates in .npy files
    - patch_model (nn.Module): patch-level Transformer
    - region_model (nn.Module): region-level Transformer
    - slide_model (nn.Module): slide-level Transformer
    - level (str): level at which the model was trained on
    - output_dir (Path): output directory for saving heatmaps
    - gamma (float): factor weighting the importance given to frozen model attention scores w.r.t finetuned model attention scores
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    - threshold (float): filter out patches with attention scores lower than this value (set to None to disbale heatmap thresholding)
    - save_to_disk (bool): wether to save individual region heatmaps to disk
    - granular (bool): create additional offset patches to get more granular heatmaps
    - offset (int): if granular is True, uses this value to offset patches
    - patch_device (torch.device): device on which patch_model is
    - region_device (torch.device): device on which region_model is
    - slide_device (torch.device): device on which slide_model is
    - main_process (bool): controls tqdm display when running with multiple gpus
    """
    slide_id = slide_path.stem.replace(" ", "_")
    wsi = wsd.WholeSlideImage(slide_path, backend="asap")

    slide_attention, _ = get_slide_attention_scores(
        slide_path,
        coordinates_dir,
        patch_model,
        region_model,
        slide_model,
        transforms=transforms,
        patch_size=patch_size,
        downscale=downscale,
        granular=smoothing.slide,
        offset=smoothing.offset.slide,
        spacing=spacing,
        patch_attn_mask=patch_attn_mask,
        mini_patch_attn_mask=mini_patch_attn_mask,
        patch_device=patch_device,
        region_device=region_device,
        slide_device=slide_device,
    )  # (M, region_size, region_size), (M)

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
    region_size = int(region_size_resized / resize_factor)

    blended_output_dir = Path(output_dir, "blended", f"{region_size}_{patch_size}")
    blended_output_dir.mkdir(exist_ok=True, parents=True)

    nhead_patch = 1
    if compute_patch_attention:
        nhead_patch = patch_model.num_heads
    nhead_region = region_model.num_heads
    offset = smoothing.offset.region
    offset_ = offset

    mask_attn_patch = (mini_patch_attn_mask is not None)
    mask_attn_region = (patch_attn_mask is not None)
    mask_attention = mask_attn_patch or mask_attn_region

    # do a first pass on the slide to get the attention scores
    # and find the min/max values for normalization

    head_min = [float('inf')] * nhead_region
    head_max = [float('-inf')] * nhead_region

    with tqdm.tqdm(
        coordinates,
        desc="First pass over regions to grab min/max attention scores for normalization",
        unit="region",
        leave=True,
        disable=not main_process,
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

            pm, mpm = None, None
            if mask_attn_patch:
                mpm = mini_patch_attn_mask[k]
            if mask_attn_region:
                pm = patch_attn_mask[k].unsqueeze(0)

            _, patch_att, region_att = get_region_attention_scores(
                region,
                patch_model,
                region_model,
                transforms=transforms,
                patch_size=patch_size,
                mini_patch_size=mini_patch_size,
                downscale=downscale,
                patch_attn_mask=pm,
                mini_patch_attn_mask=mpm,
                patch_device=patch_device,
                region_device=region_device,
                compute_patch_attention=compute_patch_attention,
            ) # (n_patch**2, nhead, patch_size, patch_size) when downscale = 1

            for i in range(nhead_region):
                head_min[i] = min(head_min[i], region_att[i].min())
                head_max[i] = max(head_max[i], region_att[i].max())

            slide_att_scores = slide_attention[k]

            if smoothing.region:
                offset = int(offset_ * region_size / 4096)
                region2 = add_margin(
                    region.crop((offset, offset, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset,
                    right=offset,
                    color=(255, 255, 255),
                )
                pm2, mpm2 = None, None
                if mask_attention:
                    pm2, mpm2 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset,
                    )
                    if not mask_attn_patch:
                        mpm2 = None
                    if not mask_attn_region:
                        pm2 = None
                region3 = add_margin(
                    region.crop((offset * 2, offset * 2, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset * 2,
                    right=offset * 2,
                    color=(255, 255, 255),
                )
                pm3, mpm3 = None, None
                if mask_attention:
                    pm3, mpm3 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset*2,
                    )
                    if not mask_attn_patch:
                        mpm3 = None
                    if not mask_attn_region:
                        pm3 = None
                region4 = add_margin(
                    region.crop((offset * 3, offset * 3, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset * 3,
                    right=offset * 3,
                    color=(255, 255, 255),
                )
                pm4, mpm4 = None, None
                if mask_attention:
                    pm4, mpm4 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset*3,
                    )
                    if not mask_attn_patch:
                        mpm4 = None
                    if not mask_attn_region:
                        pm4 = None

                _, patch_att_2, region_att_2 = get_region_attention_scores(
                    region2,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm2,
                    mini_patch_attn_mask=mpm2,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=compute_patch_attention,
                )
                _, _, region_att_3 = get_region_attention_scores(
                    region3,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm3,
                    mini_patch_attn_mask=mpm3,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=compute_patch_attention,
                )
                _, _, region_att_4 = get_region_attention_scores(
                    region4,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm4,
                    mini_patch_attn_mask=mpm4,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=compute_patch_attention,
                )

                for i in range(nhead_region):
                    head_min[i] = min(
                        head_min[i],
                        region_att_2[i].min(),
                        region_att_3[i].min(),
                        region_att_4[i].min(),
                    )
                    head_max[i] = max(
                        head_max[i],
                        region_att_2[i].max(),
                        region_att_3[i].max(),
                        region_att_4[i].max(),
                    )

    ## second pass to get normalized attention scores

    with tqdm.tqdm(
        coordinates,
        desc="Second pass to get normalized attention scores",
        unit="region",
        leave=True,
        disable=not main_process,
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

            pm, mpm = None, None
            if mask_attn_patch:
                mpm = mini_patch_attn_mask[k]
            if mask_attn_region:
                pm = patch_attn_mask[k].unsqueeze(0)

            _, patch_att, region_att = get_region_attention_scores(
                region,
                patch_model,
                region_model,
                transforms=transforms,
                patch_size=patch_size,
                mini_patch_size=mini_patch_size,
                downscale=downscale,
                patch_attn_mask=pm,
                mini_patch_attn_mask=mpm,
                patch_device=patch_device,
                region_device=region_device,
                compute_patch_attention=compute_patch_attention,
            ) # (n_patch**2, nhead, patch_size, patch_size) when downscale = 1

            slide_att_scores = slide_attention[k]

            if smoothing.region:
                offset = int(offset_ * region_size / 4096)
                region2 = add_margin(
                    region.crop((offset, offset, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset,
                    right=offset,
                    color=(255, 255, 255),
                )
                pm2, mpm2 = None, None
                if mask_attention:
                    pm2, mpm2 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset,
                    )
                    if not mask_attn_patch:
                        mpm2 = None
                    if not mask_attn_region:
                        pm2 = None
                region3 = add_margin(
                    region.crop((offset * 2, offset * 2, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset * 2,
                    right=offset * 2,
                    color=(255, 255, 255),
                )
                pm3, mpm3 = None, None
                if mask_attention:
                    pm3, mpm3 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset*2,
                    )
                    if not mask_attn_patch:
                        mpm3 = None
                    if not mask_attn_region:
                        pm3 = None
                region4 = add_margin(
                    region.crop((offset * 3, offset * 3, region_size, region_size)),
                    top=0,
                    left=0,
                    bottom=offset * 3,
                    right=offset * 3,
                    color=(255, 255, 255),
                )
                pm4, mpm4 = None, None
                if mask_attention:
                    pm4, mpm4 = get_mask(
                        slide_path,
                        segmentation_mask_path,
                        x,
                        y,
                        region_size,
                        patch_size,
                        mini_patch_size,
                        spacing,
                        backend='asap',
                        downsample=downsample,
                        background_pixel_value=background_pixel_value,
                        tissue_pixel_value=tissue_pixel_value,
                        offset=offset*3,
                    )
                    if not mask_attn_patch:
                        mpm4 = None
                    if not mask_attn_region:
                        pm4 = None

                _, patch_att_2, region_att_2 = get_region_attention_scores(
                    region2,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm2,
                    mini_patch_attn_mask=mpm2,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=compute_patch_attention,
                )
                _, _, region_att_3 = get_region_attention_scores(
                    region3,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm3,
                    mini_patch_attn_mask=mpm3,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=compute_patch_attention,
                )
                _, _, region_att_4 = get_region_attention_scores(
                    region4,
                    patch_model,
                    region_model,
                    transforms=transforms,
                    patch_size=patch_size,
                    mini_patch_size=mini_patch_size,
                    downscale=downscale,
                    patch_attn_mask=pm4,
                    mini_patch_attn_mask=mpm4,
                    patch_device=patch_device,
                    region_device=region_device,
                    compute_patch_attention=compute_patch_attention,
                )

                offset_2 = offset // downscale
                offset_3 = (offset * 2) // downscale
                offset_4 = (offset * 3) // downscale

            s = region_size // downscale
            # given region is an RGB image, so the default filter for resizing is Resampling.BICUBIC
            # which is fine as we're resizing the image here, not attention scores
            save_region = np.array(region.resize((s, s)))

            with tqdm.tqdm(
                range(nhead_region),
                desc=f"Processing region [{k+1}/{nregions}]",
                unit="head",
                leave=False,
                disable=not main_process,
            ) as t2:
                for j in t2:

                    region_att_scores = normalize_region_scores(
                        region_att[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                    )

                    if smoothing.region:
                        region_att_scores_2 = normalize_region_scores(
                            region_att_2[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                        )
                        region_att_scores_3 = normalize_region_scores(
                            region_att_3[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                        )
                        region_att_scores_4 = normalize_region_scores(
                            region_att_4[j], size=(s,) * 2, min_val=head_min[j], max_val=head_max[j]
                        )
                        region_att_scores *= 100
                        region_att_scores_2 *= 100
                        region_att_scores_3 *= 100
                        region_att_scores_4 *= 100
                        new_region_att_scores_2 = np.zeros_like(region_att_scores_2)
                        new_region_att_scores_2[
                            offset_2:s, offset_2:s
                        ] = region_att_scores_2[: (s - offset_2), : (s - offset_2)]
                        new_region_att_scores_3 = np.zeros_like(region_att_scores_3)
                        new_region_att_scores_3[
                            offset_3:s, offset_3:s
                        ] = region_att_scores_3[: (s - offset_3), : (s - offset_3)]
                        new_region_att_scores_4 = np.zeros_like(region_att_scores_4)
                        new_region_att_scores_4[
                            offset_4:s, offset_4:s
                        ] = region_att_scores_4[: (s - offset_4), : (s - offset_4)]
                        region_overlay = np.ones_like(new_region_att_scores_2) * 100
                        region_overlay[offset_2:s, offset_2:s] += 100
                        region_overlay[offset_3:s, offset_3:s] += 100
                        region_overlay[offset_4:s, offset_4:s] += 100
                        region_att_scores = (
                            region_att_scores
                            + new_region_att_scores_2
                            + new_region_att_scores_3
                            + new_region_att_scores_4
                        ) / region_overlay

                    with tqdm.tqdm(
                        range(nhead_patch),
                        desc=f"Region head [{j+1}/{nhead_region}]",
                        unit="head",
                        leave=False,
                        disable=not main_process,
                    ) as t3:
                        for i in t3:

                            if compute_patch_attention:

                                patch_att_scores = concat_patch_scores(
                                    patch_att[:, i, :, :],
                                    region_size=region_size,
                                    patch_size=patch_size,
                                    size=(s // n_patch,) * 2,
                                )
                                patch_overlay = np.ones_like(patch_att_scores) * 100

                                if smoothing.patch:
                                    patch_att_scores_2 = concat_patch_scores(
                                        patch_att_2[:, i, :, :],
                                        region_size=region_size,
                                        patch_size=patch_size,
                                        size=(s // n_patch,) * 2,
                                    )
                                    patch_att_scores *= 100
                                    patch_att_scores_2 *= 100
                                    new_patch_att_scores_2 = np.zeros_like(
                                        patch_att_scores_2
                                    )
                                    new_patch_att_scores_2[
                                        offset_2:s, offset_2:s
                                    ] = patch_att_scores_2[
                                        : (s - offset_2), : (s - offset_2)
                                    ]
                                    patch_overlay = (
                                        np.ones_like(patch_att_scores_2) * 100 * 2
                                    )
                                    patch_overlay[offset_2:s, offset_2:s] += 100 * 2
                                    patch_att_scores = (
                                        (patch_att_scores + new_patch_att_scores_2)
                                        * 2
                                        / patch_overlay
                                    )

                                if smoothing.region:
                                    if level == "global":
                                        if method == "average":
                                            n = 2
                                            score = (
                                                region_att_scores * region_overlay * (1-gamma)
                                                + patch_att_scores * patch_overlay * (1-gamma)
                                            )
                                            if gamma < 1:
                                                score = score / (region_overlay * (1-gamma) + patch_overlay * (1-gamma))
                                        elif method == "multiply":
                                            score = (
                                                (region_att_scores ** (1-gamma)) * region_overlay
                                                * (patch_att_scores ** (1-gamma)) * patch_overlay
                                            )
                                    elif level == "local":
                                        if method == "average":
                                            n = 1
                                            score = (
                                                region_att_scores * region_overlay * gamma
                                                + patch_att_scores * patch_overlay * (1-gamma)
                                            ) / (region_overlay * gamma + patch_overlay * (1-gamma))
                                        elif method == "multiply":
                                            score = (
                                                (region_att_scores ** gamma) * region_overlay
                                                * (patch_att_scores ** (1-gamma)) * patch_overlay
                                            )
                                    else:
                                        raise ValueError(
                                            "Invalid level. Choose from ['global', 'local']"
                                        )
                                else:
                                    if level == "global":
                                        if method == "average":
                                            n = 2
                                            score = region_att_scores * (1-gamma) + patch_att_scores * (1-gamma)
                                        elif method == "multiply":
                                            score = region_att_scores ** (1-gamma) * patch_att_scores ** (1-gamma)
                                    elif level == "local":
                                        if method == "average":
                                            n = 1
                                            score = region_att_scores * gamma + patch_att_scores * (1-gamma)
                                        elif method == "multiply":
                                            score = region_att_scores ** gamma * patch_att_scores ** (1-gamma)
                                    else:
                                        raise ValueError(
                                            "Invalid level. Choose from ['global', 'local']"
                                        )

                            else:

                                if smoothing.region:
                                    if level == "global":
                                        if method == "average":
                                            n = 2
                                            score = region_att_scores * region_overlay * (1-gamma)
                                            if gamma < 1:
                                                score = score / (region_overlay * (1-gamma))
                                        elif method == "multiply":
                                            score = (region_att_scores ** (1-gamma)) * region_overlay
                                    elif level == "local":
                                        if method == "average":
                                            n = 1
                                            score = region_att_scores * region_overlay * gamma / (region_overlay * gamma)
                                        elif method == "multiply":
                                            score = (region_att_scores ** gamma) * region_overlay
                                    else:
                                        raise ValueError(
                                            "Invalid level. Choose from ['global', 'local']"
                                        )
                                            
                                else:
                                    if level == "global":
                                        if method == "average":
                                            n = 2
                                            score = region_att_scores * (1-gamma)
                                        elif method == "multiply":
                                            score = region_att_scores ** (1-gamma)
                                    elif level == "local":
                                        if method == "average":
                                            n = 1
                                            score = region_att_scores * gamma
                                        elif method == "multiply":
                                            score = region_att_scores ** gamma
                                    else:
                                        raise ValueError(
                                            "Invalid level. Choose from ['global', 'local']"
                                        )

                            if method == "average":
                                score += slide_att_scores * gamma
                                score = score / (n*(1-gamma)+(3-n)*gamma)
                            elif method == "multiply":
                                score *= slide_att_scores ** gamma

                            if threshold != None:

                                if compute_patch_attention:
                                    thresh_blended_hm_output_dir = Path(
                                        blended_output_dir, "thresholded", f"region-head-{j}-patch-head-{i}"
                                    )
                                else:
                                    thresh_blended_hm_output_dir = Path(
                                        blended_output_dir, "thresholded", f"region-head-{j}"
                                    )
                                thresh_blended_hm_output_dir.mkdir(exist_ok=True, parents=True)

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
                                        thresh_blended_hm_output_dir,
                                        f"{x}_{y}.png",
                                    )
                                )

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
                                blended_hm_output_dir = Path(blended_output_dir, "regular", f"region-head-{j}-patch-head-{i}")
                            else:
                                blended_hm_output_dir = Path(blended_output_dir, "regular", f"region-head-{j}")
                            blended_hm_output_dir.mkdir(exist_ok=True, parents=True)
                            img = Image.fromarray(region_hm)
                            img.save(Path(blended_hm_output_dir, f"{x}_{y}.png",))

    return blended_output_dir


def stitch_slide_heatmaps(
    slide_path: Path,
    heatmap_dir: Path,
    output_dir: Path,
    spacing: float,
    name: str = None,
    suffix: str = None,
    downsample: int = 32,
    downscale: int = 1,
    opacity: float = 0.3,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    segmentation_mask_path: Path = None,
    restrict_to_tissue: bool = False,
):
    """
    Returns region-level heatmaps stitched together at the slide-level.

    Args:
    - slide_path (Path): path to the whole slide image
    - heatmap_dir (Path): path to the directory containing heatmaps as .png files
    - output_dir (Path): output directory for saving heatmaps
    - spacing (float): pixel spacing (in mpp) at which regions were extracted for that slide
    - name (str): file naming template
    - downsample (int): uses this value to find the closest downsample level in the WSI for slide-level heatmap visualization
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - save_to_disk (bool): whether to save the stitched heatmap to disk
    - opacity (float): set opacity for non-tissue content on stitched heatmap
    - restrict_to_tissue (bool): whether to restrict highlighted regions to tissue content only
    - seg_params (Optional[Dict]): hyperparameters for tissue segmentation
    """
    wsi_object = WholeSlideImage(slide_path, mask_path=segmentation_mask_path, downsample=downsample)

    vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
    vis_spacing = wsi_object.get_level_spacing(vis_level)
    wsi_canvas = wsi_object.wsi.get_slide(spacing=vis_spacing)
    # x and y axes get inverted when using get_slide methid
    width, height, _ = wsi_canvas.shape

    # add an alpha channel, slightly transparent (255*alpha)
    wsi_canvas = np.dstack((wsi_canvas, np.zeros((width,height),dtype=np.uint8)+int(255*opacity)))
    wsi_canvas_ = np.copy(wsi_canvas)
    
    slide_output_dir = Path(output_dir, "slide")
    slide_output_dir.mkdir(exist_ok=True, parents=True)

    head_dirs = sorted([x for x in heatmap_dir.iterdir() if x.is_dir()])
    num_heads = 1
    sub_dir = False
    if len(head_dirs) > 0:
        sub_dir = True
        num_heads = len(head_dirs)

    with tqdm.tqdm(
        range(num_heads),
        desc="Stitching heatmaps",
        unit="head",
        leave=False,
    ) as t1:
        for i in t1:
            if not sub_dir:
                hms = [fp for fp in heatmap_dir.glob("*.png")]
                fname = f"{name}"
            else:
                hd = head_dirs[i]
                head_name = hd.stem
                hms = [fp for fp in hd.glob("*.png")]
                fname = f"{name}-{head_name}"
            canvas = None
            with tqdm.tqdm(
                hms,
                desc=f"Stitching attention head [{i+1}/{num_heads}]",
                unit="heatmap",
                leave=False,
            ) as t2:
                for fp in t2:
                    x, y = int(fp.stem.split("_")[0]), int(fp.stem.split("_")[1])
                    hm = np.array(Image.open(fp))
                    w, h, _ = hm.shape
                    # need to scale coordinates from level 0 to vis_level
                    downsample_factor = tuple(dv/ds for dv,ds in zip(wsi_object.level_downsamples[vis_level], wsi_object.level_downsamples[0]))
                    x_downsampled = int(round(x * 1 / downsample_factor[0], 0))
                    y_downsampled = int(round(y * 1 / downsample_factor[1], 0))
                    # need to scale heatmap from spacing level to vis level
                    spacing_level = wsi_object.get_best_level_for_spacing(spacing)
                    downsample_factor = tuple(dv/ds for dv,ds in zip(wsi_object.level_downsamples[vis_level], wsi_object.level_downsamples[spacing_level]))
                    w_downsampled = int(round(w * downscale / downsample_factor[0], 0))
                    h_downsampled = int(round(h * downscale / downsample_factor[1], 0))
                    if hm.shape[-1] == 3:  # if heatmap has 3 channels (RGB), add an alpha channel
                        hm = np.dstack((hm, np.full(hm.shape[:2], 255, dtype=np.uint8)))
                    # we're dealing with a palette hence values should remain in palette range
                    # hm is a RGB array so the default filter for resizing is Image.BICUBIC
                    # which would not be appropriate, as we want to keep the palette values
                    # instead, we should use Image.NEAREST
                    hm_downsampled = np.array(
                        Image.fromarray(hm, mode="RGBA").resize((w_downsampled, h_downsampled), resample=Image.NEAREST)
                    )

                    wsi_canvas[
                        y_downsampled : min(y_downsampled + h_downsampled, width),
                        x_downsampled : min(x_downsampled + w_downsampled, height),
                    ] = hm_downsampled[
                        : min(h_downsampled, width - y_downsampled),
                        : min(w_downsampled, height - x_downsampled),
                    ]

                    if restrict_to_tissue:
                        tissue_mask = wsi_object.binary_mask[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                        ]
                        tissue_mask = (tissue_mask > 0).astype(int)
                        tissue_mask = tissue_mask[..., np.newaxis]
                        wsi_canvas[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                        ] = wsi_canvas[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                        ] * tissue_mask
                        # make background slightly transparent
                        background_mask = (1 - tissue_mask).astype(np.uint8)
                        wsi_canvas[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                            3,
                        ] *= (1 - background_mask[..., 0])
                        wsi_canvas[
                            y_downsampled : min(y_downsampled + h_downsampled, width),
                            x_downsampled : min(x_downsampled + w_downsampled, height),
                            3,
                        ][background_mask[..., 0] == 1] = int(255 * opacity)
                        m_black = (wsi_canvas[:, :, 0:3] == [0,0,0]).all(2)
                        wsi_canvas[m_black] = wsi_canvas_[m_black]

            stitched_hm = Image.fromarray(wsi_canvas, mode='RGBA')

            # add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap)
            fig, ax = plt.subplots(dpi=150)
            plt.colorbar(sm, ax=ax)
            ax.remove()
            plt.yticks(fontsize='large')
            plt.savefig('color_bar.png', bbox_inches='tight', dpi=150)
            plt.close()
            cbar = Image.open('color_bar.png')
            os.remove('color_bar.png')
            w_cbar, h_cbar = cbar.size
            mode = stitched_hm.mode
            pad = 20
            w, h = stitched_hm.size
            if canvas is None:
                canvas = Image.new(
                    size=(w + 2 * pad + w_cbar, h), mode=mode, color=(255,) * len(mode)
                )
            canvas.paste(stitched_hm, (0, 0))
            x, y = w + pad, (h + 2 * pad - h_cbar) // 2
            canvas.paste(cbar, (x, y))

            if suffix:
                fname = f"{fname}-{suffix}"
            stitched_hm_path = Path(slide_output_dir, f"{fname}.png")
            canvas.save(stitched_hm_path, dpi=(300, 300))

    return slide_output_dir


def display_stitched_heatmaps(
    slide_path: Path,
    heatmap_dir: Path,
    output_dir: Path,
    name: str,
    display_patching: bool = False,
    draw_grid: bool = True,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    coordinates_dir: Optional[Path] = None,
    downsample: int = 32,
    font_fp: Path = Path("arial.ttf"),
    run_id: Optional[str] = None,
):
    """
    Display stitched heatmaps from multiple heads together, optionally alongside a visualization of patch extraction results.

    Args:
    - slide_path (Path): path to the whole slide image
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
    slide_id = slide_path.stem

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

        region_size = int(region_size_resized / resize_factor)

        wsi_object = WholeSlideImage(slide_path)
        vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
        vis_spacing = wsi_object.get_level_spacing(vis_level)
        slide_canvas = wsi_object.wsi.get_slide(spacing=vis_spacing)

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
    if run_id:
        color_bar_name = f"color_bar_{run_id}.png"
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


# Smoothing utility functions

def find_neighboring_regions(coord, coordinates, region_size, scheme: int = 4):
    x, y = coord
    if scheme == 4:
        neighbors_candidates = [(x,y-region_size), (x,y+region_size), (x-region_size,y), (x+region_size,y)]
        descriptors = ["top", "bot", "left", "right"]
    elif scheme == 8:
        pass
    neighbors = [n for n in neighbors_candidates if n in coordinates]
    descriptors = [d for i, d in enumerate(descriptors) if neighbors_candidates[i] in coordinates]
    if len(neighbors) > 0:
        return neighbors, descriptors
    else:
        return None, None


def find_l2r_neighbor(coords, region_coords, region_size):
    x, y = coords
    neigh = (x+region_size,y)
    if neigh in region_coords:
        return neigh
    else:
        return None


# def create_overlap_regions(coords, region_coords, region_dir: Path, region_size: int):
#     x1, y1 = coords
#     fp1 = Path(region_dir, f"{x1}_{y1}.jpg")
#     region1 = Image.open(fp1)
#     # find top-to-bottom neighbor
#     t2b_neighbor = find_t2b_neighbor(coords, region_coords, region_size)
#     # find left-ro-right neighbor
#     l2r_neighbor = find_l2r_neighbor(coords, region_coords, region_size)
#     # iterate over neighboring pairs
#     overlap_regions, overlap_coords = [], []
#     if t2b_neighbor != None:
#         x2, y2 = t2b_neighbor
#         fp2 = Path(region_dir, f"{x2}_{y2}.jpg")
#         region2 = Image.open(fp2)
#         #TODO: might need to inverse x,y axes
#         area1 = (0, region_size//2, region_size, region_size)
#         area2 = (0, 0, region_size, region_size//2)
#         crop1 = region1.crop(area1)
#         crop2 = region2.crop(area2)
#         canvas = Image.new(
#             size=(region_size,region_size), mode=region1.mode,
#         )
#         canvas.paste(crop1, (0, 0))
#         canvas.paste(crop2, (0, region_size//2))
#         overlap_regions.append(canvas)
#         overlap_coords.append(t2b_neighbor)
#     if l2r_neighbor != None:
#         x2, y2 = l2r_neighbor
#         fp2 = Path(region_dir, f"{x2}_{y2}.jpg")
#         region2 = Image.open(fp2)
#         #TODO: might need to inverse x,y axes
#         area1 = (region_size//2, 0, region_size, region_size)
#         area2 = (0, 0, region_size//2, region_size)
#         crop1 = region1.crop(area1)
#         crop2 = region2.crop(area2)
#         canvas = Image.new(
#             size=(region_size,region_size), mode=region1.mode,
#         )
#         canvas.paste(crop1, (0, 0))
#         canvas.paste(crop2, (region_size//2, 0))
#         overlap_regions.append(canvas)
#         overlap_coords.append(l2r_neighbor)
#     return overlap_regions, overlap_coords


def load_heatmaps_from_disk(root_dir, disable: bool = True):
    heatmaps, coords = defaultdict(list), defaultdict(list)
    nhead = len([_ for _ in root_dir.glob(f"head_*/")])
    with tqdm.tqdm(
        range(nhead),
        desc="Loading heatmaps from disk",
        unit="head",
        leave=False,
        disable=disable,
    ) as t1:
        for i in t1:
            head_dir = Path(root_dir, f"head_{i}")
            hms = [fp for fp in head_dir.glob("*.png")]
            with tqdm.tqdm(
                hms,
                desc=f"Loading heatmap for head [{i+1}/{nhead}]",
                unit="heatmap",
                leave=False,
                disable=disable,
            ) as t2:
                for fp in t2:
                    x, y = int(fp.stem.split("_")[0]), int(fp.stem.split("_")[1])
                    hm = np.array(Image.open(fp))
                    heatmaps[i].append(hm)
                    coords[i].append((x, y))
    return heatmaps, coords