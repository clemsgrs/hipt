import os
from pathlib import Path
from typing import Optional
import getpass
from huggingface_hub import login
from PIL import Image
import cv2

import matplotlib
import numpy as np
import torch
import omegaconf
import wholeslidedata as wsd


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


def cmap_overlay(arr: np.ndarray, canvas: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap, alpha: float) -> Image.Image:
    """
    Create an OpenCV-style heatmap overlay:
    - arr: (H, W), values in [0,1]
    - wsi_background: (H, W, 3), RGB uint8
    Returns: blended RGB image (H, W, 3)
    """
    # create colormap (RGB only)
    colored_arr = (cmap(arr)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)
    # blend colored attention with canvas
    overlay = cv2.addWeighted(
        colored_arr,
        alpha,
        canvas,
        1 - alpha,
        0,
    )
    overlay_img = Image.fromarray(overlay)
    return overlay_img


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