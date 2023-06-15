import cv2
import tqdm
import h5py
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from einops import rearrange
from scipy.stats import rankdata
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from collections import OrderedDict, defaultdict

import source.vision_transformer as vits
from source.wsi import WholeSlideImage


def get_patch_model(pretrained_weights, arch='vit_small', device: Optional[torch.device] = None):
    
    checkpoint_key = 'teacher'
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    patch_model = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in patch_model.parameters():
        p.requires_grad = False
    patch_model.eval()
    patch_model.to(device)

    if pretrained_weights.is_file():
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = patch_model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return patch_model


def get_region_model(pretrained_weights, arch='vit4k_xs', region_size: int = 4096, device: Optional[torch.device] = None):
    
    checkpoint_key = 'teacher'
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    region_model = vits.__dict__[arch](img_size=region_size, num_classes=0)
    for p in region_model.parameters():
        p.requires_grad = False
    region_model.eval()
    region_model.to(device)

    if pretrained_weights.is_file():
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = region_model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return region_model


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


def get_patch_scores(attns, size=(256, 256)):
    rank = lambda v: rankdata(v) / len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns][0]
    return color_block


def concat_patch_scores(attns, size=(256, 256)):
    rank = lambda v: rankdata(v) / len(v)
    color_block = [
        rank(attn.flatten()).reshape(size) for attn in attns
    ]  # [(256, 256)] of length len(attns)
    color_hm = np.concatenate(
        [np.concatenate(color_block[i : (i + 16)], axis=1) for i in range(0, 256, 16)]
    )
    # (16*256, 16*256)
    return color_hm


def concat_region_scores(attn, size=(4096, 4096)):
    rank = lambda v: rankdata(v) / len(v)
    color_hm = rank(attn.flatten()).reshape(size)  # (4096, 4096)
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
        tuple(np.maximum([0, 0], coord - thickness // 2)),
        tuple(coord - thickness // 2 + np.array(shape)),
        (0, 0, 0, 255),
        thickness=thickness,
    )
    return img


def DrawMapFromCoords(
    canvas,
    wsi_object,
    coords,
    patch_size,
    vis_level: int,
    indices: Optional[List[int]] = None,
    draw_grid: bool = True,
    thickness: int = 2,
    verbose: bool = False,

):

    downsamples = wsi_object.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    patch_size = tuple(
        np.ceil((np.array(patch_size) / np.array(downsamples))).astype(np.int32)
    )
    if verbose:
        print(f"downscaled patch size: {patch_size}")

    for idx in range(total):

        patch_id = indices[idx]
        coord = coords[patch_id]
        x, y = coord
        spacing = wsi_object.spacings[vis_level]

        s = wsi_object.spacing_mapping[spacing]
        width, height = patch_size
        tile = wsi_object.wsi.get_patch(x, y, width, height, spacing=s, center=False)

        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[
            coord[1] : coord[1] + patch_size[1],
            coord[0] : coord[0] + patch_size[0],
            :3,
        ].shape[:2]
        canvas[
            coord[1] : coord[1] + patch_size[1],
            coord[0] : coord[0] + patch_size[0],
            :3,
        ] = tile[: canvas_crop_shape[0], : canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size, thickness=thickness)

    return Image.fromarray(canvas)


def get_patch_attention_scores(
    patch,
    patch_model,
    mini_patch_size: int = 16,
    downscale: int = 1,
    patch_device: torch.device = torch.device("cuda:0"),
):
    """
    Forward pass in patch-level ViT model with attention scores saved.

    Args:
    - patch (PIL.Image): input patch
    - patch_model (torch.nn): patch-level ViT
    - mini_patch_size (int): size of mini-patches used for unrolling input patch
    - downscale (int): how much to downscale the output image by (e.g. downscale=4 will resize images to be 64x64)

    Returns:
    - attention (torch.Tensor): [1, nhead, patch_size/downscale, patch_size/downscale] tensor of attention maps
    """
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    patch_size = patch.size[0]
    n_minipatch = patch_size // mini_patch_size

    with torch.no_grad():
        batch = t(patch).unsqueeze(0)   # (1, 3, patch_size, patch_size)
        batch = batch.to(patch_device, non_blocking=True)
        features = patch_model(batch)   # (1, 384)

        attention = patch_model.get_last_selfattention(batch)   # (1, 6, n_minipatch**2+1, n_minipatch**2+1)
        nh = attention.shape[1]  # number of head
        attention = attention[:, :, 0, 1:].reshape(n_minipatch**2, nh, -1)  # (1, 6, 1, n_minipatch**2) -> (n_minipatch**2, 6, 1)
        attention = attention.reshape(1, nh, n_minipatch, n_minipatch)  # (1, 6, n_minipatch, n_minipatch)
        attention = (
            nn.functional.interpolate(
                attention, scale_factor=int(mini_patch_size / downscale), mode="nearest"
            )
            .cpu()
            .numpy()
        )   # (1, 6, patch_size, patch_size) when downscale = 1

        if downscale != 1:
            batch = nn.functional.interpolate(
                batch, scale_factor=(1 / downscale), mode="nearest"
            )

    return tensorbatch2im(batch), attention


def create_patch_heatmaps_indiv(
    patch,
    patch_model,
    output_dir,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    threshold: float = 0.5,
    alpha: float = 0.5,
    cmap=plt.get_cmap("coolwarm"),
    save_to_disk: bool = True,
    patch_device: torch.device = torch.device("cuda:0"),
):
    """
    Creates patch heatmaps (saved individually).

    Args:
    - patch (PIL.Image): input patch
    - patch_model (torch.nn): patch-level ViT
    - output_dir (str): save directory
    - patch_size (int): size of input patch
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    patch1 = patch.copy()
    _, att = get_patch_attention_scores(patch1, patch_model, mini_patch_size=mini_patch_size, patch_device=patch_device)
    save_region = np.array(patch.copy())

    if threshold != None:

        with tqdm.tqdm(
            range(6),
            desc="Iterating over patch-level heads",
            unit=" head",
            leave=True,
        ) as t:

            for i in t:

                att_scores = get_patch_scores(att[:, i, :, :], size=(patch_size,) * 2)
                att_mask = att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask > threshold] = 0.95

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
                if save_to_disk:
                    img_thresh = Image.fromarray(patch_hm)
                    img_thresh.save(
                        Path(output_dir, f"{patch_size}_head_{i}_thresh.png")
                    )

    with tqdm.tqdm(
        range(6),
        desc="Iterating over patch-level heads",
        unit=" head",
        leave=True,
    ) as t:

        for i in t:

            att_scores = get_patch_scores(att[:, i, :, :], size=(patch_size,) * 2)
            color_block = (cmap(att_scores) * 255)[:, :, :3].astype(np.uint8)
            patch_hm = cv2.addWeighted(
                color_block, alpha, save_region.copy(), 1 - alpha, 0, save_region.copy()
            )
            if save_to_disk:
                img = Image.fromarray(patch_hm)
                img.save(Path(output_dir, f"{patch_size}_head_{i}.png"))


def create_patch_heatmaps_concat(
    patch,
    patch_model,
    output_dir,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    fname: str = "patch",
    threshold: float = 0.5,
    alpha: float = 0.5,
    cmap=plt.get_cmap("coolwarm"),
    patch_device=torch.device("cuda:0"),
):
    """
    Creates patch heatmaps (concatenated for easy comparison)

    Args:
    - patch (PIL.Image): input patch
    - patch_model (torch.nn): patch-level ViT
    - output_dir (str): save directory
    - patch_size (int): size of input patch
    - fname (str): naming structure of files
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    patch1 = patch.copy()
    _, att = get_patch_attention_scores(patch1, patch_model, mini_patch_size=mini_patch_size, patch_device=patch_device)
    save_region = np.array(patch.copy())

    if threshold != None:

        ths = []

        with tqdm.tqdm(
            range(6),
            desc="Iterating over patch-level heads",
            unit=" head",
            leave=True,
        ) as t:

            for i in t:

                att_scores = get_patch_scores(att[:, i, :, :], size=(patch_size,) * 2)

                att_mask = att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask > threshold] = 0.95

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
                ths.append(region_hm + img_inverse)

        ths = [Image.fromarray(img) for img in ths]

        concat_img_thresh = getConcatImage(
            [getConcatImage(ths[0:3]), getConcatImage(ths[3:6])], how="vertical"
        )
        concat_img_thresh.save(Path(output_dir, f"{fname}_{patch_size}_thresh.png"))

    hms = []

    with tqdm.tqdm(
        range(6),
        desc="Iterating over patch-level heads",
        unit=" head",
        leave=True,
    ) as t:

        for i in t:

            att_scores = get_patch_scores(att[:, i, :, :], size=(patch_size,) * 2)
            color_block = (cmap(att_scores) * 255)[:, :, :3].astype(np.uint8)
            region_hm = cv2.addWeighted(
                color_block, alpha, save_region.copy(), 1 - alpha, 0, save_region.copy()
            )
            hms.append(region_hm)

    hms = [Image.fromarray(img) for img in hms]

    concat_img = getConcatImage(
        [getConcatImage(hms[0:3]), getConcatImage(hms[3:6])], how="vertical"
    )

    concat_img.save(Path(output_dir, f"{fname}_{patch_size}_hm.png"))


def get_region_attention_scores(
    region,
    patch_model,
    region_model,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    patch_device=torch.device("cuda:0"),
    region_device=torch.device("cuda:1"),
):
    """
    Forward pass in hierarchical model with attention scores saved.

    Args:
    - region (PIL.Image): input region
    - patch_model (torch.nn): patch-level ViT
    - region_model (torch.nn): region-level Transformer
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output image by (e.g. downscale=4 will resize images to be 1024x1024)

    Returns:
    - np.array: [n_patch**2, patch_size/downscale, patch_size/downscale, 3] array sequence of image patch_size-sized patches from the input region.
    - patch_attention (torch.Tensor): [n_patch**2, nhead, patch_size/downscale, patch_size/downscale] tensor sequence of attention maps for patch_size-sized patches.
    - region_attention (torch.Tensor): [nhead, region_size/downscale, region_size/downscale] tensor sequence of attention maps for input region.
    """
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    region_size = region.size[0]
    n_patch = region_size // patch_size
    n_minipatch = patch_size // mini_patch_size

    with torch.no_grad():
        patches = (
            t(region)
            .unsqueeze(0)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
        )  # (1, 3, region_size, region_size) -> (1, 3, n_patch, n_patch, patch_size, patch_size)
        patches = rearrange(
            patches, "b c p1 p2 w h -> (b p1 p2) c w h"
        )  # (n_patch**2, 3, patch_size, patch_size)
        patches = patches.to(patch_device, non_blocking=True)
        patch_features = patch_model(patches)  # (n_patch**2, 384)

        patch_attention = patch_model.get_last_selfattention(patches)   # (n_patch**2, nhead, n_minipatch**2+1, n_minipatch**2+1)
        nh = patch_attention.shape[1]  # number of head
        patch_attention = patch_attention[:, :, 0, 1:].reshape(n_patch**2, nh, -1)  # (n_patch**2, nhead, n_minipatch**2)
        patch_attention = patch_attention.reshape(
            n_patch**2, nh, n_minipatch, n_minipatch
        )  # (n_patch**2, nhead, n_minipatch, n_minipatch)
        patch_attention = (
            nn.functional.interpolate(
                patch_attention, scale_factor=int(mini_patch_size / downscale), mode="nearest"
            )
            .cpu()
            .numpy()
        )  # (n_patch**2, nhead, patch_size, patch_size) when downscale = 1

        region_features = (
            patch_features.unfold(0, n_patch, n_patch)
            .transpose(0, 1)
            .unsqueeze(dim=0)
        )  # (1, 384, n_patch, n_patch)
        region_attention = region_model.get_last_selfattention(
            region_features.detach().to(region_device)
        ) # (1, nhead, n_patch**2+1, n_patch**2+1)
        nh = region_attention.shape[1]  # number of head
        region_attention = region_attention[0, :, 0, 1:].reshape(nh, -1)    # (nhead, 1, n_patch**2) -> (nhead, n_patch**2)
        region_attention = region_attention.reshape(
            nh, n_patch, n_patch
        ) # (nhead, n_patch, n_patch)
        region_attention = (
            nn.functional.interpolate(
                region_attention.unsqueeze(0),
                scale_factor=int(patch_size / downscale),
                mode="nearest",
            )[0]
            .cpu()
            .numpy()
        ) # (nhead, region_size, region_size) when downscale = 1

        if downscale != 1:
            patches = nn.functional.interpolate(
                patches, scale_factor=(1 / downscale), mode="nearest"
            )

    return tensorbatch2im(patches), patch_attention, region_attention


def create_hierarchical_heatmaps_indiv(
    region,
    patch_model,
    region_model,
    output_dir,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    fname: str = "region",
    downscale: int = 1,
    alpha: float = 0.5,
    cmap=plt.get_cmap("coolwarm"),
    threshold: Optional[float] = None,
    patch_device=torch.device("cuda:0"),
    region_device=torch.device("cuda:1"),
):
    """
    Creates hierarchical heatmaps (Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps saved individually).

    Args:
    - region (PIL.Image): input region
    - patch_model (torch.nn): patch-level ViT
    - region_model (torch.nn): region-level Transformer
    - output_dir (str): save directory / subdirectory
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - fname (str): naming structure of files
    - downscale (int): how much to downscale the output image by
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    region_size = region.size[0]
    _, patch_att, region_att = get_region_attention_scores(
        region,
        patch_model,
        region_model,
        patch_size=patch_size,
        mini_patch_size=mini_patch_size,
        downscale=downscale,
        patch_device=patch_device,
        region_device=region_device,
    )
    s = region_size // downscale
    save_region = np.array(region.resize((s, s)))

    if threshold != None:

        with tqdm.tqdm(
            range(6),
            desc="Iterating over patch-level heads",
            unit=" head",
            leave=True,
        ) as t:

            for i in t:

                patch_att_scores = concat_patch_scores(
                    patch_att[:, i, :, :], size=(s // mini_patch_size,) * 2
                )

                att_mask = patch_att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask > threshold] = 0.95

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
                img = Image.fromarray(patch_hm + img_inverse)
                img.save(Path(output_dir, f"{fname}_256_head_{i}_thresh.png"))

    with tqdm.tqdm(
        range(6),
        desc="Iterating over region-level heads",
        unit=" head",
        leave=True,
    ) as t:

        for j in t:

            region_att_scores = concat_region_scores(region_att[j], size=(s,) * 2)
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
            img.save(Path(output_dir, f"{fname}_1024_head_{j}.png"))

    with tqdm.tqdm(
        range(6),
        desc="Iterating over patch-level heads",
        unit=" head",
        leave=True,
    ) as t:

        for i in t:

            patch_att_scores = concat_patch_scores(
                patch_att[:, i, :, :], size=(s // mini_patch_size,) * 2
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
            )
            img = Image.fromarray(patch_hm)
            img.save(Path(output_dir, f"{fname}_256_head_{i}.png"))

    with tqdm.tqdm(
        range(6),
        desc="Iterating over region-level heads",
        unit=" head",
        leave=True,
    ) as t1:

        for j in t1:

            region_att_scores = concat_region_scores(region_att[j], size=(s,) * 2)

            with tqdm.tqdm(
                range(6),
                desc="Iterating over patch-level heads",
                unit=" head",
                leave=False,
            ) as t2:

                for i in t2:

                    patch_att_scores = concat_patch_scores(
                        patch_att[:, i, :, :], size=(s // mini_patch_size,) * 2
                    )
                    score = region_att_scores + patch_att_scores
                    color_block = (cmap(score) * 255)[:, :, :3].astype(np.uint8)
                    region_hm = cv2.addWeighted(
                        color_block,
                        alpha,
                        save_region.copy(),
                        1 - alpha,
                        0,
                        save_region.copy(),
                    )
                    img = Image.fromarray(region_hm)
                    img.save(
                        Path(
                            output_dir,
                            f"{fname}_factorized_4k_head_{j}_256_head_{i}.png",
                        )
                    )


def create_hierarchical_heatmaps_concat(
    region,
    patch_model,
    region_model,
    output_dir,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    fname: str = "region",
    downscale: int = 1,
    alpha: float = 0.5,
    cmap=plt.get_cmap("coolwarm"),
    patch_device=torch.device("cuda:0"),
    region_device=torch.device("cuda:1"),
):
    """
    Creates hierarchical heatmaps (With Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps concatenated for easy comparison)

    Args:
    - region (PIL.Image): input region
    - patch_model (torch.nn): patch-level ViT
    - region_model (torch.nn): region-level Transformer
    - output_dir (str): save directory / subdirectory
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - fname (str): naming structure of files
    - downscale (int): how much to downscale the output image by
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    region_size = region.size[0]
    _, patch_att, region_att = get_region_attention_scores(
        region,
        patch_model,
        region_model,
        patch_size=patch_size,
        mini_patch_size=mini_patch_size,
        downscale=downscale,
        patch_device=patch_device,
        region_device=region_device,
    )  # (256, 6, 128, 128), (6, 2048, 2048) when downscale = 2
    s = region_size // downscale  # 2048 for downscale = 2, region_size = 4096
    save_region = np.array(
        region.resize((s, s))
    )  # (2048, 2048) for downscale = 2, region_size = 4096

    with tqdm.tqdm(
        range(6),
        desc="Iterating over region-level heads",
        unit=" head",
        leave=True,
    ) as t1:

        for j in t1:

            region_att_scores = concat_region_scores(
                region_att[j], size=(s,) * 2
            )  # (2048, 2048) for downscale = 2
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
            )  # (2048, 2048) for downscale = 2, region_size = 4096

            with tqdm.tqdm(
                range(6),
                desc="Iterating over patch-level heads",
                unit=" head",
                leave=False,
            ) as t2:

                for i in t2:

                    patch_att_scores = concat_patch_scores(
                        patch_att[:, i, :, :], size=(s // mini_patch_size,) * 2
                    )  # (2048, 2048) for downscale = 2
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

                    score = region_att_scores + patch_att_scores
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
                    canvas.save(Path(output_dir, f"{fname}_4k[{j}]_256[{i}].png"))


def get_slide_patch_level_heatmaps(
    slide_id: str,
    patch_model,
    region_model,
    region_dir: Path,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap = plt.get_cmap('coolwarm'),
    threshold: Optional[float] = None,
    region_fmt: str = "jpg",
    save_to_disk: bool = False,
    patch_device: torch.device = torch.device('cuda:0'),
    region_device: torch.device = torch.device('cuda:0'),
):
    """
    Creates slide-level heatmaps of patch-level Transformer attention

    Args:
    - slide_id (str): input slide id
    - patch_model (torch.nn): patch-level Transformer
    - region_model (torch.nn): region-level Transformer
    - output_dir (str): save directory / subdirectory
    - patch_size (int): size of patches used for unrolling region_model inputs
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output image by
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    patch_output_dir = Path(output_dir, "patch")
    patch_output_dir.mkdir(exist_ok=True, parents=True)

    region_paths = [fp for fp in Path(region_dir, slide_id, "imgs").glob(f"*.{region_fmt}")]
    nregions = len(region_paths)
    patch_heatmaps, coords = defaultdict(list), defaultdict(list)

    with tqdm.tqdm(
        region_paths,
        desc=f"Processing {slide_id}",
        unit=" region",
        leave=True,
        position=0,
    ) as t1:

        for k, fp in enumerate(t1):

            region = Image.open(fp)
            region_size = region.size[0]
            _, patch_att, _ = get_region_attention_scores(
                region,
                patch_model,
                region_model,
                patch_size=patch_size,
                mini_patch_size=mini_patch_size,
                downscale=downscale,
                patch_device=patch_device,
                region_device=region_device,
            )
            s = region_size // downscale
            save_region = np.array(region.resize((s, s)))

            if threshold != None:

                patch_hm_output_dir = Path(patch_output_dir, f"{patch_size}_thresh")
                patch_hm_output_dir.mkdir(exist_ok=True, parents=True)

                with tqdm.tqdm(
                    range(6),
                    desc=f"Processing region [{k+1}/{nregions}]",
                    unit=" head",
                    leave=True,
                ) as t2:

                    for i in t2:

                        x, y = int(fp.stem.split('_')[0]), int(fp.stem.split('_')[1])
                        coords[i].append((x,y))

                        patch_att_scores = concat_patch_scores(
                            patch_att[:, i, :, :], size=(s // mini_patch_size,) * 2
                        )

                        att_mask = patch_att_scores.copy()
                        att_mask[att_mask < threshold] = 0
                        att_mask[att_mask > threshold] = 0.95

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
                        patch_heatmaps[i].append(patch_hm)
                        if save_to_disk:
                            img = Image.fromarray(patch_hm)
                            img.save(Path(patch_hm_output_dir, f"head_{i}.png"))

            else:

                patch_hm_output_dir = Path(patch_output_dir, f"{patch_size}")
                patch_hm_output_dir.mkdir(exist_ok=True, parents=True)

                with tqdm.tqdm(
                    range(6),
                    desc=f"Processing region [{k+1}/{nregions}]",
                    unit=" head",
                    leave=True,
                ) as t2:

                    for i in t2:

                        x, y = int(fp.stem.split('_')[0]), int(fp.stem.split('_')[1])
                        coords[i].append((x,y))

                        patch_att_scores = concat_patch_scores(
                            patch_att[:, i, :, :], size=(s // mini_patch_size,) * 2
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
                        )
                        patch_heatmaps[i].append(patch_hm)
                        if save_to_disk:
                            img = Image.fromarray(patch_hm)
                            img.save(Path(output_dir, f"head_{i}.png"))

    return patch_heatmaps, coords


def get_slide_region_level_heatmaps(
    slide_id: str,
    patch_model,
    region_model,
    region_dir: Path,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap = plt.get_cmap('coolwarm'),
    threshold: Optional[float] = None,
    region_fmt: str = "jpg",
    save_to_disk: bool = None,
    patch_device: torch.device = torch.device('cuda:0'),
    region_device: torch.device = torch.device('cuda:0'),
):
    """
    Creates slide-level heatmaps of region-level Transformer.

    Args:
    - slide_id (str): input slide id
    - patch_model (torch.nn): patch-level Transformer
    - region_model (torch.nn): region-level Transformer
    - output_dir (str): save directory / subdirectory
    - patch_size (int): size of patches used for unrolling region_model inputs
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output image by
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    region_output_dir = Path(output_dir, "region")
    region_output_dir.mkdir(exist_ok=True, parents=True)

    region_paths = [fp for fp in Path(region_dir, slide_id, "imgs").glob(f"*.{region_fmt}")]
    nregions = len(region_paths)
    region_heatmaps, coords = defaultdict(list), defaultdict(list)

    with tqdm.tqdm(
        region_paths,
        desc=f"Processing {slide_id}",
        unit=" region",
        leave=True,
        position=0,
    ) as t1:

        for k, fp in enumerate(t1):

            region = Image.open(fp)
            region_size = region.size[0]

            region_hm_output_dir = Path(region_output_dir, f"{region_size}")
            region_hm_output_dir.mkdir(exist_ok=True, parents=True)

            _, _, region_att = get_region_attention_scores(
                region,
                patch_model,
                region_model,
                patch_size=patch_size,
                mini_patch_size=mini_patch_size,
                downscale=downscale,
                patch_device=patch_device,
                region_device=region_device,
            )
            s = region_size // downscale
            save_region = np.array(region.resize((s, s)))

            with tqdm.tqdm(
                range(6),
                desc=f"Processing region [{k+1}/{nregions}]",
                unit=" head",
                leave=True,
            ) as t2:

                for j in t2:

                    x, y = int(fp.stem.split('_')[0]), int(fp.stem.split('_')[1])
                    coords[j].append((x,y))

                    region_att_scores = concat_region_scores(region_att[j], size=(s,) * 2)
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
                    region_heatmaps[j].append(region_hm)
                    if save_to_disk:
                        img = Image.fromarray(region_hm)
                        img.save(Path(region_hm_output_dir, f"head_{j}.png"))

    return region_heatmaps, coords



def get_slide_hierarchical_heatmaps(
    slide_id: str,
    patch_model,
    region_model,
    region_dir: Path,
    output_dir: Path,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    downscale: int = 1,
    alpha: float = 0.5,
    cmap = plt.get_cmap('coolwarm'),
    threshold: Optional[float] = None,
    region_fmt: str = "jpg",
    save_to_disk: bool = False,
    patch_device: torch.device = torch.device('cuda:0'),
    region_device: torch.device = torch.device('cuda:0'),
):
    """
    Creates slide-level hierarchical heatmaps (patch-level & region-level Transformer heatmaps blended together).

    Args:
    - slide_id (str): input slide id
    - patch_model (torch.nn): patch-level Transformer
    - region_model (torch.nn): region-level Transformer
    - output_dir (str): save directory / subdirectory
    - patch_size (int): size of patches used for unrolling region_model inputs
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - downscale (int): how much to downscale the output image by
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    region_paths = [fp for fp in Path(region_dir, slide_id, "imgs").glob(f"*.{region_fmt}")]
    nregions = len(region_paths)
    
    hierarchical_heatmaps, coords = defaultdict(dict), defaultdict(dict)
    hm_dict, coord_dict = defaultdict(list), defaultdict(list)

    with tqdm.tqdm(
        region_paths,
        desc=f"Processing {slide_id}",
        unit=" region",
        leave=True,
        position=0,
    ) as t1:

        for k, fp in enumerate(t1):

            region = Image.open(fp)
            region_size = region.size[0]

            hierarchical_hm_output_dir = Path(output_dir, f"hierarchical_{region_size}_{patch_size}")
            hierarchical_hm_output_dir.mkdir(exist_ok=True, parents=True)

            _, patch_att, region_att = get_region_attention_scores(
                region,
                patch_model,
                region_model,
                patch_size=patch_size,
                mini_patch_size=mini_patch_size,
                downscale=downscale,
                patch_device=patch_device,
                region_device=region_device,
            )
            s = region_size // downscale
            save_region = np.array(region.resize((s, s)))
        
            with tqdm.tqdm(
                range(6),
                desc=f"Processing region [{k+1}/{nregions}]",
                unit=" head",
                leave=True,
            ) as t1:

                for j in t1:

                    region_att_scores = concat_region_scores(region_att[j], size=(s,) * 2)

                    with tqdm.tqdm(
                        range(6),
                        desc="Iterating over patch-level heads",
                        unit=" head",
                        leave=False,
                    ) as t2:

                        for i in t2:

                            x, y = int(fp.stem.split('_')[0]), int(fp.stem.split('_')[1])
                            coord_dict[i].append((x,y))

                            patch_att_scores = concat_patch_scores(
                                patch_att[:, i, :, :], size=(s // mini_patch_size,) * 2
                            )
                            score = region_att_scores + patch_att_scores
                            color_block = (cmap(score) * 255)[:, :, :3].astype(np.uint8)
                            region_hm = cv2.addWeighted(
                                color_block,
                                alpha,
                                save_region.copy(),
                                1 - alpha,
                                0,
                                save_region.copy(),
                            )
                            hm_dict[i].append(region_hm)
                            if save_to_disk:
                                img = Image.fromarray(region_hm)
                                img.save(
                                    Path(
                                        hierarchical_hm_output_dir,
                                        f"rhead_{j}_phead_{i}.png",
                                    )
                                )

                    hierarchical_heatmaps[j] = hm_dict
                    coords[j] = coord_dict

    return hierarchical_heatmaps, coords


def stitch_slide_heatmaps(
    slide_path: Path,
    heatmaps: List[np.array],
    coords: List[Tuple[(int, int)]],
    output_dir: Path,
    fname: str,
    downsample: int = 32,
    downscale: int = 1,
    save_to_disk: bool = False,
):
    slide_output_dir = Path(output_dir, 'slide')
    slide_output_dir.mkdir(exist_ok=True, parents=True)

    wsi_object = WholeSlideImage(slide_path)
    vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
    (width, height) = wsi_object.level_dimensions[vis_level]
    vis_spacing = wsi_object.spacings[vis_level]
    canvas = wsi_object.wsi.get_patch(0, 0, width, height, spacing=wsi_object.spacing_mapping[vis_spacing], center=False)

    for hm, (x,y) in zip(heatmaps, coords):

        w, h, _ = hm.shape
        downsample_factor = wsi_object.level_downsamples[vis_level]
        x_downsampled = int(x * 1 / downsample_factor[0])
        y_downsampled = int(y * 1 / downsample_factor[1])
        w_downsampled = int(w * downscale * 1 / downsample_factor[0])
        h_downsampled = int(h * downscale * 1 / downsample_factor[1])
        hm_downsampled = np.array(Image.fromarray(hm).resize((w_downsampled, h_downsampled)))

        canvas[y_downsampled:y_downsampled + h_downsampled, x_downsampled :x_downsampled + w_downsampled] = hm_downsampled

    stitched_hm = Image.fromarray(canvas)
    if save_to_disk:
        stitched_hm_path = Path(slide_output_dir, f"{fname}.png")
        stitched_hm.save(stitched_hm_path)

    return stitched_hm


def display_stitched_heatmaps(
    slide_path: Path,
    heatmaps: Dict[str, Image.Image],
    output_dir: Path,
    display_patching: bool = False,
    region_dir: Optional[Path] = None,
    region_size: Optional[int] = None,
    downsample: int = 32,
    key: str = "coords",
):
    """
    """
    slide_id = slide_path.stem
    slide_dir = Path(region_dir, slide_id)

    if display_patching:

        hdf5_file_path = Path(slide_dir, f"{slide_id}.h5")
        h5_file = h5py.File(hdf5_file_path, "r")
        dset = h5_file[key]
        coords = dset[:]

        wsi_object = WholeSlideImage(slide_path)
        vis_level = wsi_object.get_best_level_for_downsample_custom(downsample)
        (width, height) = wsi_object.level_dimensions[vis_level]
        vis_spacing = wsi_object.spacings[vis_level]
        slide_canvas = wsi_object.wsi.get_patch(0, 0, width, height, spacing=wsi_object.spacing_mapping[vis_spacing], center=False)

        patching_im = DrawMapFromCoords(
            slide_canvas,
            wsi_object,
            coords,
            region_size,
            vis_level,
        )

        data = [(f"{region_size}x{region_size} patching", patching_im)] + [(k, v) for k,v in heatmaps.items()] 
        heatmaps = OrderedDict(data)

    nhm = len(heatmaps)
    w, h = heatmaps[0].size
    pad = 20
    canvas = Image.new(size=(w*nhm+pad*(nhm+1), h+2*pad), mode="RGB", color=(255,)*3)

    for i, (fname, hm) in enumerate(heatmaps.items()):
        x, y = w*i+pad*(i+1), pad
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (x//2, pad//2),
            fname,
            (0, 0, 0),
        )
        canvas.paste(hm, (x, y))