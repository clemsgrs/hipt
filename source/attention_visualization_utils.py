import cv2
import tqdm
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
from typing import Optional

import source.vision_transformer as vits


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
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def get_patch_scores(attns, size=(256,256)):
    rank = lambda v: rankdata(v) / len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns][0]
    return color_block


def concat_patch_scores(attns, size=(256,256)):
    rank = lambda v: rankdata(v) / len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns] # [(256, 256)] of length len(attns)
    color_hm = np.concatenate([
        np.concatenate(color_block[i:(i+16)], axis=1)
        for i in range(0,256,16)
    ])
    # (16*256, 16*256)
    return color_hm


def concat_region_scores(attn, size=(4096, 4096)):
    rank = lambda v: rankdata(v) / len(v)
    color_hm = rank(attn.flatten()).reshape(size)   # (4096, 4096)
    return color_hm


def getConcatImage(imgs, how='horizontal', gap=0):
    """
    Function to concatenate list of images (vertical or horizontal).

    Args:
        - imgs (list of PIL.Image): List of PIL Images to concatenate.
        - how (str): How the images are concatenated (either 'horizontal' or 'vertical')
        - gap (int): Gap (in px) between images

    Return:
        - dst (PIL.Image): Concatenated image result.
    """
    gap_dist = (len(imgs)-1)*gap
    
    if how == 'vertical':
        w, h = np.max([img.width for img in imgs]), np.sum([img.height for img in imgs])
        h += gap_dist
        curr_h = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))
        for img in imgs:
            dst.paste(img, (0, curr_h))
            curr_h += img.height + gap

    elif how == 'horizontal':
        w, h = np.sum([img.width for img in imgs]), np.min([img.height for img in imgs])
        w += gap_dist
        curr_w = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))

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
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def get_patch_attention_scores(
    patch,
    patch_model,
    scale=1,
    patch_device=torch.device('cuda:0'),
):
    """
    Forward pass in patch-level ViT model with attention scores saved.
    
    Args:
    - patch (PIL.Image):  256x256 input patch
    - patch_model (torch.nn): patch-level ViT
    - scale (int): how much to scale the output image by (e.g. scale=4 will resize images to be 1024x1024)
    
    Returns:
    - attention_256 (torch.Tensor): [1, 256/scale, 256/scale, 3] tensor of attention maps for 256-sized patches
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ])

    with torch.no_grad():   
        batch = t(patch).unsqueeze(0)
        batch = batch.to(patch_device, non_blocking=True)
        features = patch_model(batch)

        attention = patch_model.get_last_selfattention(batch)
        nh = attention.shape[1] # number of head
        attention = attention[:, :, 0, 1:].reshape(256, nh, -1)
        attention = attention.reshape(1, nh, 16, 16)
        attention = nn.functional.interpolate(attention, scale_factor=int(16/scale), mode="nearest").cpu().numpy()

        if scale != 1:
            batch = nn.functional.interpolate(batch, scale_factor=(1/scale), mode="nearest")
            
    return tensorbatch2im(batch), attention


def create_patch_heatmaps_indiv(
    patch,
    patch_model,
    output_dir,
    patch_size: int = 256,
    fname: str = 'patch',
    threshold: float = 0.5,
    alpha: float = 0.5,
    cmap=plt.get_cmap('coolwarm'),
    patch_device=torch.device('cuda:0'),
):
    """
    Creates patch heatmaps (saved individually).
    
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
    _, att = get_patch_attention_scores(patch1, patch_model, patch_device=patch_device)
    save_region = np.array(patch.copy())

    if threshold != None:

        with tqdm.tqdm(
            range(6),
            desc='Iterating over patch-level heads',
            unit=' head',
            leave=True,
        ) as t:

            for i in t:

                att_scores = get_patch_scores(att[:,i,:,:], size=(patch_size,)*2)
                att_mask = att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask > threshold] = 0.95

                color_block = (cmap(att_mask)*255)[:,:,:3].astype(np.uint8)
                region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
                region_hm[att_mask == 0] = 0
                img_inverse = save_region.copy()
                img_inverse[att_mask == 0.95] = 0
                img_thresh = Image.fromarray(region_hm+img_inverse)
                img_thresh.save(Path(output_dir, f'{fname}_{patch_size}_head_{i}_thresh.png'))

    with tqdm.tqdm(
        range(6),
        desc='Iterating over patch-level heads',
        unit=' head',
        leave=True,
    ) as t:

        for i in t:

            att_scores = get_patch_scores(att[:,i,:,:], size=(patch_size,)*2)
            color_block = (cmap(att_scores)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            img = Image.fromarray(region_hm)
            img.save(Path(output_dir, f'{fname}_{patch_size}_head_{i}.png'))

        
def create_patch_heatmaps_concat(
    patch,
    patch_model,
    output_dir,
    patch_size: int = 256,
    fname: str = 'patch',
    threshold: float = 0.5,
    alpha: float = 0.5,
    cmap=plt.get_cmap('coolwarm'),
    patch_device=torch.device('cuda:0'),
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
    _, att = get_patch_attention_scores(patch1, patch_model, patch_device=patch_device)
    save_region = np.array(patch.copy())

    if threshold != None:

        ths = []

        with tqdm.tqdm(
            range(6),
            desc='Iterating over patch-level heads',
            unit=' head',
            leave=True,
        ) as t:

            for i in t:

                att_scores = get_patch_scores(att[:,i,:,:], size=(patch_size,)*2)

                att_mask = att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask > threshold] = 0.95

                color_block = (cmap(att_mask)*255)[:,:,:3].astype(np.uint8)
                region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
                region_hm[att_mask == 0] = 0
                img_inverse = save_region.copy()
                img_inverse[att_mask == 0.95] = 0
                ths.append(region_hm+img_inverse)
            
        ths = [Image.fromarray(img) for img in ths]
            
        concat_img_thresh = getConcatImage(
            [getConcatImage(ths[0:3]),
            getConcatImage(ths[3:6])
        ], how='vertical')
        concat_img_thresh.save(Path(output_dir, f'{fname}_{patch_size}_thresh.png'))
    
    hms = []

    with tqdm.tqdm(
        range(6),
        desc='Iterating over patch-level heads',
        unit=' head',
        leave=True,
    ) as t:

        for i in t:

            att_scores = get_patch_scores(att[:,i,:,:], size=(patch_size,)*2)
            color_block = (cmap(att_scores)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            hms.append(region_hm)

    hms = [Image.fromarray(img) for img in hms]

    concat_img = getConcatImage(
        [
            getConcatImage(hms[0:3]),
            getConcatImage(hms[3:6])
        ], how='vertical')

    concat_img.save(Path(output_dir, f'{fname}_{patch_size}_hm.png'))


def get_region_attention_scores(
    region,
    patch_model,
    region_model,
    patch_size: int = 256,
    mini_patch_size: int = 16,
    scale: int = 1,
    patch_device=torch.device('cuda:0'),
    region_device=torch.device('cuda:1'),
):
    """
    Forward pass in hierarchical model with attention scores saved.
    
    Args:
    - region (PIL.Image): input region 
    - patch_model (torch.nn): patch-level ViT 
    - region_model (torch.nn): region-level Transformer 
    - patch_size (int): size of patches used for unrolling input region
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - scale (int): how much to scale the output image by (e.g. scale=4 will resize images to be 1024x1024)
    
    Returns:
    - np.array: [256, 256/scale, 256/scale, 3] np.array sequence of image patches from the 4K x 4K region.
    - patch_attention (torch.Tensor): [256, 256/scale, 256/scale, 3] torch.Tensor sequence of attention maps for 256-sized patches.
    - region_attention (torch.Tensor): [1, 4096/scale, 4096/scale, 3] torch.Tensor sequence of attention maps for input region.
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ])
    n_patch = region_size // patch_size
    n_minipatch = patch_size // mini_patch_size

    with torch.no_grad():   
        patches = t(region).unsqueeze(0).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) # (1, 3, region_size, region_size) -> (1, 3, n_patch, n_patch, patch_size, patch_size)
        patches = rearrange(patches, 'b c p1 p2 w h -> (b p1 p2) c w h') # (n_patch**2, 3, patch_size, patch_size)
        patches = patches.to(patch_device, non_blocking=True)
        patch_features = patch_model(patches)   # (n_patch**2, 384)

        patch_attention = patch_model.get_last_selfattention(patches)
        nh = patch_attention.shape[1] # number of head
        patch_attention = patch_attention[:, :, 0, 1:].reshape(n_minipatch**2, nh, -1)
        patch_attention = patch_attention.reshape(n_minipatch**2, nh, mini_patch_size, mini_patch_size) # (n_minipatch**2, 6, 16, 16)
        patch_attention = nn.functional.interpolate(patch_attention, scale_factor=int(16/scale), mode="nearest").cpu().numpy()  # (n_minipatch**2, 6, 256, 256) when scale = 1

        region_features = patch_features.unfold(0, mini_patch_size, mini_patch_size).transpose(0,1).unsqueeze(dim=0)    # (n_minipatch**2, 384)
        region_attention = region_model.get_last_selfattention(region_features.detach().to(region_device))
        nh = region_attention.shape[1] # number of head
        region_attention = region_attention[0, :, 0, 1:].reshape(nh, -1)
        region_attention = region_attention.reshape(nh, mini_patch_size, mini_patch_size)
        region_attention = nn.functional.interpolate(region_attention.unsqueeze(0), scale_factor=int(256/scale), mode="nearest")[0].cpu().numpy()

        if scale != 1:
            patches = nn.functional.interpolate(patches, scale_factor=(1/scale), mode="nearest")

    return tensorbatch2im(patches), patch_attention, region_attention


def create_hierarchical_heatmaps_indiv(
    region,
    patch_model,
    region_model,
    output_dir,
    mini_patch_size: int = 16,
    fname: str = 'region',
    scale: int = 4,
    alpha: float = 0.5,
    cmap=plt.get_cmap('coolwarm'),
    threshold: Optional[float] = None,
    patch_device=torch.device('cuda:0'),
    region_device=torch.device('cuda:1'),
):
    """
    Creates hierarchical heatmaps (Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps saved individually).  
    
    Args:
    - region (PIL.Image): input region 
    - patch_model (torch.nn): patch-level ViT 
    - region_model (torch.nn): region-level Transformer 
    - output_dir (str): save directory / subdirectory
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - fname (str): naming structure of files
    - scale (int): how much to scale the output image by 
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    region_size = region.size[0]
    _, patch_att, region_att = get_region_attention_scores(region, patch_model, region_model, scale=scale, patch_device=patch_device, region_device=region_device)
    s = region_size // scale
    save_region = np.array(region.resize((s, s)))
    
    if threshold != None:

        with tqdm.tqdm(
            range(6),
            desc='Iterating over patch-level heads',
            unit=' head',
            leave=True,
        ) as t:

            for i in t:

                patch_att_scores = concat_patch_scores(patch_att[:,i,:,:], size=(s//mini_patch_size,)*2)
                
                att_mask = patch_att_scores.copy()
                att_mask[att_mask < threshold] = 0
                att_mask[att_mask > threshold] = 0.95
                
                patch_color_block = (cmap(att_mask)*255)[:,:,:3].astype(np.uint8)
                patch_hm = cv2.addWeighted(patch_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
                patch_hm[att_mask == 0] = 0
                img_inverse = save_region.copy()
                img_inverse[att_mask == 0.95] = 0
                img = Image.fromarray(patch_hm+img_inverse)
                img.save(Path(output_dir, f'{fname}_256_head_{i}_thresh.png'))

    with tqdm.tqdm(
        range(6),
        desc='Iterating over region-level heads',
        unit=' head',
        leave=True,
    ) as t:

        for j in t:

            region_att_scores = concat_region_scores(region_att[j], size=(s,)*2)
            region_color_block = (cmap(region_att_scores)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(region_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            img = Image.fromarray(region_hm)
            img.save(Path(output_dir, f'{fname}_1024_head_{j}.png'))

    with tqdm.tqdm(
        range(6),
        desc='Iterating over patch-level heads',
        unit=' head',
        leave=True,
    ) as t:

        for i in t:

            patch_att_scores = concat_patch_scores(patch_att[:,i,:,:], size=(s//mini_patch_size,)*2)
            patch_color_block = (cmap(patch_att_scores)*255)[:,:,:3].astype(np.uint8)
            patch_hm = cv2.addWeighted(patch_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            img = Image.fromarray(patch_hm)
            img.save(Path(output_dir, f'{fname}_256_head_{i}.png'))

    with tqdm.tqdm(
        range(6),
        desc='Iterating over region-level heads',
        unit=' head',
        leave=True,
    ) as t1:

        for j in t1:

        region_att_scores = concat_region_scores(region_att[j], size=(s,)*2)

        with tqdm.tqdm(
            range(6),
            desc='Iterating over patch-level heads',
            unit=' head',
            leave=False,
        ) as t2:

            for i in t2:

                patch_att_scores = concat_patch_scores(patch_att[:,i,:,:], size=(s//mini_patch_size,)*2)
                score = region_att_scores + patch_att_scores
                color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
                region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
                img = Image.fromarray(region_hm)
                img.save(Path(output_dir, f'{fname}_factorized_4k_head_{j}_256_head_{i}.png'))


def create_hierarchical_heatmaps_concat(
    region,
    patch_model,
    region_model,
    output_dir,
    mini_patch_size: int = 16,
    fname: str = 'region',
    scale: int = 4,
    alpha: float = 0.5,
    cmap=plt.get_cmap('coolwarm'),
    patch_device=torch.device('cuda:0'),
    region_device=torch.device('cuda:1'),
):
    """
    Creates hierarchical heatmaps (With Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps concatenated for easy comparison)
    
    Args:
    - region (PIL.Image): input region 
    - patch_model (torch.nn): patch-level ViT 
    - region_model (torch.nn): region-level Transformer 
    - output_dir (str): save directory / subdirectory
    - mini_patch_size (int): size of mini-patches used for unrolling patch_model inputs
    - fname (str): naming structure of files
    - scale (int): how much to scale the output image by 
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    region_size = region.size[0]
    _, patch_att, region_att = get_region_attention_scores(region, patch_model, region_model, scale=scale, patch_device=patch_device, region_device=region_device) # (256, 6, 128, 128), (6, 2048, 2048) when scale = 2
    s = region_size // scale                        # 2048 for scale = 2, region_size = 4096
    save_region = np.array(region.resize((s, s)))   # (2048, 2048) for scale = 2, region_size = 4096

    with tqdm.tqdm(
        range(6),
        desc='Iterating over region-level heads',
        unit=' head',
        leave=True,
    ) as t1:

        for j in t1:

            region_att_scores = concat_region_scores(region_att[j], size=(s,)*2)    # (2048, 2048) for scale = 2
            region_color_block = (cmap(region_att_scores)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(region_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())  # (2048, 2048) for scale = 2, region_size = 4096
            
            with tqdm.tqdm(
                range(6),
                desc='Iterating over patch-level heads',
                unit=' head',
                leave=False,
            ) as t2:

                for i in t2:

                    patch_att_scores = concat_patch_scores(patch_att[:,i,:,:], size=(s//mini_patch_size,)*2) # (2048, 2048) for scale = 2
                    patch_color_block = (cmap(patch_att_scores)*255)[:,:,:3].astype(np.uint8)
                    patch_hm = cv2.addWeighted(patch_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())    # (2048, 2048) for scale = 2
                
                    score = region_att_scores + patch_att_scores
                    color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
                    hierarchical_region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())    # (2048, 2048) for scale = 2
                    
                    pad = 100
                    canvas = Image.new('RGB', (s*2+pad*3,)*2, (255,)*3)   # (2 * region_size // scale + 100, 2 * region_size // scale + 100, 3) ; (4096, 4096, 3) for scale = 2, region_size = 4096
                    draw = ImageDraw.Draw(canvas)
                    draw.text((s*0.5-pad*2, pad//4), f"patch-level Transformer (Head: {i})", (0, 0, 0))
                    canvas = canvas.rotate(90)
                    draw = ImageDraw.Draw(canvas)
                    draw.text((s*1.5-pad, pad//4), f"region-level Transformer (Head: {j})", (0, 0, 0))
                    canvas.paste(Image.fromarray(save_region), (pad,pad))                       # (2048, 2048) for scale = 2, region_size = 4096 ; (100, 100)
                    canvas.paste(Image.fromarray(region_hm), (s+2*pad,pad))                     # (2048, 2048) for scale = 2, region_size = 4096 ; (2048+100, 100)
                    canvas.paste(Image.fromarray(patch_hm), (pad,s+2*pad))                      # (2048, 2048) for scale = 2, region_size = 4096 ; (100, 2048+100)
                    canvas.paste(Image.fromarray(hierarchical_region_hm), (s+2*pad,s+2*pad))    # (2048, 2048) for scale = 2, region_size = 4096 ; (2048+100, 2048+100)
                    canvas.save(Path(output_dir, f'{fname}_4k[{j}]_256[{i}].png'))
