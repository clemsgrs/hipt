import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat

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
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns]
    color_hm = np.concatenate([
        np.concatenate(color_block[i:(i+16)], axis=1)
        for i in range(0,256,16)
    ])
    return color_hm


def concat_region_scores(attn, size=(4096, 4096)):
    rank = lambda v: rankdata(v) / len(v)
    color_hm = rank(attn.flatten()).reshape(size)
    return color_hm


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


def create_patch_heatmaps_indiv_custom(
    patch,
    patch_model,
    output_dir,
    fname,
    threshold=0.5,
    offset=16,
    alpha=0.5,
    cmap=plt.get_cmap('coolwarm'),
    patch_device=torch.device('cuda:0'),
):
    """
    Creates patch heatmaps (saved individually).
    
    Args:
    - patch (PIL.Image): 256x256 input patch
    - patch_model (torch.nn): patch-level ViT 
    - output_dir (str): save directory
    - fname (str): naming structure of files
    - offset (int): how much to offset (from top-left corner with zero-padding) the region by for blending 
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    patch1 = patch.copy()
    _, att = get_patch_attention_scores(patch1, patch_model, patch_device=patch_device)
    save_region = np.array(patch.copy())
    s = 256

    if threshold != None:
        for i in range(6):
            att_scores = get_patch_scores(att[:,i,:,:], size=(s,)*2)
            att_mask = att_scores.copy()
            att_mask[att_mask < threshold] = 0
            att_mask[att_mask > threshold] = 0.95

            color_block = (cmap(att_mask)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            region_hm[att_mask == 0] = 0
            img_inverse = save_region.copy()
            img_inverse[att_mask == 0.95] = 0
            img_thresh = Image.fromarray(region_hm+img_inverse)
            img_thresh.save(Path(output_dir, f'{fname}_{s}_head_{i}_thresh.png'))

    for i in range(6):
        att_scores = get_patch_scores(att[:,i,:,:], size=(s,)*2)
        color_block = (cmap(att_scores)*255)[:,:,:3].astype(np.uint8)
        region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        img = Image.fromarray(region_hm)
        img.save(Path(output_dir, f'{fname}_{s}_head_{i}.png'))


        
def create_patch_heatmaps_concat(
    patch,
    patch_model,
    output_dir,
    fname,
    threshold=0.5,
    offset=16,
    alpha=0.5,
    cmap=plt.get_cmap('coolwarm'),
):
    """
    Creates patch heatmaps (concatenated for easy comparison)
    
    Args:
    - patch (PIL.Image): 256x256 input patch
    - patch_model (torch.nn): patch-level ViT 
    - output_dir (str): save directory
    - fname (str): naming structure of files
    - offset (int): how much to offset (from top-left corner with zero-padding) the region by for blending 
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    
    Returns:
    - None
    """
    patch1 = patch.copy()
    _, att = get_patch_attention_scores(patch1, patch_model)
    save_region = np.array(patch.copy())
    s = 256

    if threshold != None:
        ths = []
        for i in range(6):
            att_scores = get_patch_scores(att[:,i,:,:], size=(s,)*2)

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
        concat_img_thresh.save(Path(output_dir, f'{fname}_{s}thresh.png'))
    
    hms = []
    for i in range(6):
        att_scores = get_patch_scores(att[:,i,:,:], size=(s,)*2)
        color_block = (cmap(att_scores)*255)[:,:,:3].astype(np.uint8)
        region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        hms.append(region_hm)

    hms = [Image.fromarray(img) for img in hms]

    concat_img = getConcatImage(
        [
            getConcatImage(hms[0:3]),
            getConcatImage(hms[3:6])
        ], how='vertical').

    concat_img.save(Path(output_dir, f'{fname}_{s}_hm.png'))


def get_region_attention_scores(
    region,
    patch_model,
    region_model,
    scale=1,
    patch_device=torch.device('cuda:0'),
    region_device=torch.device('cuda:1'),
):
    """
    Forward pass in hierarchical model with attention scores saved.
    
    Args:
    - region (PIL.Image): 4096x4096 input region 
    - patch_model (torch.nn): patch-level ViT 
    - region_model (torch.nn): region-level Transformer 
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

    with torch.no_grad():   
        patches = t(region).unsqueeze(0).unfold(2, 256, 256).unfold(3, 256, 256)
        patches = rearrange(patches, 'b c p1 p2 w h -> (b p1 p2) c w h')
        patches = patches.to(patch_device, non_blocking=True)
        patch_features = patch_model(patches)

        patch_attention = patch_model.get_last_selfattention(patches)
        nh = patch_attention.shape[1] # number of head
        patch_attention = patch_attention[:, :, 0, 1:].reshape(256, nh, -1)
        patch_attention = patch_attention.reshape(256, nh, 16, 16)
        patch_attention = nn.functional.interpolate(patch_attention, scale_factor=int(16/scale), mode="nearest").cpu().numpy()

        region_features = patch_features.unfold(0, 16, 16).transpose(0,1).unsqueeze(dim=0)
        region_attention = region_model.get_last_selfattention(region_features.detach().to(region_device))
        nh = region_attention.shape[1] # number of head
        region_attention = region_attention[0, :, 0, 1:].reshape(nh, -1)
        region_attention = region_attention.reshape(nh, 16, 16)
        region_attention = nn.functional.interpolate(region_attention.unsqueeze(0), scale_factor=int(256/scale), mode="nearest")[0].cpu().numpy()

        if scale != 1:
            patches = nn.functional.interpolate(patches, scale_factor=(1/scale), mode="nearest")

    return tensorbatch2im(patches), patch_attention, region_attention


def create_hierarchical_heatmaps_indiv(
    region,
    patch_model,
    region_model,
    output_dir,
    fname,
    offset=128,
    scale=4,
    alpha=0.5,
    cmap=plt.get_cmap('coolwarm'),
    threshold=None,
):
    """
    Creates hierarchical heatmaps (Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps saved individually).  
    
    Args:
    - region (PIL.Image): 4096x4096 input region 
    - patch_model (torch.nn): patch-level ViT 
    - region_model (torch.nn): region-level Transformer 
    - output_dir (str): save directory / subdirectory
    - fname (str): naming structure of files
    - offset (int): how much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int): how much to scale the output image by 
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    _, patch_att, region_att = get_region_attention_scores(region, patch_model, region_model, scale)
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))
    
    if threshold != None:
        for i in range(6):
            patch_att_scores = concat_patch_scores(patch_att[:,i,:,:], size=(s//16,)*2)
            
            att_mask = patch_att_scores.copy()
            att_mask[att_mask < threshold] = 0
            att_mask[att_mask > threshold] = 0.95
            
            patch_color_block = (cmap(att_mask)*255)[:,:,:3].astype(np.uint8)
            patch_hm = cv2.addWeighted(patch_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            patch_hm[att_mask == 0] = 0
            img_inverse = save_region.copy()
            img_inverse[att_mask == 0.95] = 0
            img = Image.fromarray(patch_hm+img_inverse)
            img.save(os.path.join(output_dir, f'{fname}_256th[{i}].png'))
    
    if False:
        for j in range(6):
            region_att_scores = concat_region_scores(region_att[j], size=(s,)*2)
            region_att_scores = region_att_scores / 100
            region_color_block = (cmap(region_att_scores)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(region_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            img = Image.fromarray(region_hm)
            img.save(os.path.join(output_dir, f'{fname}_4k[{j}].png'))
        
    for j in range(6):
        region_att_scores = concat_region_scores(region_att[j], size=(s,)*2)
        region_color_block = (cmap(region_att_scores)*255)[:,:,:3].astype(np.uint8)
        region_hm = cv2.addWeighted(region_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        img = Image.fromarray(region_hm)
        img.save(os.path.join(output_dir, f'{fname}_1024[{j}].png'))
        
    for i in range(6):
        patch_att_scores = concat_patch_scores(patch_att[:,i,:,:], size=(s//16,)*2)
        patch_color_block = (cmap(patch_att_scores)*255)[:,:,:3].astype(np.uint8)
        patch_hm = cv2.addWeighted(patch_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        img = Image.fromarray(patch_hm)
        img.save(Path(output_dir, f'{fname}_256[{i}].png'))
    
    for j in range(6):
        region_att_scores = concat_region_scores(region_att[j], size=(s,)*2)
        for i in range(6):
            patch_att_scores = concat_patch_scores(patch_att[:,i,:,:], size=(s//16,)*2)
            score = region_att_scores + patch_att_scores
            color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            img = Image.fromarray(region_hm)
            img.save(Path(output_dir, f'{fname}_factorized_4k{j}]_256[{i}].png'))


def create_hierarchical_heatmaps_concat(
    region,
    patch_model,
    region_model,
    output_dir,
    fname,
    offset=128,
    scale=4,
    alpha=0.5,
    cmap=plt.get_cmap('coolwarm'),
):
    """
    Creates hierarchical heatmaps (With Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps concatenated for easy comparison)
    
    Args:
    - region (PIL.Image): 4096x4096 input region 
    - patch_model (torch.nn): patch-level ViT 
    - region_model (torch.nn): region-level Transformer 
    - output_dir (str): save directory / subdirectory
    - fname (str): naming structure of files
    - offset (int): how much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int): how much to scale the output image by 
    - alpha (float): image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): colormap for creating heatmaps
    """
    _, patch_att, region_att = get_region_attention_scores(region, patch_model, region_model, scale)
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))

    for j in range(6):
        region_att_scores = concat_region_scores(region_att[j], size=(s,)*2)
        region_color_block = (cmap(region_att_scores/100)*255)[:,:,:3].astype(np.uint8)
        region_hm = cv2.addWeighted(region_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        
        for i in range(6):
            patch_att_scores = concat_patch_scores(patch_att[:,i,:,:], size=(s//16,)*2)
            patch_color_block = (cmap(patch_att_scores)*255)[:,:,:3].astype(np.uint8)
            patch_hm = cv2.addWeighted(patch_color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        
            score = region_att_scores + patch_att_scores
            color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
            hierarchical_region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            
            pad = 100
            canvas = Image.new('RGB', (s*2+pad,)*2, (255,)*3)
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype("arial.ttf", 50)
            draw.text((1024*0.5-pad*2, pad//4), "ViT-256 (Head: %d)" % i, (0, 0, 0), font=font)
            canvas = canvas.rotate(90)
            draw = ImageDraw.Draw(canvas)
            draw.text((1024*1.5-pad, pad//4), "ViT-4K (Head: %d)" % j, (0, 0, 0), font=font)
            canvas.paste(Image.fromarray(save_region), (pad,pad))
            canvas.paste(Image.fromarray(region_hm), (1024+pad,pad))
            canvas.paste(Image.fromarray(patch_hm), (pad,1024+pad))
            canvas.paste(Image.fromarray(hierarchical_region_hm), (s+pad,s+pad))
            canvas.save(Path(output_dir, f'{fname}_4k[{j}]_256[{i}].png'))