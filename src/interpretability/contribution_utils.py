from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import seaborn as sns
from scipy.ndimage import gaussian_filter

from src.interpretability.wsi import WholeSlideImage, SegmentationParameters
from src.interpretability.utils import cmap_overlay


@torch.no_grad()
def risk_from_logits(logits: torch.Tensor):
    """
    logits: [..., K] per-bin logits
    returns: risk [...], using risk = -sum_t prod_{j<=t} (1 - sigmoid(z_j))
    """
    hazards = torch.sigmoid(logits)
    surv = torch.cumprod(1.0 - hazards, dim=-1)
    risk = -surv.sum(dim=-1)
    return risk


def clip_contributions(contributions: np.ndarray, lower=5, upper=95) -> np.ndarray:
    """
    Clip tile contribution scores using percentiles.
    Preserves sign and shrinks outliers.
    """
    lo = np.percentile(contributions, lower)
    hi = np.percentile(contributions, upper)
    return np.clip(contributions, lo, hi)


def normalize_to_unit(contributions: np.ndarray) -> np.ndarray:
    """
    Normalize contribution scores to [-1, 1] where 0 = no effect,
    -1 = strongest negative effect, +1 = strongest positive effect.
    """
    max_abs = np.max(np.abs(contributions))
    if max_abs == 0:
        return contributions
    return contributions / max_abs


def plot_distribution(contributions: np.ndarray, save_dir: Path, suffix: str | None = None):
    # KDE plot (fallback to histogram if seaborn KDE fails)
    plt.figure(figsize=(6, 4))
    try:
        sns.kdeplot(contributions, fill=True, bw_method="scott")
    except Exception:
        plt.hist(contributions, bins=50, density=True, alpha=0.6)
    plt.title(f"Contribution distribution")
    plt.xlabel("Contribution score")
    plt.ylabel("Density")
    plt.tight_layout()
    save_path = save_dir / f"contributions-distribution{f'-{suffix}' if suffix else ''}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


@torch.no_grad()
def get_tile_contribution_scores(
    wsi_path: Path,
    feature_path: Path,
    region_transformer: nn.Module,
    slide_transformer: nn.Module,
    classifier: nn.Module,
    zero_content: bool = True,
    batch_size: int = 32,
):

    slide_id = wsi_path.stem.replace(" ", "_")

    tile_features = torch.load(feature_path)

    num_regions = tile_features.shape[0]
    npatch = int(np.sqrt(tile_features.shape[1]))
    num_tiles = int(npatch * npatch)
    tile_features = tile_features.unfold(1, npatch, npatch).transpose(1, 2)

    device = tile_features.device

    # baseline forward (keep all tiles)
    base_mask = torch.ones(num_regions, num_tiles + 1, device=device, dtype=torch.long)
    region_features = region_transformer(tile_features, mask=base_mask)                                         # [M, E]
    wsi_feature = slide_transformer.global_phi(region_features)                                                 # [M, E]
    region_features_contextualized = slide_transformer.global_transformer(wsi_feature.unsqueeze(1)).squeeze(1)  # [M, E]
    att_full, _ = slide_transformer.global_attn_pool(region_features_contextualized)                            # [M, 1]
    att_full = att_full.T                                                                                       # [1, M]
    att_full = F.softmax(att_full, dim=1)                                                                       # [1, M]
    x_att = att_full @ region_features_contextualized                                                           # [1, M] @ [M, E] = [1, E]
    x_wsi = slide_transformer.global_rho(x_att)                                                                 # [1, E]
    logits = classifier(x_wsi)                                                                                  # [1, b]
    risk = risk_from_logits(logits).squeeze().item()

    del region_features_contextualized, wsi_feature, x_att, x_wsi, logits, att_full
    torch.cuda.empty_cache()

    contributions = torch.zeros((num_regions, num_tiles), device=device)
    with tqdm.tqdm(
        range(num_regions),
        desc=f"Processing {slide_id}",
        unit=" region",
        leave=True,
    ) as t:
        for r in t:
            # process removal of each tile in chunks to limit peak memory
            for start in range(0, num_tiles, batch_size):
                end = min(start + batch_size, num_tiles)
                idx = torch.arange(start, end, device=device)
                b = end - start

                # build mask_target: replicate baseline region mask and zero the tile index per sample
                mask_target = base_mask[r].unsqueeze(0).expand(b, -1).clone()           # [b, num_tiles+1]
                mask_target[torch.arange(b, device=device), idx + 1] = 0

                # build x_target by expanding only the target region r, not the whole tile_features
                x_target = tile_features[r].unsqueeze(0).expand(b, -1, -1, -1).clone()  # [b, e, npatch, npatch]

                if zero_content:
                    rows = (idx // npatch).long()
                    cols = (idx % npatch).long()
                    x_target[torch.arange(b, device=device), :, rows, cols] = 0.0

                # get region embedding for the modified region r in this chunk
                region_features_wo_r = region_transformer(x_target, mask=mask_target)   # [b, E]

                # compose full region embeddings per batch: others stay baseline, r replaced
                region_features_wo = region_features.unsqueeze(0).expand(b, -1, -1).clone()  # [b, M, E]
                region_features_wo[:, r, :] = region_features_wo_r

                # forward through slide-level agg.
                wsi_feature_wo = slide_transformer.global_phi(region_features_wo)                                                       # [b, ...]
                region_features_contextualized_wo = slide_transformer.global_transformer(wsi_feature_wo.transpose(0, 1)).transpose(0, 1)
                att_wo, _ = slide_transformer.global_attn_pool(region_features_contextualized_wo)   # [b, M, 1]
                att_wo = torch.transpose(att_wo, 2, 1)                                              # [b, 1, M]
                att_wo = F.softmax(att_wo, dim=-1)                                                  # [b, 1, M]
                x_att_wo = torch.bmm(att_wo, region_features_contextualized_wo)                     # [b, 1, E]
                x_wsi_wo = slide_transformer.global_rho(x_att_wo.squeeze(1))                        # [b, E]
                logits_wo = classifier(x_wsi_wo)                                                    # [b, b_out]
                risk_wo = risk_from_logits(logits_wo).squeeze()
                contribution = risk - risk_wo                                                       # [b]

                contributions[r, start:end] = contribution

                torch.cuda.empty_cache()

    return contributions # (num_regions, num_tiles)


@torch.no_grad()
def get_region_contribution_scores(
    wsi_path: Path,
    feature_path: Path,
    region_transformer: nn.Module,
    slide_transformer: nn.Module,
    classifier: nn.Module,
):
    """
    LOO with re-attention: recompute attention weights with region r removed.
    """

    slide_id = wsi_path.stem.replace(" ", "_")

    feature = torch.load(feature_path)

    npatch = int(np.sqrt(feature.shape[1]))
    feature = feature.unfold(1, npatch, npatch).transpose(1, 2)
    region_features_contextualized = region_transformer(feature)
    num_regions, _ = region_features_contextualized.shape

    # true risk: using all regions
    att_full, _ = slide_transformer.global_attn_pool(region_features_contextualized)    # [num_regions, 1]
    att_full = att_full.T                                                               # [1, num_regions]
    att_full = F.softmax(att_full, dim=1)                                               # [1, num_regions]
    x_att = att_full @ region_features_contextualized                                   # [1, num_regions] @ [num_regions, E] = [1, E]
    x_wsi = slide_transformer.global_rho(x_att)                                         # [1, E]
    logits = classifier(x_wsi)                                                          # [1, b]
    risk = risk_from_logits(logits).squeeze().item()

    # for each region r, drop it, recompute attention on the remaining regions
    contributions = []
    with tqdm.tqdm(
        range(num_regions),
        desc=f"Processing {slide_id}",
        unit=" region",
        leave=True,
    ) as t:
        for r in t:
            # mask out region r
            mask = torch.ones(num_regions, dtype=torch.bool, device=region_features_contextualized.device)
            mask[r] = False
            x_wo = region_features_contextualized[mask]                                                     # [num_regions-1, E]
            att_wo, _ = slide_transformer.global_attn_pool(x_wo)                                            # [num_regions-1, 1]
            att_wo = att_wo.T                                                                               # [1, num_regions-1]
            att_wo = F.softmax(att_wo, dim=1)                                                               # [1, num_regions-1]
            x_att_wo = att_wo @ x_wo                                                                        # [1, num_regions-1] @ [num_regions-1, E] = [1, E]
            x_wsi_wo = slide_transformer.global_rho(x_att_wo)                                               # [1, E]
            logits_wo = classifier(x_wsi_wo)                                                                # [1, b]
            risk_wo = risk_from_logits(logits_wo).squeeze()
            contribution = risk - risk_wo                                                                   # [num_tiles]
            contributions.append(contribution)

    contributions = torch.stack(contributions)                                          # [num_regions]
    return contributions # (num_regions,)


def stitch_contribution_scores(
    wsi_path: Path,
    contributions: list[np.ndarray],
    coordinates_dir: Path,
    output_dir: Path,
    spacing: float,
    tolerance: float,
    patch_size: int | None = None,
    name: str | None = None,
    segmentation_parameters: SegmentationParameters | None = None,
    downscale: int = 1,
    cmap: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("coolwarm"),
    segmentation_mask_path: Path | None = None,
    smoothing: bool = True,
    binarize: bool = False,
    opacity: float = 0.5,
    threshold: bool = False,
    restrict_to_tissue: bool = False,
    verbose: bool = True,
):
    """
    Returns region-level contribution scores stitched at the slide-level.

    Args:
    - wsi_path (Path): path to the whole slide image
    - contributions (list[np.ndarray]): list of contribution score arrays for each region
    - coordinates_dir (Path): path to the directory containing region coordinates as .npy files
    - output_dir (Path): output directory for saving heatmaps
    - spacing (float): pixel spacing (in mpp) at which regions were extracted for that slide
    - tolerance (float): tolerance when matching spacing levels
    - name (str): file naming template
    - downscale (int): how much to downscale the output heatmap by (e.g. downscale=4 will resize 256x256 heatmaps to 64x64)
    - cmap (matplotlib.colors.LinearSegmentedColormap): colormap for plotting heatmaps
    - opacity (float): set opacity for non-tissue content on stitched heatmap
    - threshold (bool): whether to filter out attention scores based on percentile thresholds
    - restrict_to_tissue (bool): whether to restrict highlighted regions to tissue content only
    """
    wsi_object = WholeSlideImage(wsi_path, mask_path=segmentation_mask_path, segment_params=segmentation_parameters)

    slide_name = wsi_object.name
    coordinates_file = coordinates_dir / f"{slide_name}.npy"
    coordinates_arr = np.load(coordinates_file)

    try:
        coordinates = list(zip(coordinates_arr[:,0], coordinates_arr[:,1]))
    except Exception as e:
        coordinates = list(zip(coordinates_arr["x"], coordinates_arr["y"]))

    vis_level = wsi_object.get_best_level_for_downsample_custom(segmentation_parameters.downsample)
    vis_spacing = wsi_object.get_level_spacing(vis_level)
    wsi_canvas = wsi_object.get_slide(spacing=vis_spacing)

    # x and y axes get inverted when using get_slide method
    height, width, _ = wsi_canvas.shape

    slide_output_dir = Path(output_dir, "visualization", "slide")
    slide_output_dir.mkdir(exist_ok=True, parents=True)

    stitched_contribution = np.full((height, width), np.nan) # wsi size at vis_level

    starting_spacing = wsi_object.get_level_spacing(0)

    # find spacing level and its spacing (for scaling tile sizes if necessary)
    spacing_level, is_within_tolerance = wsi_object.get_best_level_for_spacing(spacing, tolerance)
    tile_spacing = wsi_object.get_level_spacing(spacing_level)
    if not is_within_tolerance:
        resize_factor = spacing / tile_spacing
    else:
        resize_factor = 1.0

    # map size from tile_spacing -> spacing -> vis_spacing
    spacing_2_vis = vis_spacing / tile_spacing / resize_factor

    # need to scale coordinates from starting_spacing to vis_spacing
    starting_to_vis = vis_spacing / starting_spacing

    with tqdm.tqdm(
        contributions,
        desc="Stitching contribution scores",
        unit=" region",
        leave=False,
        disable=not verbose,
    ) as t2:
        for i, region_contribution in enumerate(t2):
            x, y = coordinates[i] # defined w.r.t level 0
            region_h, region_w = region_contribution.shape # at spacing

            # map coordinates from starting_spacing -> vis_spacing
            x_start_vis_f = x / starting_to_vis
            y_start_vis_f = y / starting_to_vis

            # map region size from spacing -> vis_spacing
            region_w_vis_f = region_w * downscale / spacing_2_vis
            region_h_vis_f = region_h * downscale / spacing_2_vis

            # compute float edges
            x_end_vis_f = x_start_vis_f + region_w_vis_f
            y_end_vis_f = y_start_vis_f + region_h_vis_f

            # round edges: floor(start), ceil(end)
            x_start_vis = int(np.floor(x_start_vis_f))
            y_start_vis = int(np.floor(y_start_vis_f))
            x_end_vis = int(np.ceil(x_end_vis_f))
            y_end_vis = int(np.ceil(y_end_vis_f))

            # resize region to integer size
            target_w = x_end_vis - x_start_vis
            target_h = y_end_vis - y_start_vis
            region_contribution_vis = cv2.resize(
                region_contribution,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )

            # clip to canvas bounds
            x_start_vis = max(0, x_start_vis)
            y_start_vis = max(0, y_start_vis)
            x_end_vis = min(width, x_end_vis)
            y_end_vis = min(height, y_end_vis)

            # determine portion of resized region to use
            slice_h = y_end_vis - y_start_vis
            slice_w = x_end_vis - x_start_vis

            # crop the resized-full tile to the portion that intersects the canvas
            cropped_vis = region_contribution_vis[:slice_h, :slice_w]

            # assign cropped portion to stitched canvas
            stitched_contribution[y_start_vis:y_end_vis, x_start_vis:x_end_vis] = cropped_vis

    if smoothing:
        sigma = int(patch_size / spacing_2_vis * 0.25)
        stitched_contribution = gaussian_filter(stitched_contribution, sigma=sigma)

    if threshold:
        # map values between percentiles to nan so they aren't colored in the overlay
        lo = np.nanpercentile(stitched_contribution, 5)
        hi = np.nanpercentile(stitched_contribution, 95)
        mask = (stitched_contribution >= lo) & (stitched_contribution <= hi)
        print(f"Proportion of values masked: {100 * np.sum(mask) / stitched_contribution.size:.4f}%")
        stitched_contribution[mask] = np.nan

    if binarize:
        stitched_contribution = np.where(
            np.isnan(stitched_contribution),      # if value is nan
            np.nan,                              # leave it as nan
            np.where(stitched_contribution > 0, 1.0, -1)  # otherwise binarize
        )

    if restrict_to_tissue:
        tissue_mask = np.where(wsi_object.binary_mask > 0, 1.0, np.nan)
        stitched_contribution = stitched_contribution * tissue_mask

    overlayed_contribution = cmap_overlay(stitched_contribution, wsi_canvas, cmap, opacity)

    stitched_hm_path = Path(slide_output_dir, f"{name}.png")
    overlayed_contribution.save(stitched_hm_path, dpi=(300, 300))

    return slide_output_dir
