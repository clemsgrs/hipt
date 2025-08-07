import torch
import random
import argparse
import matplotlib
import numpy as np

from pathlib import Path
from omegaconf import DictConfig

from src.utils import setup
from src.interpretability.utils import (
    cmap_map,
    get_patch_transformer,
    get_region_transformer,
    get_case_transformer,
    generate_masks,
    get_config_from_path,
    get_region_level_heatmaps,
    get_case_level_heatmaps,
    get_factorized_heatmaps,
    stitch_slide_heatmaps,
    display_stitched_heatmaps,
    create_transforms,
)


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("attention-maps", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command. For Yacs configs, use space-separated \"PATH.KEY VALUE\" pairs. For python-based LazyConfig, use \"path.key=value\".",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory to save logs and checkpoints",
    )
    return parser


def main(cfg: DictConfig):

    cfg = setup(args)

    wsi_path = Path(cfg.wsi_path)
    wsi_name = wsi_path.stem.replace(" ", "_")

    mask_path = None
    if cfg.mask_path:
        mask_path = Path(cfg.mask_path)

    tiling_config = get_config_from_path(cfg.tiling_config)

    output_dir = Path(cfg.output_dir, wsi_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    coordinates_dir = Path(cfg.coordinates_dir)

    mask_attention = (cfg.encoder.mask_attn is True) or (cfg.aggregator.mask_attn is True)

    # seed everything to ensure reproducible heatmaps
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### INSTANTIATE MODEL COMPONENTS

    print("=+="*10)
    patch_transformer = get_patch_transformer(cfg.encoder, device)
    print("=+="*10)

    # create transforms
    transforms = create_transforms(patch_transformer, cfg.aggregator.level, cfg.aggregator.patch_size)
    print("Encoder transforms:")
    print(transforms)
    print("=+="*10)

    print(f"Aggregator weights: {cfg.aggregator.pretrained_weights}")
    print("=+="*10)

    aggregator_sd = torch.load(cfg.aggregator.pretrained_weights, map_location="cpu")

    start_idx = list(aggregator_sd.keys()).index("vit_region.cls_token")
    end_idx = list(aggregator_sd.keys()).index("vit_region.norm.bias")
    region_transformer_sd = {
        k: v
        for i, (k, v) in enumerate(aggregator_sd.items())
        if i >= start_idx and i <= end_idx
    }
    region_transformer = get_region_transformer(
        state_dict=region_transformer_sd,
        input_embed_dim=patch_transformer.features_dim,
        region_size=cfg.aggregator.region_size,
        patch_size=cfg.aggregator.patch_size,
        mask_attn=cfg.aggregator.mask_attn,
        device=device,
        verbose=True,
    )
    print("=+="*10)

    start_idx = list(aggregator_sd.keys()).index("global_phi.0.weight")
    end_idx = list(aggregator_sd.keys()).index("global_rho.0.bias")
    case_transformer_sd = {
        k: v
        for i, (k, v) in enumerate(aggregator_sd.items())
        if i >= start_idx and i <= end_idx
    }
    case_transformer = get_case_transformer(state_dict=case_transformer_sd, device=device)
    print("=+="*10)

    custom_cmap = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.jet)

    mask_patch, mask_token = None, None
    if mask_attention:
        mask_patch, mask_token = generate_masks(
            wsi_name,
            wsi_path,
            mask_path,
            coordinates_dir,
            region_size=cfg.aggregator.region_size,
            patch_size=cfg.aggregator.patch_size,
            token_size=patch_transformer.token_size,
            spacing=tiling_config.params.spacing,
            downsample=1,
            tissue_pixel_value=tiling_config.seg_params.tissue_pixel_value,
        )

    #########################
    # REGION-LEVEL HEATMAPS #
    #########################

    print(f"Computing region-level heatmaps...")

    attention_dir = get_region_level_heatmaps(
        wsi_path=wsi_path,
        coordinates_dir=coordinates_dir,
        patch_transformer=patch_transformer,
        region_transformer=region_transformer,
        patch_size=cfg.aggregator.patch_size,
        transforms=transforms,
        output_dir=output_dir,
        downscale=cfg.downscale,
        granular=cfg.granularity.region,
        offset=cfg.granularity.offset.region,
        segmentation_mask_path=mask_path,
        segmentation_parameters=tiling_config.seg_params,
        spacing=tiling_config.params.spacing,
        tissue_pixel_value=tiling_config.seg_params.tissue_pixel_value,
        patch_attn_mask=mask_patch,
        token_attn_mask=mask_token,
        compute_patch_attention=False,
        patch_device=device,
        region_device=device,
        verbose=True,
    )

    attention_dir = Path("/data/temporary/clement/code/hipt/output/attention-maps/jthnfzot/2025-08-06_23_00/3da443b54bdf/attention/region/2048")
    stitched_heatmap_dir = stitch_slide_heatmaps(
        wsi_path,
        attention_dir,
        output_dir,
        name="region",
        spacing=tiling_config.params.spacing,
        tolerance=tiling_config.params.tolerance,
        patch_size=cfg.aggregator.patch_size,
        segmentation_parameters=tiling_config.seg_params,
        downscale=cfg.downscale,
        cmap=custom_cmap,
        segmentation_mask_path=mask_path,
        opacity=cfg.opacity,
        threshold=cfg.threshold,
        restrict_to_tissue=cfg.restrict_to_tissue,
        verbose=True,
    )

    if cfg.display:
        display_stitched_heatmaps(
            wsi_path,
            stitched_heatmap_dir,
            output_dir,
            name="region",
            display_patching=True,
            cmap=custom_cmap,
            coordinates_dir=coordinates_dir,
            downsample=tiling_config.seg_params.downsample,
            font_fp=cfg.font_fp,
        )

    ########################
    # SLIDE-LEVEL HEATMAPS #
    ########################

    print(f"Computing slide-level heatmaps...")

    attention_dir = get_case_level_heatmaps(
        wsi_path,
        coordinates_dir,
        patch_transformer,
        region_transformer,
        case_transformer,
        cfg.aggregator.patch_size,
        transforms,
        output_dir,
        downscale=cfg.downscale,
        threshold=cfg.threshold,
        highlight=cfg.highlight,
        cmap=custom_cmap,
        granular=cfg.granularity.slide,
        offset=cfg.granularity.offset.slide,
        patch_attn_mask=mask_patch,
        token_attn_mask=mask_token,
        patch_device=device,
        region_device=device,
        slide_device=device,
        verbose=True,
    )

    stitched_heatmap_dir = stitch_slide_heatmaps(
        wsi_path,
        attention_dir,
        output_dir,
        name="wsi",
        spacing=tiling_config.params.spacing,
        tolerance=tiling_config.params.tolerance,
        patch_size=cfg.aggregator.patch_size,
        segmentation_parameters=tiling_config.seg_params,
        downscale=cfg.downscale,
        cmap=custom_cmap,
        segmentation_mask_path=mask_path,
        restrict_to_tissue=cfg.restrict_to_tissue,
        opacity=cfg.opacity,
        verbose=True,
    )

    if cfg.display:
        display_stitched_heatmaps(
            wsi_path,
            stitched_heatmap_dir,
            output_dir,
            fname="wsi",
            display_patching=True,
            draw_grid=False,
            cmap=custom_cmap,
            coordinates_dir=coordinates_dir,
            downsample=tiling_config.seg_params.downsample,
            font_fp=cfg.font_fp,
        )

    ####################
    # FACTORIZED HEATMAPS #
    ####################

    print(f"Computing factorized heatmaps (gamma={cfg.gamma})")

    heatmap_dir = get_factorized_heatmaps(
        wsi_path,
        coordinates_dir,
        patch_transformer,
        region_transformer,
        case_transformer,
        cfg.aggregator.patch_size,
        transforms,
        cfg.aggregator.level,
        output_dir,
        gamma=cfg.gamma,
        downscale=cfg.downscale,
        threshold=cfg.threshold,
        cmap=custom_cmap,
        granularity=cfg.granularity,
        segmentation_mask_path=mask_path,
        spacing=tiling_config.params.spacing,
        downsample=tiling_config.seg_params.downsample,
        tissue_pixel_value=tiling_config.seg_params.tissue_pixel_value,
        patch_attn_mask=mask_patch,
        token_attn_mask=mask_token,
        compute_patch_attention=False,
        patch_device=device,
        region_device=device,
        slide_device=device,
        verbose=True,
    )

    heatmap_dir_reg = heatmap_dir / "regular"
    stitched_heatmap_dir = stitch_slide_heatmaps(
        wsi_path,
        heatmap_dir_reg,
        output_dir,
        name="factorized",
        spacing=tiling_config.params.spacing,
        tolerance=tiling_config.params.tolerance,
        segmentation_parameters=tiling_config.seg_params,
        downscale=cfg.downscale,
        cmap=custom_cmap,
        segmentation_mask_path=mask_path,
        restrict_to_tissue=cfg.restrict_to_tissue,
        opacity=cfg.opacity,
        verbose=True,
    )

    if cfg.display:
        display_stitched_heatmaps(
            wsi_path,
            stitched_heatmap_dir,
            output_dir,
            fname="factorized",
            display_patching=True,
            draw_grid=False,
            cmap=custom_cmap,
            coordinates_dir=coordinates_dir,
            downsample=tiling_config.seg_params.downsample,
            font_fp=cfg.font_fp,
        )


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)