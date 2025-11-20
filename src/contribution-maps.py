import tqdm
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig

from src.utils import setup
from src.interpretability import (
    cmap_map,
    get_config_from_path,
    get_patch_transformer,
    get_region_transformer,
    get_slide_transformer,
    get_classifier,
    clip_contributions,
    normalize_to_unit,
    plot_distribution,
    get_tile_contribution_scores,
    get_region_contribution_scores,
    stitch_contribution_scores,
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
    output_dir = Path(cfg.output_dir)

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
    slide_transformer_sd = {
        k: v
        for i, (k, v) in enumerate(aggregator_sd.items())
        if i >= start_idx and i <= end_idx
    }
    slide_transformer = get_slide_transformer(state_dict=slide_transformer_sd, device=device)
    print("=+="*10)

    start_idx = list(aggregator_sd.keys()).index("classifier.weight")
    end_idx = list(aggregator_sd.keys()).index("classifier.bias")
    classifier_sd = {
        k: v
        for i, (k, v) in enumerate(aggregator_sd.items())
        if i >= start_idx and i <= end_idx
    }
    classifier = get_classifier(
        input_dim=cfg.model.embed_dim_slide,
        num_classes=cfg.aggregator.num_classes,
        state_dict=classifier_sd,
        device=device
    )
    print("=+="*10)

    custom_cmap = plt.get_cmap("jet")
    custom_cmap.set_bad(color=(0,0,0,0))

    # hyperparameters
    k = 10
    binarize = False    # whether to binarize contributions before stitching
    clip = True         # clip contributions at 95th percentile

    #########################
    #       TILE BANK       #
    #########################

    # # need to enable masking
    # region_transformer = get_region_transformer(
    #     state_dict=region_transformer_sd,
    #     input_embed_dim=patch_transformer.features_dim,
    #     region_size=cfg.aggregator.region_size,
    #     patch_size=cfg.aggregator.patch_size,
    #     mask_attn=True,
    #     device=device,
    #     verbose=False,
    # )

    # df = pd.read_csv(cfg.csv)
    # wsi_paths = [Path(x) for x in df.wsi_path.tolist()]
    # mask_paths = [Path(x) for x in df.mask_path.tolist()]
    # feature_paths = [Path(x) for x in df.feature_path.tolist()]

    # wsis = []
    # contributors, protectors = [], []
    # contributions, protections = [], []
    # with tqdm.tqdm(
    #     zip(wsi_paths, mask_paths, feature_paths),
    #     desc="Building tile bank",
    #     unit=" wsi",
    #     leave=True,
    # ) as t:
    #     for wsi_path, mask_path, feature_path in t:

    #         wsi_name = wsi_path.stem.replace(" ", "_")
    #         output_subdir = output_dir / wsi_name
    #         output_subdir.mkdir(exist_ok=True, parents=True)

    #         root_dir = feature_path.parents[1]
    #         tiling_config_file = root_dir / "config.yaml"
    #         tiling_config = get_config_from_path(tiling_config_file)
    #         coordinates_dir = root_dir / "coordinates"

    #         tile_contributions = get_tile_contribution_scores(
    #             wsi_path=wsi_path,
    #             feature_path=feature_path,
    #             region_transformer=region_transformer,
    #             slide_transformer=slide_transformer,
    #             classifier=classifier,
    #         ) # (num_regions, num_tiles)

    #         # k = int(round(thresh * tile_contributions.numel(),0))
    #         tile_contributions_flat = tile_contributions.flatten()

    #         # plot distribution of tile contributions
    #         plot_distribution(
    #             contributions=tile_contributions_flat.cpu().numpy(),
    #             save_dir=output_subdir,
    #             suffix="tile-raw",
    #         )

    #         topk_values, topk_indices = torch.topk(tile_contributions_flat, k=k, largest=True)
    #         bottomk_values, bottomk_indices = torch.topk(tile_contributions_flat, k=k, largest=False)
    #         contributors.extend(topk_indices.cpu().numpy().tolist())
    #         protectors.extend(bottomk_indices.cpu().numpy().tolist())
    #         contributions.extend(topk_values.cpu().numpy().tolist())
    #         protections.extend(bottomk_values.cpu().numpy().tolist())
    #         wsis.extend([wsi_path]*k)

    #         tile_contributions = tile_contributions.cpu().numpy()
    #         if clip:
    #             tile_contributions = clip_contributions(tile_contributions, lower=5, upper=95)
    #             plot_distribution(
    #                 contributions=tile_contributions.flatten(),
    #                 save_dir=output_subdir,
    #                 suffix="tile-clipped",
    #             )
    #         tile_contributions = normalize_to_unit(tile_contributions)
    #         plot_distribution(
    #             contributions=tile_contributions.flatten(),
    #             save_dir=output_subdir,
    #             suffix="tile-normalized",
    #         )

    #         # reshape to (num_regions, npatch, npatch)
    #         num_regions, num_tiles = tile_contributions.shape
    #         npatch = int(np.sqrt(num_tiles))
    #         _tile_contributions = tile_contributions.reshape(num_regions, npatch, npatch)    # (num_regions, npatch, npatch)

    #         # expand each (npatch,npatch) score to a (npatch*scale, npatch*scale) tile
    #         scale = int(cfg.aggregator.patch_size / cfg.downscale)
    #         _tile_contributions = np.repeat(_tile_contributions, scale, axis=1)
    #         _tile_contributions = np.repeat(_tile_contributions, scale, axis=2)

    #         stitched_heatmap_dir = stitch_contribution_scores(
    #             wsi_path=wsi_path,
    #             contributions=_tile_contributions,
    #             coordinates_dir=coordinates_dir,
    #             output_dir=output_subdir,
    #             name="tile",
    #             spacing=tiling_config.params.spacing,
    #             tolerance=tiling_config.params.tolerance,
    #             patch_size=cfg.aggregator.patch_size,
    #             segmentation_parameters=tiling_config.seg_params,
    #             downscale=cfg.downscale,
    #             cmap=custom_cmap,
    #             segmentation_mask_path=mask_path,
    #             smoothing=True,
    #             binarize=binarize,
    #             opacity=cfg.opacity,
    #             threshold=True,
    #             restrict_to_tissue=cfg.restrict_to_tissue,
    #             verbose=True,
    #         )

    #         torch.cuda.empty_cache()
    #         break

    # df = pd.DataFrame({
    #     "wsi_path": wsis,
    #     "contributor_index": contributors,
    #     "contribution_score": contributions,
    #     "protector_index": protectors,
    #     "protection_score": protections,
    # })
    # df.to_csv(output_dir / "tile-bank.csv", index=False)

    # del region_transformer
    # torch.cuda.empty_cache()

    #########################
    #      REGION BANK      #
    #########################

    # hyperparameters
    k = 3
    binarize = False    # whether to binarize contributions before stitching
    clip = True         # clip contributions at 95th percentile

    # need to reset masking
    region_transformer = get_region_transformer(
        state_dict=region_transformer_sd,
        input_embed_dim=patch_transformer.features_dim,
        region_size=cfg.aggregator.region_size,
        patch_size=cfg.aggregator.patch_size,
        mask_attn=cfg.aggregator.mask_attn,
        device=device,
        verbose=False,
    )

    df = pd.read_csv(cfg.csv)
    wsi_paths = [Path(x) for x in df.wsi_path.tolist()]
    mask_paths = [Path(x) for x in df.mask_path.tolist()]
    feature_paths = [Path(x) for x in df.feature_path.tolist()]

    wsis = []
    contributors, protectors = [], []
    contributions, protections = [], []
    with tqdm.tqdm(
        zip(wsi_paths, mask_paths, feature_paths),
        desc="Building region bank",
        unit=" wsi",
        leave=True,
    ) as t:
        for wsi_path, mask_path, feature_path in t:

            wsi_name = wsi_path.stem.replace(" ", "_")
            output_subdir = output_dir / wsi_name
            output_subdir.mkdir(exist_ok=True, parents=True)

            root_dir = feature_path.parents[1]
            tiling_config_file = root_dir / "config.yaml"
            tiling_config = get_config_from_path(tiling_config_file)
            coordinates_dir = root_dir / "coordinates"

            region_contributions = get_region_contribution_scores(
                wsi_path=wsi_path,
                feature_path=feature_path,
                region_transformer=region_transformer,
                slide_transformer=slide_transformer,
                classifier=classifier,
            ) # (num_regions,)

            # plot distribution of region contributions
            plot_distribution(
                contributions=region_contributions.cpu().numpy(),
                save_dir=output_subdir,
                suffix="region-raw",
            )

            topk_values, topk_indices = torch.topk(region_contributions, k=k, largest=True)
            bottomk_values, bottomk_indices = torch.topk(region_contributions, k=k, largest=False)
            contributors.extend(topk_indices.cpu().numpy().tolist())
            protectors.extend(bottomk_indices.cpu().numpy().tolist())
            contributions.extend(topk_values.cpu().numpy().tolist())
            protections.extend(bottomk_values.cpu().numpy().tolist())
            wsis.extend([wsi_path]*k)

            region_contributions = region_contributions.cpu().numpy()
            if clip:
                region_contributions = clip_contributions(region_contributions, lower=5, upper=95)
                plot_distribution(
                    contributions=region_contributions,
                    save_dir=output_subdir,
                    suffix="region-clipped",
                )
            region_contributions = normalize_to_unit(region_contributions)
            plot_distribution(
                contributions=region_contributions,
                save_dir=output_subdir,
                suffix="region-normalized",
            )

            # reshape to (num_regions, region_size, region_size)
            num_regions = len(region_contributions)
            _region_contributions = region_contributions.reshape(-1, 1, 1)             # (num_regions, 1, 1)
            scale = int(cfg.aggregator.region_size / cfg.downscale)

            # expand each (1,1) score to a (scale, scale) region
            _region_contributions = np.tile(_region_contributions, (1, scale, scale))   # (num_regions, region_size, region_size) if downscale=1

            stitched_heatmap_dir = stitch_contribution_scores(
                wsi_path=wsi_path,
                contributions=_region_contributions,
                coordinates_dir=coordinates_dir,
                output_dir=output_subdir,
                name="region",
                spacing=tiling_config.params.spacing,
                tolerance=tiling_config.params.tolerance,
                patch_size=cfg.aggregator.patch_size,
                segmentation_parameters=tiling_config.seg_params,
                downscale=cfg.downscale,
                cmap=custom_cmap,
                segmentation_mask_path=mask_path,
                smoothing=True,
                binarize=binarize,
                opacity=cfg.opacity,
                threshold=False,
                restrict_to_tissue=cfg.restrict_to_tissue,
                verbose=True,
            )

            torch.cuda.empty_cache()

    df = pd.DataFrame({
        "wsi_path": wsis,
        "contributor_index": contributors,
        "contribution_score": contributions,
        "protector_index": protectors,
        "protection_score": protections,
    })
    df.to_csv(output_dir / "region-bank.csv", index=False)


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)