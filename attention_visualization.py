import os
import tqdm
import wandb
import hydra
import torch
import random
import datetime
import matplotlib
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from collections import defaultdict

import source.distributed as distributed
from source.dataset import SlideFilepathsDataset
from source.utils import initialize_wandb
from source.attention_visualization_utils import (
    cmap_map,
    get_encoder,
    get_region_model,
    get_slide_model,
    generate_masks,
    create_patch_heatmaps_indiv,
    create_patch_heatmaps_concat,
    create_region_heatmaps_indiv,
    create_region_heatmaps_concat,
    get_slide_heatmaps_patch_level,
    get_slide_heatmaps_region_level,
    get_slide_heatmaps_slide_level,
    get_slide_blended_heatmaps,
    stitch_slide_heatmaps,
    display_stitched_heatmaps,
    create_transforms,
)


@hydra.main(version_base="1.2.0", config_path="config/heatmaps", config_name="default")
def main(cfg: DictConfig):

    distributed.setup_distributed()
    distributed.fix_random_seeds(cfg.seed)

    if distributed.is_main_process():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        # set up wandb
        if cfg.wandb.enable:
            key = os.environ.get("WANDB_API_KEY")
            wandb_run = initialize_wandb(cfg, key=key)
            wandb_run.define_metric("processed", summary="max")
            run_id = wandb_run.id
    else:
        run_id = ""

    if distributed.is_enabled_and_multiple_gpus():
        obj = [run_id]
        torch.distributed.broadcast_object_list(
            obj, 0, device=torch.device(f"cuda:{distributed.get_global_rank()}")
        )
        run_id = obj[0]

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    if distributed.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    mask_attention = (cfg.mask_attn_patch is True) or (cfg.mask_attn_region is True)

    # seed everything to ensure reproducible heatmaps
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{distributed.get_global_rank()}")

    if distributed.is_main_process():
        print("=+="*10)
    encoder = get_encoder(cfg.encoder, device)
    if distributed.is_main_process():
        print("=+="*10)

    # create transforms
    transforms = create_transforms(encoder, cfg.level)
    if distributed.is_main_process():
        print(f"Encoder transforms: {transforms}")
        print("=+="*10)

    if distributed.is_main_process():
        print(f"Aggregator weights: {cfg.aggregator.pretrained_weights}...")
        print("=+="*10)

    aggregator_sd = torch.load(cfg.aggregator.pretrained_weights, map_location="cpu")
    
    start_idx = list(aggregator_sd.keys()).index("vit_region.cls_token")
    end_idx = list(aggregator_sd.keys()).index("vit_region.norm.bias")
    region_sd = {
        k: v
        for i, (k, v) in enumerate(aggregator_sd.items())
        if i >= start_idx and i <= end_idx
    }
    region_model = get_region_model(
        region_sd,
        input_embed_dim=encoder.features_dim,
        region_size=cfg.aggregator.region_size,
        mask_attn=cfg.aggregator.mask_attn,
        img_size_pretrained=cfg.aggregator.img_size_pretrained,
        device=device,
        verbose=distributed.is_main_process(),
    )
    if distributed.is_main_process():
        print("=+="*10)

    start_idx = list(aggregator_sd.keys()).index("global_phi.0.weight")
    end_idx = list(aggregator_sd.keys()).index("global_rho.0.bias")
    slide_sd = {
        k: v
        for i, (k, v) in enumerate(aggregator_sd.items())
        if i >= start_idx and i <= end_idx
    }
    slide_model = get_slide_model(slide_sd, device=device)
    if distributed.is_main_process():
        print("=+="*10)

    custom_cmap = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.jet)
    # custom_cmap = cmap_map(lambda x: x / 2 + 0.5, matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","white","lime"]))

    if cfg.patch_fp and distributed.is_main_process():
        patch = Image.open(cfg.patch_fp)

        print(f"Computing indiviudal patch-level attention heatmaps")
        output_dir_patch = Path(output_dir, "patch_indiv")
        output_dir_patch.mkdir(exist_ok=True, parents=True)
        create_patch_heatmaps_indiv(
            patch,
            encoder,
            output_dir_patch,
            threshold=0.5,
            alpha=0.5,
            cmap=custom_cmap,
            granular=cfg.smoothing.patch,
            offset=cfg.smoothing.offset.patch,
            downscale=cfg.downscale,
            patch_device=device,
        )
        print("done!")

        print(f"Computing concatenated patch-level attention heatmaps")
        output_dir_patch_concat = Path(output_dir, "patch_concat")
        output_dir_patch_concat.mkdir(exist_ok=True, parents=True)
        create_patch_heatmaps_concat(
            patch,
            encoder,
            output_dir_patch_concat,
            threshold=0.5,
            alpha=0.5,
            cmap=custom_cmap,
            granular=cfg.smoothing.patch,
            offset=cfg.smoothing.offset.patch,
            downscale=cfg.downscale,
            patch_device=device,
        )
        print("done!")

    if cfg.region_fp and distributed.is_main_process():
        region = Image.open(cfg.region_fp)

        print(f"Computing individual region-level attention heatmaps")
        output_dir_region = Path(output_dir, "region_indiv")
        output_dir_region.mkdir(exist_ok=True, parents=True)
        create_region_heatmaps_indiv(
            region,
            encoder,
            region_model,
            transforms,
            output_dir_region,
            downscale=cfg.downscale,
            threshold=0.5,
            alpha=0.5,
            cmap=custom_cmap,
            granular=cfg.smoothing.region,
            offset=cfg.smoothing.offset.region,
            patch_device=device,
            region_device=device,
        )
        print("done!")

        print(f"Computing concatenated region-level attention heatmaps")
        output_dir_region_concat = Path(output_dir, "region_concat")
        output_dir_region_concat.mkdir(exist_ok=True, parents=True)
        create_region_heatmaps_concat(
            region,
            encoder,
            region_model,
            transforms,
            output_dir_region_concat,
            downscale=cfg.downscale,
            alpha=0.5,
            cmap=custom_cmap,
            granular=cfg.smoothing.region,
            offset=cfg.smoothing.offset.region,
            patch_device=device,
            region_device=device,
        )
        print("done!")

    if cfg.slide_fp and distributed.is_main_process():

        slide_path = Path(cfg.slide_fp)
        slide_id = slide_path.stem.replace(" ", "_")
        region_dir = Path(cfg.region_dir)
        coordinates_dir = Path(cfg.coordinates_dir)

        output_dir_slide = Path(output_dir, slide_id)

        mask_p, mask_mp = None, None
        print(f"mask_attention: {mask_attention}")
        print(f"restrict_to_tissue: {cfg.restrict_to_tissue}")
        if mask_attention:
            if cfg.attention_masks_dir is not None:
                att_mask_fp = Path(cfg.attention_masks_dir, f"{slide_id}.npy")
                attn_mask = np.load(att_mask_fp) # (M, npatch**2, nminipatch**2)
                pct = torch.Tensor(attn_mask)
                if cfg.mask_attn_region:
                    # infer patch-level mask
                    pct_patch = torch.sum(pct, axis=-1) / pct[0].numel()
                    mask_p = (pct_patch > 0.0).int()
                    # add the [CLS] token to the mask
                    cls_token = mask_p.new_ones((mask_p.size(0),1))
                    mask_p = torch.cat((cls_token, mask_p), dim=1)  # [M, num_patches+1]
                if cfg.mask_attn_patch:
                    # infer mini-patch-level mask
                    mask_mp = (pct > 0.0).int()  # (M, num_patches, nminipatch**2)
                    # add the [CLS] token to the mask
                    cls_token = mask_mp.new_ones((mask_mp.size(0),mask_mp.size(1),1))
                    mask_mp = torch.cat((cls_token, mask_mp), dim=2)  # [M, num_patches, nminipatch**2+1]
            else:
                mask_p, mask_mp = generate_masks(
                    slide_id,
                    slide_path,
                    cfg.segmentation_mask_fp,
                    region_dir,
                    region_size=cfg.region_size,
                    spacing=cfg.spacing,
                    downsample=1,
                    tissue_pixel_value=cfg.tissue_pixel_value,
                )
                if not cfg.mask_attn_patch:
                    mask_mp = None
                if not cfg.mask_attn_region:
                    mask_p = None

        ########################
        # PATCH-LEVEL HEATMAPS #
        ########################

        print("Computing patch-level heatmaps")

        heatmap_dir, num_head = get_slide_heatmaps_patch_level(
            slide_path,
            coordinates_dir,
            encoder,
            region_model,
            transforms,
            output_dir_slide,
            downscale=cfg.downscale,
            cmap=custom_cmap,
            threshold=cfg.threshold,
            highlight=cfg.highlight,
            granular=cfg.smoothing.patch,
            offset=cfg.smoothing.offset.patch,
            segmentation_mask_path=cfg.segmentation_mask_fp,
            spacing=cfg.spacing,
            downsample=1,
            background_pixel_value=cfg.background_pixel_value,
            tissue_pixel_value=cfg.tissue_pixel_value,
            patch_attn_mask=mask_p,
            mini_patch_attn_mask=mask_mp,
            patch_device=device,
            region_device=device,
        )

        stitched_heatmap_dir = stitch_slide_heatmaps(
            slide_path,
            heatmap_dir,
            output_dir_slide,
            num_head,
            name="patch",
            spacing=cfg.spacing,
            downsample=cfg.downsample,
            downscale=cfg.downscale,
            cmap=custom_cmap,
            restrict_to_tissue=cfg.restrict_to_tissue,
        )

        heatmap_dir_thresh = heatmap_dir / "thresholded"
        if heatmap_dir_thresh.exists():
            stitched_heatmap_thresh_dir = stitch_slide_heatmaps(
                slide_path,
                heatmap_dir_thresh,
                output_dir_slide,
                num_head,
                name="patch",
                suffix="thresholded",
                spacing=cfg.spacing,
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                cmap=custom_cmap,
                restrict_to_tissue=cfg.restrict_to_tissue,
            )

        heatmap_dir_highlight = heatmap_dir / "highlighted"
        if heatmap_dir_highlight.exists():
            stitched_heatmap_highlight_dir = stitch_slide_heatmaps(
                slide_path,
                heatmap_dir_highlight,
                output_dir_slide,
                num_head,
                name="patch",
                suffix="highlighted",
                spacing=cfg.spacing,
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                cmap=custom_cmap,
                highlight=(cfg.highlight != None),
                opacity=cfg.opacity,
                restrict_to_tissue=cfg.restrict_to_tissue,
            )

        if cfg.display:
            display_stitched_heatmaps(
                slide_path,
                stitched_heatmap_dir,
                output_dir_slide,
                name="patch",
                display_patching=True,
                cmap=custom_cmap,
                coordinates_dir=coordinates_dir,
                downsample=cfg.downsample,
                font_fp=cfg.font_fp,
                run_id=run_id,
            )
            if heatmap_dir_thresh.exists():
                display_stitched_heatmaps(
                    slide_path,
                    stitched_heatmap_thresh_dir,
                    output_dir_slide,
                    name="patch_thresh",
                    display_patching=True,
                    cmap=custom_cmap,
                    coordinates_dir=coordinates_dir,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )
            if heatmap_dir_highlight.exists():
                display_stitched_heatmaps(
                    slide_path,
                    stitched_heatmap_highlight_dir,
                    output_dir_slide,
                    name="patch_highlight",
                    display_patching=True,
                    cmap=custom_cmap,
                    coordinates_dir=coordinates_dir,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )

        #########################
        # REGION-LEVEL HEATMAPS #
        #########################

        print("Computing region-level heatmaps")

        heatmap_dir, num_head = get_slide_heatmaps_region_level(
            slide_path,
            coordinates_dir,
            encoder,
            region_model,
            transforms,
            output_dir_slide,
            downscale=cfg.downscale,
            cmap=custom_cmap,
            save_to_disk=True,
            threshold=cfg.threshold,
            highlight=cfg.highlight,
            granular=cfg.smoothing.region,
            offset=cfg.smoothing.offset.region,
            gaussian_smoothing=cfg.gaussian_smoothing,
            segmentation_mask_path=cfg.segmentation_mask_fp,
            spacing=cfg.spacing,
            downsample=1,
            background_pixel_value=cfg.background_pixel_value,
            tissue_pixel_value=cfg.tissue_pixel_value,
            restrict_to_tissue=cfg.restrict_to_tissue,
            patch_attn_mask=mask_p,
            mini_patch_attn_mask=mask_mp,
            patch_device=device,
            region_device=device,
        )

        stitched_heatmap_dir = stitch_slide_heatmaps(
            slide_path,
            heatmap_dir,
            output_dir_slide,
            num_head,
            name="region",
            spacing=cfg.spacing,
            downsample=cfg.downsample,
            downscale=cfg.downscale,
            cmap=custom_cmap,
            restrict_to_tissue=cfg.restrict_to_tissue,
        )

        heatmap_dir_thresh = heatmap_dir / "thresholded"
        if heatmap_dir_thresh.exists():
            stitched_heatmap_thresh_dir = stitch_slide_heatmaps(
                slide_path,
                heatmap_dir_thresh,
                output_dir_slide,
                num_head,
                name="region",
                suffix="thresholded",
                spacing=cfg.spacing,
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                cmap=custom_cmap,
                restrict_to_tissue=cfg.restrict_to_tissue,
            )

        heatmap_dir_highlight = heatmap_dir / "highlighted"
        if heatmap_dir_highlight.exists():
            stitched_heatmap_highlight_dir = stitch_slide_heatmaps(
                slide_path,
                heatmap_dir_highlight,
                output_dir_slide,
                num_head,
                name="region",
                suffix="highlighted",
                spacing=cfg.spacing,
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                cmap=custom_cmap,
                highlight=(cfg.highlight != None),
                opacity=cfg.opacity,
                restrict_to_tissue=cfg.restrict_to_tissue,
            )

        if cfg.display:
            display_stitched_heatmaps(
                slide_path,
                stitched_heatmap_dir,
                output_dir_slide,
                name="region",
                display_patching=True,
                cmap=custom_cmap,
                coordinates_dir=coordinates_dir,
                downsample=cfg.downsample,
                font_fp=cfg.font_fp,
                run_id=run_id,
            )
            if heatmap_dir_thresh.exists():
                display_stitched_heatmaps(
                    slide_path,
                    stitched_heatmap_thresh_dir,
                    output_dir_slide,
                    name="region_thresh",
                    display_patching=True,
                    cmap=custom_cmap,
                    coordinates_dir=coordinates_dir,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )
            if heatmap_dir_highlight.exists():
                display_stitched_heatmaps(
                    slide_path,
                    stitched_heatmap_highlight_dir,
                    output_dir_slide,
                    name="region_highlight",
                    display_patching=True,
                    cmap=custom_cmap,
                    coordinates_dir=coordinates_dir,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )

        ##################################
        # HIERARCHICAL HEATMAPS (REGION) #
        ##################################

        # hms, hms_thresh, coords = get_slide_hierarchical_heatmaps_region(
        #     slide_path,
        #     coordinates_dir,
        #     encoder,
        #     region_model,
        #     transforms,
        #     region_dir,
        #     output_dir_slide,
        #     downscale=cfg.downscale,
        #     cmap=custom_cmap,
        #     threshold=None,
        #     save_to_disk=True,
        #     granular=cfg.smoothing.region,
        #     offset=cfg.smoothing.offset.region,
        #     patch_device=device,
        #     region_device=device,
        # )

        # stitched_hms = defaultdict(list)
        # for rhead_num, hm_dict in hms.items():
        #     coords_dict = coords[rhead_num]
        #     for phead_num, heatmaps in hm_dict.items():
        #         coordinates = coords_dict[phead_num]
        #         stitched_hm = stitch_slide_heatmaps(
        #             slide_path,
        #             heatmaps,
        #             coordinates,
        #             output_dir_slide,
        #             fname=f"rhead_{rhead_num}_phead_{phead_num}",
        #             spacing=cfg.spacing,
        #             downsample=cfg.downsample,
        #             downscale=cfg.downscale,
        #             save_to_disk=True,
        #         )
        #         stitched_hms[rhead_num].append(stitched_hm)

        # stitched_hms_thresh = defaultdict(list)
        # for rhead_num, hm_dict in hms_thresh.items():
        #     coords_dict = coords[rhead_num]
        #     for phead_num, heatmaps in hm_dict.items():
        #         coordinates = coords_dict[phead_num]
        #         stitched_hm = stitch_slide_heatmaps(
        #             slide_path,
        #             heatmaps,
        #             coordinates,
        #             output_dir_slide,
        #             fname=f"rhead_{rhead_num}_phead_{phead_num}_thresh",
        #             spacing=cfg.spacing,
        #             downsample=cfg.downsample,
        #             downscale=cfg.downscale,
        #             save_to_disk=True,
        #         )
        #         stitched_hms_thresh[rhead_num].append(stitched_hm)

        # if cfg.display:
        #     for rhead_num, hms in stitched_hms.items():
        #         d = {
        #             f"Region Head {rhead_num} & Patch Head {phead}": hm
        #             for phead, hm in enumerate(hms)
        #         }
        #         display_stitched_heatmaps(
        #             slide_path,
        #             d,
        #             output_dir_slide,
        #             fname=f"rhead_{rhead_num}",
        #             display_patching=True,
        #             cmap=custom_cmap,
        #             coordinates_dir=coordinates_dir,
        #             downsample=cfg.downsample,
        #             font_fp=cfg.font_fp,
        #             run_id=run_id,
        #         )

        if cfg.slide_weights:

            ########################
            # SLIDE-LEVEL HEATMAPS #
            ########################

            print("Computing slide-level heatmaps")

            hms, thresh_hms, highlight_hms, coords = get_slide_heatmaps_slide_level(
                slide_path,
                coordinates_dir,
                encoder,
                region_model,
                slide_model,
                transforms,
                output_dir_slide,
                patch_size=256,
                downscale=cfg.downscale,
                threshold=cfg.threshold,
                highlight=cfg.highlight,
                cmap=custom_cmap,
                region_fmt="jpg",
                granular=cfg.smoothing.slide,
                offset=cfg.smoothing.offset.slide,
                spacing=cfg.spacing,
                gaussian_smoothing=cfg.gaussian_smoothing,
                gaussian_offset=cfg.gaussian_offset,
                patch_attn_mask=mask_p,
                mini_patch_attn_mask=mask_mp,
                patch_device=device,
                region_device=device,
                slide_device=device,
            )

            stitched_hms = {}
            stitched_hm = stitch_slide_heatmaps(
                slide_path,
                hms,
                coords,
                output_dir_slide,
                fname=f"wsi",
                spacing=cfg.spacing,
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                cmap=custom_cmap,
                save_to_disk=True,
                restrict_to_tissue=cfg.restrict_to_tissue,
                segmentation_mask_path=cfg.segmentation_mask_fp,
                tissue_pixel_value=cfg.tissue_pixel_value,
            )
            stitched_hms[f"slide-level"] = stitched_hm

            if len(thresh_hms) > 0:
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    thresh_hms,
                    coords,
                    output_dir_slide,
                    fname=f"wsi_thresh",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    save_to_disk=True,
                )
                stitched_hms[f"thresholded"] = stitched_hm

            if len(highlight_hms) > 0:
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    highlight_hms,
                    coords,
                    output_dir_slide,
                    fname=f"wsi_highlight",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    save_to_disk=True,
                    highlight=(cfg.highlight != None),
                    opacity=cfg.opacity,
                    restrict_to_tissue=cfg.restrict_to_tissue,
                    seg_params=cfg.seg_params,
                )
                stitched_hms[f"highlight"] = stitched_hm

            if cfg.display:
                display_stitched_heatmaps(
                    slide_path,
                    stitched_hms,
                    output_dir_slide,
                    fname=f"wsi",
                    display_patching=True,
                    draw_grid=False,
                    cmap=custom_cmap,
                    coordinates_dir=coordinates_dir,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )

            ####################
            # BLENDED HEATMAPS #
            ####################

            print("Computing factorized heatmaps")

            hms, thresh_hms, coords = get_slide_blended_heatmaps(
                slide_path,
                coordinates_dir,
                encoder,
                region_model,
                slide_model,
                transforms,
                cfg.level,
                output_dir_slide,
                gamma=cfg.gamma,
                patch_size=256,
                mini_patch_size=16,
                downscale=cfg.downscale,
                threshold=cfg.threshold,
                cmap=custom_cmap,
                smoothing=cfg.smoothing,
                segmentation_mask_path=cfg.segmentation_mask_fp,
                spacing=cfg.spacing,
                downsample=1,
                background_pixel_value=cfg.background_pixel_value,
                tissue_pixel_value=cfg.tissue_pixel_value,
                patch_attn_mask=mask_p,
                mini_patch_attn_mask=mask_mp,
                patch_device=device,
                region_device=device,
                slide_device=device,
            )

            stitched_hms = defaultdict(list)
            for rhead_num in range(region_model.num_heads):
                for phead_num in range(encoder.num_heads):
                    heatmaps = hms[(rhead_num,phead_num)]
                    coordinates = coords[(rhead_num,phead_num)]
                    stitched_hm = stitch_slide_heatmaps(
                        slide_path,
                        heatmaps,
                        coordinates,
                        output_dir_slide,
                        fname=f"rhead_{rhead_num}_phead_{phead_num}",
                        spacing=cfg.spacing,
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        cmap=custom_cmap,
                        save_to_disk=True,
                        restrict_to_tissue=cfg.restrict_to_tissue,
                        segmentation_mask_path=cfg.segmentation_mask_fp,
                        tissue_pixel_value=cfg.tissue_pixel_value,
                    )
                    stitched_hms[rhead_num].append(stitched_hm)

            if len(thresh_hms) > 0:
                stitched_hms_thresh = defaultdict(list)
                for rhead_num in range(region_model.num_heads):
                    for phead_num in range(encoder.num_heads):
                        heatmaps = thresh_hms[(rhead_num,phead_num)]
                        coordinates = coords[(rhead_num,phead_num)]
                        stitched_hm = stitch_slide_heatmaps(
                            slide_path,
                            heatmaps,
                            coordinates,
                            output_dir_slide,
                            fname=f"rhead_{rhead_num}_phead_{phead_num}_thresh",
                            spacing=cfg.spacing,
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            cmap=custom_cmap,
                            save_to_disk=True,
                        )
                        stitched_hms_thresh[rhead_num].append(stitched_hm)

            if cfg.display:
                for rhead_num, hms in stitched_hms.items():
                    d = {
                        f"Region Head {rhead_num} & Patch Head {phead}": hm
                        for phead, hm in enumerate(hms)
                    }
                    display_stitched_heatmaps(
                        slide_path,
                        d,
                        output_dir_slide,
                        fname=f"rhead_{rhead_num}",
                        display_patching=True,
                        cmap=custom_cmap,
                        coordinates_dir=coordinates_dir,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                        run_id=run_id,
                    )

    if cfg.slide_csv:

        df = pd.read_csv(cfg.slide_csv)
        dataset = SlideFilepathsDataset(df)

        if distributed.is_enabled_and_multiple_gpus():
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=cfg.num_workers,
            shuffle=False,
            drop_last=False,
        )

        with tqdm.tqdm(
            loader,
            desc="Attention Heatmap Generation",
            unit=" slide",
            ncols=80,
            position=0,
            leave=True,
            disable=not distributed.is_main_process(),
        ) as t1:
            for i, batch in enumerate(t1):

                _, slide_fp, seg_mask_path = batch
                slide_path = Path(slide_fp[0])
                if seg_mask_path != None:
                    seg_mask_path = Path(seg_mask_path[0])
                slide_id = slide_path.stem.replace(" ", "_")
                coordinates_dir = Path(cfg.coordinates_dir)

                output_dir_slide = Path(output_dir, slide_id)
                output_dir_slide.mkdir(exist_ok=True, parents=True)

                mask_p, mask_mp = None, None
                if mask_attention:
                    mask_p, mask_mp = generate_masks(
                        slide_id,
                        slide_path,
                        seg_mask_path,
                        coordinates_dir,
                        region_size=cfg.region_size,
                        spacing=cfg.spacing,
                        downsample=1,
                        tissue_pixel_value=cfg.tissue_pixel_value,
                    )

                ########################
                # PATCH-LEVEL HEATMAPS #
                ########################

                # if distributed.is_main_process():
                #     tqdm.tqdm.write(f"Computing patch-level heatmaps for {slide_id}")

                # heatmap_dir = get_slide_heatmaps_patch_level(
                #     slide_path,
                #     coordinates_dir,
                #     encoder,
                #     region_model,
                #     transforms,
                #     output_dir_slide,
                #     downscale=cfg.downscale,
                #     cmap=custom_cmap,
                #     threshold=cfg.threshold,
                #     highlight=cfg.highlight,
                #     granular=cfg.smoothing.patch,
                #     offset=cfg.smoothing.offset.patch,
                #     segmentation_mask_path=cfg.segmentation_mask_fp,
                #     spacing=cfg.spacing,
                #     downsample=cfg.downsample,,
                #     background_pixel_value=cfg.background_pixel_value,
                #     tissue_pixel_value=cfg.tissue_pixel_value,
                #     patch_attn_mask=mask_p,
                #     mini_patch_attn_mask=mask_mp,
                #     patch_device=device,
                #     region_device=device,
                #     main_process=distributed.is_main_process(),
                # )

                # heatmap_dir_reg = heatmap_dir / "regular"
                # stitched_heatmap_dir = stitch_slide_heatmaps(
                #     slide_path,
                #     heatmap_dir_reg,
                #     output_dir_slide,
                #     name="patch",
                #     spacing=cfg.spacing,
                #     downsample=cfg.downsample,
                #     downscale=cfg.downscale,
                #     cmap=custom_cmap,
                #     restrict_to_tissue=cfg.restrict_to_tissue,
                # )

                # heatmap_dir_thresh = heatmap_dir / "thresholded"
                # if heatmap_dir_thresh.exists():
                #     stitched_heatmap_thresh_dir = stitch_slide_heatmaps(
                #         slide_path,
                #         heatmap_dir_thresh,
                #         output_dir_slide,
                #         name="patch",
                #         suffix="thresholded",
                #         spacing=cfg.spacing,
                #         downsample=cfg.downsample,
                #         downscale=cfg.downscale,
                #         cmap=custom_cmap,
                #         restrict_to_tissue=cfg.restrict_to_tissue,
                #     )

                # heatmap_dir_highlight = heatmap_dir / "highlighted"
                # if heatmap_dir_highlight.exists():
                #     stitched_heatmap_highlight_dir = stitch_slide_heatmaps(
                #         slide_path,
                #         heatmap_dir_highlight,
                #         output_dir_slide,
                #         name="patch",
                #         suffix="highlighted",
                #         spacing=cfg.spacing,
                #         downsample=cfg.downsample,
                #         downscale=cfg.downscale,
                #         cmap=custom_cmap,
                #         highlight=(cfg.highlight != None),
                #         opacity=cfg.opacity,
                #         restrict_to_tissue=cfg.restrict_to_tissue,
                #     )

                # if cfg.display:
                #     display_stitched_heatmaps(
                #         slide_path,
                #         stitched_heatmap_dir,
                #         output_dir_slide,
                #         name="patch",
                #         display_patching=True,
                #         cmap=custom_cmap,
                #         coordinates_dir=coordinates_dir,
                #         downsample=cfg.downsample,
                #         font_fp=cfg.font_fp,
                #         run_id=run_id,
                #     )
                #     if heatmap_dir_thresh.exists():
                #         display_stitched_heatmaps(
                #             slide_path,
                #             stitched_heatmap_thresh_dir,
                #             output_dir_slide,
                #             name="patch_thresh",
                #             display_patching=True,
                #             cmap=custom_cmap,
                #             coordinates_dir=coordinates_dir,
                #             downsample=cfg.downsample,
                #             font_fp=cfg.font_fp,
                #             run_id=run_id,
                #         )
                #     if heatmap_dir_highlight.exists():
                #         display_stitched_heatmaps(
                #             slide_path,
                #             stitched_heatmap_highlight_dir,
                #             output_dir_slide,
                #             name="patch_highlight",
                #             display_patching=True,
                #             cmap=custom_cmap,
                #             coordinates_dir=coordinates_dir,
                #             downsample=cfg.downsample,
                #             font_fp=cfg.font_fp,
                #             run_id=run_id,
                #         )

                #########################
                # REGION-LEVEL HEATMAPS #
                #########################

                if distributed.is_main_process():
                    tqdm.tqdm.write(f"Computing region-level heatmaps for {slide_id}")

                heatmap_dir = get_slide_heatmaps_region_level(
                    slide_path,
                    coordinates_dir,
                    encoder,
                    region_model,
                    transforms,
                    output_dir_slide,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    threshold=cfg.threshold,
                    highlight=cfg.highlight,
                    granular=cfg.smoothing.region,
                    offset=cfg.smoothing.offset.region,
                    gaussian_smoothing=cfg.gaussian_smoothing,
                    segmentation_mask_path=seg_mask_path,
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    background_pixel_value=cfg.background_pixel_value,
                    tissue_pixel_value=cfg.tissue_pixel_value,
                    patch_attn_mask=mask_p,
                    mini_patch_attn_mask=mask_mp,
                    patch_device=device,
                    region_device=device,
                    main_process=distributed.is_main_process(),
                )

                heatmap_dir_reg = heatmap_dir / "regular"
                stitched_heatmap_dir = stitch_slide_heatmaps(
                    slide_path,
                    heatmap_dir_reg,
                    output_dir_slide,
                    name="region",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    segmentation_mask_path=seg_mask_path,
                    restrict_to_tissue=cfg.restrict_to_tissue,
                )

                heatmap_dir_thresh = heatmap_dir / "thresholded"
                if heatmap_dir_thresh.exists():
                    stitched_heatmap_thresh_dir = stitch_slide_heatmaps(
                        slide_path,
                        heatmap_dir_thresh,
                        output_dir_slide,
                        name="region",
                        suffix="thresholded",
                        spacing=cfg.spacing,
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        cmap=custom_cmap,
                        segmentation_mask_path=seg_mask_path,
                        restrict_to_tissue=cfg.restrict_to_tissue,
                    )

                heatmap_dir_highlight = heatmap_dir / "highlighted"
                if heatmap_dir_highlight.exists():
                    stitched_heatmap_highlight_dir = stitch_slide_heatmaps(
                        slide_path,
                        heatmap_dir_highlight,
                        output_dir_slide,
                        name="region",
                        suffix="highlighted",
                        spacing=cfg.spacing,
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        cmap=custom_cmap,
                        highlight=(cfg.highlight != None),
                        opacity=cfg.opacity,
                        segmentation_mask_path=seg_mask_path,
                        restrict_to_tissue=cfg.restrict_to_tissue,
                    )

                if cfg.display:
                    display_stitched_heatmaps(
                        slide_path,
                        stitched_heatmap_dir,
                        output_dir_slide,
                        name="region",
                        display_patching=True,
                        cmap=custom_cmap,
                        coordinates_dir=coordinates_dir,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                        run_id=run_id,
                    )
                    if heatmap_dir_thresh.exists():
                        display_stitched_heatmaps(
                            slide_path,
                            stitched_heatmap_thresh_dir,
                            output_dir_slide,
                            name="region_thresh",
                            display_patching=True,
                            cmap=custom_cmap,
                            coordinates_dir=coordinates_dir,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                            run_id=run_id,
                        )
                    if heatmap_dir_highlight.exists():
                        display_stitched_heatmaps(
                            slide_path,
                            stitched_heatmap_highlight_dir,
                            output_dir_slide,
                            name="region_highlight",
                            display_patching=True,
                            cmap=custom_cmap,
                            coordinates_dir=coordinates_dir,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                            run_id=run_id,
                        )

                ########################
                # SLIDE-LEVEL HEATMAPS #
                ########################

                if distributed.is_main_process():
                    tqdm.tqdm.write(f"Computing slide-level heatmaps for {slide_id}")

                heatmap_dir = get_slide_heatmaps_slide_level(
                    slide_path,
                    coordinates_dir,
                    encoder,
                    region_model,
                    slide_model,
                    transforms,
                    output_dir_slide,
                    patch_size=256,
                    downscale=cfg.downscale,
                    threshold=cfg.threshold,
                    highlight=cfg.highlight,
                    cmap=custom_cmap,
                    granular=cfg.smoothing.slide,
                    offset=cfg.smoothing.offset.slide,
                    spacing=cfg.spacing,
                    gaussian_smoothing=cfg.gaussian_smoothing,
                    gaussian_offset=cfg.gaussian_offset,
                    patch_attn_mask=mask_p,
                    mini_patch_attn_mask=mask_mp,
                    patch_device=device,
                    region_device=device,
                    slide_device=device,
                )

                heatmap_dir_reg = heatmap_dir / "regular"
                stitched_heatmap_dir = stitch_slide_heatmaps(
                    slide_path,
                    heatmap_dir_reg,
                    output_dir_slide,
                    name="wsi",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    segmentation_mask_path=seg_mask_path,
                    restrict_to_tissue=cfg.restrict_to_tissue,
                )

                heatmap_dir_thresh = heatmap_dir / "thresholded"
                if heatmap_dir_thresh.exists():
                    stitched_heatmap_thresh_dir = stitch_slide_heatmaps(
                        slide_path,
                        heatmap_dir_thresh,
                        output_dir_slide,
                        name="wsi",
                        suffix="thresholded",
                        spacing=cfg.spacing,
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        cmap=custom_cmap,
                        segmentation_mask_path=seg_mask_path,
                        restrict_to_tissue=cfg.restrict_to_tissue,
                    )

                heatmap_dir_highlight = heatmap_dir / "highlighted"
                if heatmap_dir_highlight.exists():
                    stitched_heatmap_highlight_dir = stitch_slide_heatmaps(
                        slide_path,
                        heatmap_dir_highlight,
                        output_dir_slide,
                        name="wsi",
                        suffix="highlighted",
                        spacing=cfg.spacing,
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        cmap=custom_cmap,
                        highlight=(cfg.highlight != None),
                        opacity=cfg.opacity,
                        segmentation_mask_path=seg_mask_path,
                        restrict_to_tissue=cfg.restrict_to_tissue,
                    )

                if cfg.display:
                    display_stitched_heatmaps(
                        slide_path,
                        stitched_heatmap_dir,
                        output_dir_slide,
                        fname="wsi",
                        display_patching=True,
                        draw_grid=False,
                        cmap=custom_cmap,
                        coordinates_dir=coordinates_dir,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                        run_id=run_id,
                    )

                ####################
                # BLENDED HEATMAPS #
                ####################

                if distributed.is_main_process():
                    tqdm.tqdm.write(f"Computing blended heatmaps (gamma={cfg.gamma}) for {slide_id}")

                heatmap_dir = get_slide_blended_heatmaps(
                    slide_path,
                    coordinates_dir,
                    encoder,
                    region_model,
                    slide_model,
                    transforms,
                    cfg.level,
                    output_dir_slide,
                    gamma=cfg.gamma,
                    patch_size=256,
                    mini_patch_size=16,
                    downscale=cfg.downscale,
                    threshold=cfg.threshold,
                    cmap=custom_cmap,
                    smoothing=cfg.smoothing,
                    segmentation_mask_path=seg_mask_path,
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    background_pixel_value=cfg.background_pixel_value,
                    tissue_pixel_value=cfg.tissue_pixel_value,
                    patch_attn_mask=mask_p,
                    mini_patch_attn_mask=mask_mp,
                    compute_patch_attention=False,
                    patch_device=device,
                    region_device=device,
                    slide_device=device,
                )

                heatmap_dir_reg = heatmap_dir / "regular"
                stitched_heatmap_dir = stitch_slide_heatmaps(
                    slide_path,
                    heatmap_dir_reg,
                    output_dir_slide,
                    name="blended",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    segmentation_mask_path=seg_mask_path,
                    restrict_to_tissue=cfg.restrict_to_tissue,
                )

                heatmap_dir_thresh = heatmap_dir / "thresholded"
                if heatmap_dir_thresh.exists():
                    stitched_heatmap_thresh_dir = stitch_slide_heatmaps(
                        slide_path,
                        heatmap_dir_thresh,
                        output_dir_slide,
                        name="blended",
                        suffix="thresholded",
                        spacing=cfg.spacing,
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        cmap=custom_cmap,
                        segmentation_mask_path=seg_mask_path,
                        restrict_to_tissue=cfg.restrict_to_tissue,
                    )

                heatmap_dir_highlight = heatmap_dir / "highlighted"
                if heatmap_dir_highlight.exists():
                    stitched_heatmap_highlight_dir = stitch_slide_heatmaps(
                        slide_path,
                        heatmap_dir_highlight,
                        output_dir_slide,
                        name="blended",
                        suffix="highlighted",
                        spacing=cfg.spacing,
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        cmap=custom_cmap,
                        highlight=(cfg.highlight != None),
                        opacity=cfg.opacity,
                        segmentation_mask_path=seg_mask_path,
                        restrict_to_tissue=cfg.restrict_to_tissue,
                    )

                if cfg.display:
                    display_stitched_heatmaps(
                        slide_path,
                        stitched_heatmap_dir,
                        output_dir_slide,
                        fname="blended",
                        display_patching=True,
                        draw_grid=False,
                        cmap=custom_cmap,
                        coordinates_dir=coordinates_dir,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                        run_id=run_id,
                    )

                if cfg.wandb.enable and not distributed.is_enabled_and_multiple_gpus():
                    wandb.log({"processed": i + 1})


if __name__ == "__main__":

    # python3 attention_visualization.py --config-name 'default'

    main()
