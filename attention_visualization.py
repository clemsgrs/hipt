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

from source.dataset import SlideFilepathsDataset
from source.utils import initialize_wandb, is_main_process
from source.attention_visualization_utils import (
    cmap_map,
    get_patch_model,
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
    get_slide_hierarchical_heatmaps_region,
    get_slide_blended_heatmaps,
    stitch_slide_heatmaps,
    display_stitched_heatmaps,
    load_heatmaps_from_disk,
)


@hydra.main(version_base="1.2.0", config_path="config/heatmaps", config_name="default")
def main(cfg: DictConfig):
    distributed = torch.cuda.device_count() > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        if gpu_id == 0:
            print(f"Distributed session successfully initialized")
    else:
        gpu_id = -1

    if is_main_process():
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

    if distributed:
        obj = [run_id]
        torch.distributed.broadcast_object_list(
            obj, 0, device=torch.device(f"cuda:{gpu_id}")
        )
        run_id = obj[0]

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    if is_main_process():
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

    if gpu_id == -1:
        device = torch.device(f"cuda")
    else:
        device = torch.device(f"cuda:{gpu_id}")

    patch_weights = Path(cfg.patch_weights)
    patch_model = get_patch_model(
        pretrained_weights=patch_weights,
        mask_attn=cfg.mask_attn_patch,
        device=device,
        verbose=is_main_process(),
    )

    region_weights = Path(cfg.region_weights)
    region_model = get_region_model(
        pretrained_weights=region_weights,
        region_size=cfg.region_size,
        mask_attn=cfg.mask_attn_region,
        img_size_pretrained=cfg.img_size_pretrained,
        device=device,
        verbose=is_main_process(),
    )

    if cfg.slide_weights:

        slide_weights = Path(cfg.slide_weights)
        sd = torch.load(slide_weights, map_location="cpu")
        start_idx = list(sd.keys()).index("global_phi.0.weight")
        end_idx = list(sd.keys()).index("global_rho.0.bias")
        sd = {
            k: v
            for i, (k, v) in enumerate(sd.items())
            if i >= start_idx and i <= end_idx
        }
        if is_main_process():
            print(f"Pretrained weights found at {slide_weights}")

        slide_model = get_slide_model(sd, device=device)

    custom_cmap = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.jet)
    # custom_cmap = cmap_map(lambda x: x / 2 + 0.5, matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","white","lime"]))

    if cfg.patch_fp and is_main_process():
        patch = Image.open(cfg.patch_fp)

        print(f"Computing indiviudal patch-level attention heatmaps")
        output_dir_patch = Path(output_dir, "patch_indiv")
        output_dir_patch.mkdir(exist_ok=True, parents=True)
        create_patch_heatmaps_indiv(
            patch,
            patch_model,
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
            patch_model,
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

    if cfg.region_fp and is_main_process():
        region = Image.open(cfg.region_fp)

        print(f"Computing individual region-level attention heatmaps")
        output_dir_region = Path(output_dir, "region_indiv")
        output_dir_region.mkdir(exist_ok=True, parents=True)
        create_region_heatmaps_indiv(
            region,
            patch_model,
            region_model,
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
            patch_model,
            region_model,
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

    if cfg.slide_fp and is_main_process():

        slide_path = Path(cfg.slide_fp)
        slide_id = slide_path.stem
        region_dir = Path(cfg.region_dir)

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

        load_from_disk, hms, hms_thresh, hms_highlight, coords = get_slide_heatmaps_patch_level(
            slide_id,
            region_dir,
            patch_model,
            region_model,
            output_dir_slide,
            downscale=cfg.downscale,
            cmap=custom_cmap,
            threshold=cfg.threshold,
            highlight=cfg.highlight,
            save_to_disk=False,
            granular=cfg.smoothing.patch,
            offset=cfg.smoothing.offset.patch,
            slide_path=slide_path,
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

        if load_from_disk:
            patch_hm_output_dir = Path(output_dir_slide, "patch", "256")
            hms, coords = load_heatmaps_from_disk(patch_hm_output_dir, disable=is_main_process())

        stitched_hms = {}
        with tqdm.tqdm(
            hms.items(),
            desc=f"Stitching {slide_id} patch-level heatmaps",
            unit=" head",
            leave=False,
            disable=not is_main_process(),
        ) as t:
            for head_num, heatmaps in t:
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    heatmaps,
                    coords[head_num],
                    output_dir_slide,
                    fname=f"patch_head_{head_num}",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    save_to_disk=True,
                    restrict_to_tissue=cfg.restrict_to_tissue,
                    segmentation_mask_path=cfg.segmentation_mask_fp,
                    tissue_pixel_value=cfg.tissue_pixel_value,
                )
                stitched_hms[f"Head {head_num}"] = stitched_hm

        del hms

        stitched_hms_thresh = {}
        if len(hms_thresh) > 0:
            for head_num, heatmaps in hms_thresh.items():
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    heatmaps,
                    coords[head_num],
                    output_dir_slide,
                    fname=f"patch_head_{head_num}_thresh",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    save_to_disk=True,
                )
                stitched_hms_thresh[f"Head {head_num}"] = stitched_hm

        stitched_hms_highlight = {}
        if len(hms_highlight) > 0:
            for head_num, heatmaps in hms_highlight.items():
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    heatmaps,
                    coords[head_num],
                    output_dir_slide,
                    fname=f"patch_head_{head_num}_highlight",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    save_to_disk=True,
                    highlight=(cfg.highlight != None),
                    opacity=cfg.opacity,
                    restrict_to_tissue=cfg.restrict_to_tissue,
                    seg_params=cfg.seg_params,
                )
                stitched_hms_highlight[f"Head {head_num}"] = stitched_hm

        if cfg.display:
            display_stitched_heatmaps(
                slide_path,
                stitched_hms,
                output_dir_slide,
                fname=f"patch",
                display_patching=True,
                cmap=custom_cmap,
                region_dir=region_dir,
                region_size=cfg.region_size,
                downsample=cfg.downsample,
                font_fp=cfg.font_fp,
                run_id=run_id,
            )
            if len(stitched_hms_thresh) > 0:
                display_stitched_heatmaps(
                    slide_path,
                    stitched_hms_thresh,
                    output_dir_slide,
                    fname=f"patch_thresh",
                    display_patching=True,
                    cmap=custom_cmap,
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )
            if len(stitched_hms_highlight) > 0:
                display_stitched_heatmaps(
                    slide_path,
                    stitched_hms_highlight,
                    output_dir_slide,
                    fname=f"patch_highlight",
                    display_patching=True,
                    cmap=custom_cmap,
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )

        #########################
        # REGION-LEVEL HEATMAPS #
        #########################

        print("Computing region-level heatmaps")

        hms, hms_thresh, hms_highlight, coords = get_slide_heatmaps_region_level(
            slide_id,
            region_dir,
            patch_model,
            region_model,
            output_dir_slide,
            downscale=cfg.downscale,
            cmap=custom_cmap,
            save_to_disk=True,
            threshold=cfg.threshold,
            highlight=cfg.highlight,
            granular=cfg.smoothing.region,
            offset=cfg.smoothing.offset.region,
            gaussian_smoothing=cfg.gaussian_smoothing,
            slide_path=slide_path,
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

        stitched_hms = {}
        for head_num, heatmaps in hms.items():
            stitched_hm = stitch_slide_heatmaps(
                slide_path,
                heatmaps,
                coords[head_num],
                output_dir_slide,
                fname=f"region_head_{head_num}",
                spacing=cfg.spacing,
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                cmap=custom_cmap,
                save_to_disk=True,
                restrict_to_tissue=cfg.restrict_to_tissue,
                segmentation_mask_path=cfg.segmentation_mask_fp,
                tissue_pixel_value=cfg.tissue_pixel_value,
            )
            stitched_hms[f"Head {head_num}"] = stitched_hm

        stitched_hms_thresh = {}
        if len(hms_thresh) > 0:
            for head_num, heatmaps in hms_thresh.items():
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    heatmaps,
                    coords[head_num],
                    output_dir_slide,
                    fname=f"region_head_{head_num}_thresh",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    save_to_disk=True,
                )
                stitched_hms_thresh[f"Head {head_num}"] = stitched_hm

        stitched_hms_highlight = {}
        if len(hms_highlight) > 0:
            for head_num, heatmaps in hms_highlight.items():
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    heatmaps,
                    coords[head_num],
                    output_dir_slide,
                    fname=f"region_head_{head_num}_highlight",
                    spacing=cfg.spacing,
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    save_to_disk=True,
                    highlight=(cfg.highlight != None),
                    opacity=cfg.opacity,
                    restrict_to_tissue=cfg.restrict_to_tissue,
                    seg_params=cfg.seg_params,
                )
                stitched_hms_highlight[f"Head {head_num}"] = stitched_hm

        if cfg.display:
            display_stitched_heatmaps(
                slide_path,
                stitched_hms,
                output_dir_slide,
                fname=f"region",
                display_patching=True,
                cmap=custom_cmap,
                region_dir=region_dir,
                region_size=cfg.region_size,
                downsample=cfg.downsample,
                font_fp=cfg.font_fp,
                run_id=run_id,
            )
            if len(stitched_hms_thresh) > 0:
                display_stitched_heatmaps(
                    slide_path,
                    stitched_hms_thresh,
                    output_dir_slide,
                    fname=f"region_thresh",
                    display_patching=True,
                    cmap=custom_cmap,
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )
            if len(stitched_hms_highlight) > 0:
                display_stitched_heatmaps(
                    slide_path,
                    stitched_hms_highlight,
                    output_dir_slide,
                    fname=f"region_highlight",
                    display_patching=True,
                    cmap=custom_cmap,
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )

        ##################################
        # HIERARCHICAL HEATMAPS (REGION) #
        ##################################

        # hms, hms_thresh, coords = get_slide_hierarchical_heatmaps_region(
        #     slide_id,
        #     patch_model,
        #     region_model,
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
        #             region_dir=region_dir,
        #             region_size=cfg.region_size,
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
                slide_id,
                region_dir,
                patch_model,
                region_model,
                slide_model,
                patch_size=256,
                downscale=cfg.downscale,
                threshold=cfg.threshold,
                highlight=cfg.highlight,
                cmap=custom_cmap,
                region_fmt="jpg",
                granular=cfg.smoothing.slide,
                offset=cfg.smoothing.offset.slide,
                slide_path=slide_path,
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
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                    run_id=run_id,
                )

            ####################
            # BLENDED HEATMAPS #
            ####################

            print("Computing factorized heatmaps")

            hms, thresh_hms, coords = get_slide_blended_heatmaps(
                slide_id,
                region_dir,
                patch_model,
                region_model,
                slide_model,
                cfg.level,
                output_dir_slide,
                gamma=cfg.gamma,
                patch_size=256,
                mini_patch_size=16,
                downscale=cfg.downscale,
                threshold=cfg.threshold,
                cmap=custom_cmap,
                region_fmt="jpg",
                save_to_disk=False,
                smoothing=cfg.smoothing,
                slide_path=slide_path,
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
                for phead_num in range(patch_model.num_heads):
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
                    for phead_num in range(patch_model.num_heads):
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
                        region_dir=region_dir,
                        region_size=cfg.region_size,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                        run_id=run_id,
                    )

    if cfg.slide_csv:

        df = pd.read_csv(cfg.slide_csv)
        dataset = SlideFilepathsDataset(df)

        if distributed:
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
            disable=not is_main_process(),
        ) as t1:
            for i, batch in enumerate(t1):

                _, slide_fp, seg_mask_path = batch
                slide_path = Path(slide_fp[0])
                if seg_mask_path != None:
                    seg_mask_path = Path(seg_mask_path[0])
                slide_id = slide_path.stem
                region_dir = Path(cfg.region_dir)

                output_dir_slide = Path(output_dir, slide_id)

                mask_p, mask_mp = None, None
                if mask_attention:
                    mask_p, mask_mp = generate_masks(
                        slide_id,
                        slide_path,
                        seg_mask_path,
                        region_dir,
                        region_size=cfg.region_size,
                        spacing=cfg.spacing,
                        downsample=1,
                        tissue_pixel_value=cfg.tissue_pixel_value,
                    )

                ########################
                # PATCH-LEVEL HEATMAPS #
                ########################

                load_from_disk, hms, hms_thresh, hms_highlight, coords = get_slide_heatmaps_patch_level(
                    slide_id,
                    region_dir,
                    patch_model,
                    region_model,
                    output_dir_slide,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    threshold=cfg.threshold,
                    highlight=cfg.highlight,
                    save_to_disk=True,
                    granular=cfg.smoothing.patch,
                    offset=cfg.smoothing.offset.patch,
                    slide_path=slide_path,
                    segmentation_mask_path=seg_mask_path,
                    spacing=cfg.spacing,
                    downsample=1,
                    background_pixel_value=cfg.background_pixel_value,
                    tissue_pixel_value=cfg.tissue_pixel_value,
                    patch_attn_mask=mask_p,
                    mini_patch_attn_mask=mask_mp,
                    patch_device=device,
                    region_device=device,
                    main_process=is_main_process(),
                )

                if load_from_disk:
                    patch_hm_output_dir = Path(output_dir_slide, "patch", "256")
                    hms, coords = load_heatmaps_from_disk(patch_hm_output_dir, disable=is_main_process())

                stitched_hms = {}
                with tqdm.tqdm(
                    hms.items(),
                    desc=f"Stitching {slide_id} patch-level heatmaps",
                    unit=" head",
                    leave=False,
                    disable=not is_main_process(),
                ) as t2:
                    for head_num, heatmaps in t2:
                        stitched_hm = stitch_slide_heatmaps(
                            slide_path,
                            heatmaps,
                            coords[head_num],
                            output_dir_slide,
                            fname=f"patch_head_{head_num}",
                            spacing=cfg.spacing,
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            cmap=custom_cmap,
                            save_to_disk=True,
                            restrict_to_tissue=cfg.restrict_to_tissue,
                            segmentation_mask_path=seg_mask_path,
                            tissue_pixel_value=cfg.tissue_pixel_value,
                        )
                        stitched_hms[f"Head {head_num}"] = stitched_hm

                del hms

                stitched_hms_thresh = {}
                if len(hms_thresh) > 0:
                    for head_num, heatmaps in hms_thresh.items():
                        stitched_hm = stitch_slide_heatmaps(
                            slide_path,
                            heatmaps,
                            coords[head_num],
                            output_dir_slide,
                            fname=f"region_head_{head_num}_thresh",
                            spacing=cfg.spacing,
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            cmap=custom_cmap,
                            save_to_disk=True,
                        )
                        stitched_hms_thresh[f"Head {head_num}"] = stitched_hm

                stitched_hms_highlight = {}
                if len(hms_highlight) > 0:
                    for head_num, heatmaps in hms_highlight.items():
                        stitched_hm = stitch_slide_heatmaps(
                            slide_path,
                            heatmaps,
                            coords[head_num],
                            output_dir_slide,
                            fname=f"patch_head_{head_num}_highlight",
                            spacing=cfg.spacing,
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            cmap=custom_cmap,
                            save_to_disk=True,
                            highlight=(cfg.highlight != None),
                            opacity=cfg.opacity,
                            restrict_to_tissue=cfg.restrict_to_tissue,
                            seg_params=cfg.seg_params,
                        )
                        stitched_hms_highlight[f"Head {head_num}"] = stitched_hm


                if cfg.display:
                    display_stitched_heatmaps(
                        slide_path,
                        stitched_hms,
                        output_dir_slide,
                        fname=f"patch",
                        display_patching=True,
                        cmap=custom_cmap,
                        region_dir=region_dir,
                        region_size=cfg.region_size,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                        run_id=run_id,
                    )
                    if len(stitched_hms_thresh) > 0:
                        display_stitched_heatmaps(
                            slide_path,
                            stitched_hms_thresh,
                            output_dir_slide,
                            fname=f"patch_thresh",
                            display_patching=True,
                            cmap=custom_cmap,
                            region_dir=region_dir,
                            region_size=cfg.region_size,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                            run_id=run_id,
                        )
                    if len(stitched_hms_highlight) > 0:
                        display_stitched_heatmaps(
                            slide_path,
                            stitched_hms_highlight,
                            output_dir_slide,
                            fname=f"patch_highlight",
                            display_patching=True,
                            cmap=custom_cmap,
                            region_dir=region_dir,
                            region_size=cfg.region_size,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                            run_id=run_id,
                        )

                #########################
                # REGION-LEVEL HEATMAPS #
                #########################

                hms, hms_thresh, hms_highlight, coords = get_slide_heatmaps_region_level(
                    slide_id,
                    region_dir,
                    patch_model,
                    region_model,
                    output_dir_slide,
                    downscale=cfg.downscale,
                    cmap=custom_cmap,
                    save_to_disk=False,
                    threshold=cfg.threshold,
                    highlight=cfg.highlight,
                    granular=cfg.smoothing.region,
                    offset=cfg.smoothing.offset.region,
                    gaussian_smoothing=cfg.gaussian_smoothing,
                    slide_path=slide_path,
                    segmentation_mask_path=seg_mask_path,
                    spacing=cfg.spacing,
                    downsample=1,
                    background_pixel_value=cfg.background_pixel_value,
                    tissue_pixel_value=cfg.tissue_pixel_value,
                    patch_attn_mask=mask_p,
                    mini_patch_attn_mask=mask_mp,
                    patch_device=device,
                    region_device=device,
                    main_process=is_main_process(),
                )

                stitched_hms = {}
                for head_num, heatmaps in hms.items():
                    stitched_hm = stitch_slide_heatmaps(
                        slide_path,
                        heatmaps,
                        coords[head_num],
                        output_dir_slide,
                        fname=f"region_head_{head_num}",
                        spacing=cfg.spacing,
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        cmap=custom_cmap,
                        save_to_disk=True,
                        restrict_to_tissue=cfg.restrict_to_tissue,
                        segmentation_mask_path=cfg.segmentation_mask_fp,
                        tissue_pixel_value=cfg.tissue_pixel_value,
                    )
                    stitched_hms[f"Head {head_num}"] = stitched_hm

                stitched_hms_thresh = {}
                if len(hms_thresh) > 0:
                    for head_num, heatmaps in hms_thresh.items():
                        stitched_hm = stitch_slide_heatmaps(
                            slide_path,
                            heatmaps,
                            coords[head_num],
                            output_dir_slide,
                            fname=f"region_head_{head_num}_thresh",
                            spacing=cfg.spacing,
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            cmap=custom_cmap,
                            save_to_disk=True,
                        )
                        stitched_hms_thresh[f"Head {head_num}"] = stitched_hm

                stitched_hms_highlight = {}
                if len(hms_highlight) > 0:
                    for head_num, heatmaps in hms_highlight.items():
                        stitched_hm = stitch_slide_heatmaps(
                            slide_path,
                            heatmaps,
                            coords[head_num],
                            output_dir_slide,
                            fname=f"region_head_{head_num}_highlight",
                            spacing=cfg.spacing,
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            cmap=custom_cmap,
                            save_to_disk=True,
                            highlight=(cfg.highlight != None),
                            opacity=cfg.opacity,
                            restrict_to_tissue=cfg.restrict_to_tissue,
                            seg_params=cfg.seg_params,
                        )
                        stitched_hms_highlight[f"Head {head_num}"] = stitched_hm

                if cfg.display:
                    display_stitched_heatmaps(
                        slide_path,
                        stitched_hms,
                        output_dir_slide,
                        fname=f"region",
                        display_patching=True,
                        cmap=custom_cmap,
                        region_dir=region_dir,
                        region_size=cfg.region_size,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                        run_id=run_id,
                    )
                    if len(stitched_hms_thresh) > 0:
                        display_stitched_heatmaps(
                            slide_path,
                            stitched_hms_thresh,
                            output_dir_slide,
                            fname=f"region_thresh",
                            display_patching=True,
                            cmap=custom_cmap,
                            region_dir=region_dir,
                            region_size=cfg.region_size,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                            run_id=run_id,
                        )
                    if len(stitched_hms_highlight) > 0:
                        display_stitched_heatmaps(
                            slide_path,
                            stitched_hms_highlight,
                            output_dir_slide,
                            fname=f"region_highlight",
                            display_patching=True,
                            cmap=custom_cmap,
                            region_dir=region_dir,
                            region_size=cfg.region_size,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                            run_id=run_id,
                        )

                if cfg.slide_weights:

                    ########################
                    # SLIDE-LEVEL HEATMAPS #
                    ########################

                    hms, thresh_hms, highlight_hms, coords = get_slide_heatmaps_slide_level(
                        slide_id,
                        region_dir,
                        patch_model,
                        region_model,
                        slide_model,
                        patch_size=256,
                        downscale=cfg.downscale,
                        threshold=cfg.threshold,
                        highlight=cfg.highlight,
                        cmap=custom_cmap,
                        region_fmt="jpg",
                        granular=cfg.smoothing.slide,
                        offset=cfg.smoothing.offset.slide,
                        slide_path=slide_path,
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
                        restrict_to_tissue=False,
                        segmentation_mask_path=seg_mask_path,
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
                            cmap=custom_cmap,
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
                            cmap=custom_cmap,
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
                            region_dir=region_dir,
                            region_size=cfg.region_size,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                            run_id=run_id,
                        )

                    ####################
                    # BLENDED HEATMAPS #
                    ####################

                    hms, thresh_hms, coords = get_slide_blended_heatmaps(
                        slide_id,
                        region_dir,
                        patch_model,
                        region_model,
                        slide_model,
                        cfg.level,
                        output_dir_slide,
                        gamma=cfg.gamma,
                        patch_size=256,
                        mini_patch_size=16,
                        downscale=cfg.downscale,
                        threshold=cfg.threshold,
                        cmap=custom_cmap,
                        region_fmt="jpg",
                        save_to_disk=False,
                        smoothing=cfg.smoothing,
                        slide_path=slide_path,
                        segmentation_mask_path=seg_mask_path,
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
                        for phead_num in range(patch_model.num_heads):
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
                                segmentation_mask_path=seg_mask_path,
                                tissue_pixel_value=cfg.tissue_pixel_value,
                            )
                            stitched_hms[rhead_num].append(stitched_hm)

                    if len(thresh_hms) > 0:
                        stitched_hms_thresh = defaultdict(list)
                        for rhead_num in range(region_model.num_heads):
                            for phead_num in range(patch_model.num_heads):
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
                                region_dir=region_dir,
                                region_size=cfg.region_size,
                                downsample=cfg.downsample,
                                font_fp=cfg.font_fp,
                                run_id=run_id,
                            )

                if cfg.wandb.enable and not distributed:
                    wandb.log({"processed": i + 1})


if __name__ == "__main__":

    # python3 attention_visualization.py --config-name 'default'

    main()
