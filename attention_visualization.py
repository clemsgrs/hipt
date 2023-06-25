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
    create_patch_heatmaps_indiv,
    create_patch_heatmaps_concat,
    create_hierarchical_heatmaps_indiv,
    create_hierarchical_heatmaps_concat,
    get_slide_patch_level_heatmaps,
    get_slide_region_level_heatmaps,
    get_slide_hierarchical_heatmaps,
    get_slide_level_heatmaps,
    stitch_slide_heatmaps,
    display_stitched_heatmaps,
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
    patch_model = get_patch_model(pretrained_weights=patch_weights, device=device)

    region_weights = Path(cfg.region_weights)
    region_model = get_region_model(
        pretrained_weights=region_weights, region_size=cfg.region_size, device=device
    )

    light_jet = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.jet)

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
            cmap=light_jet,
            granular=cfg.granular,
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
            cmap=light_jet,
            granular=cfg.granular,
            patch_device=device,
        )
        print("done!")

    if cfg.region_fp and is_main_process():

        region = Image.open(cfg.region_fp)

        print(f"Computing individual region-level attention heatmaps")
        output_dir_region = Path(output_dir, "region_indiv")
        output_dir_region.mkdir(exist_ok=True, parents=True)
        create_hierarchical_heatmaps_indiv(
            region,
            patch_model,
            region_model,
            output_dir_region,
            downscale=cfg.downscale,
            threshold=0.5,
            alpha=0.5,
            cmap=light_jet,
            granular=cfg.granular,
            patch_device=device,
            region_device=device,
        )
        print("done!")

        print(f"Computing concatenated region-level attention heatmaps")
        output_dir_region_concat = Path(output_dir, "region_concat")
        output_dir_region_concat.mkdir(exist_ok=True, parents=True)
        create_hierarchical_heatmaps_concat(
            region,
            patch_model,
            region_model,
            output_dir_region_concat,
            downscale=cfg.downscale,
            alpha=0.5,
            cmap=light_jet,
            granular=cfg.granular,
            patch_device=device,
            region_device=device,
        )
        print("done!")

    if cfg.slide_fp and is_main_process():

        slide_path = Path(cfg.slide_fp)
        slide_id = slide_path.stem
        region_dir = Path(cfg.region_dir)

        output_dir_slide = Path(output_dir, slide_id)

        hms, hms_thresh, coords = get_slide_region_level_heatmaps(
            slide_id,
            patch_model,
            region_model,
            region_dir,
            output_dir_slide,
            downscale=cfg.downscale,
            cmap=light_jet,
            save_to_disk=True,
            granular=cfg.granular,
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
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                save_to_disk=True,
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
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    save_to_disk=True,
                )
                stitched_hms_thresh[f"Head {head_num}"] = stitched_hm

        if cfg.display:
            display_stitched_heatmaps(
                slide_path,
                stitched_hms,
                output_dir_slide,
                fname=f"region",
                display_patching=True,
                region_dir=region_dir,
                region_size=cfg.region_size,
                downsample=cfg.downsample,
                font_fp=cfg.font_fp,
            )
            if len(stitched_hms_thresh) > 0:
                display_stitched_heatmaps(
                    slide_path,
                    stitched_hms_thresh,
                    output_dir_slide,
                    fname=f"region_thresh",
                    display_patching=True,
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                )

        hms, hms_thresh, coords = get_slide_patch_level_heatmaps(
            slide_id,
            patch_model,
            region_model,
            region_dir,
            output_dir_slide,
            downscale=cfg.downscale,
            cmap=light_jet,
            threshold=None,
            save_to_disk=True,
            granular=cfg.granular,
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
                fname=f"patch_head_{head_num}",
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                save_to_disk=True,
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
                    fname=f"patch_head_{head_num}_thresh",
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    save_to_disk=True,
                )
                stitched_hms_thresh[f"Head {head_num}"] = stitched_hm

        if cfg.display:
            display_stitched_heatmaps(
                slide_path,
                stitched_hms,
                output_dir_slide,
                fname=f"patch",
                display_patching=True,
                region_dir=region_dir,
                region_size=cfg.region_size,
                downsample=cfg.downsample,
                font_fp=cfg.font_fp,
            )
            if len(stitched_hms_thresh) > 0:
                display_stitched_heatmaps(
                    slide_path,
                    stitched_hms_thresh,
                    output_dir_slide,
                    fname=f"patch_thresh",
                    display_patching=True,
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                )

        hms, hms_thresh, coords = get_slide_hierarchical_heatmaps(
            slide_id,
            patch_model,
            region_model,
            region_dir,
            output_dir_slide,
            downscale=cfg.downscale,
            cmap=light_jet,
            threshold=None,
            save_to_disk=True,
            granular=cfg.granular,
            patch_device=device,
            region_device=device,
        )

        stitched_hms = defaultdict(list)
        for rhead_num, hm_dict in hms.items():
            coords_dict = coords[rhead_num]
            for phead_num, heatmaps in hm_dict.items():
                coordinates = coords_dict[phead_num]
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    heatmaps,
                    coordinates,
                    output_dir_slide,
                    fname=f"rhead_{rhead_num}_phead_{phead_num}",
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    save_to_disk=True,
                )
                stitched_hms[rhead_num].append(stitched_hm)

        stitched_hms_thresh = defaultdict(list)
        for rhead_num, hm_dict in hms_thresh.items():
            coords_dict = coords[rhead_num]
            for phead_num, heatmaps in hm_dict.items():
                coordinates = coords_dict[phead_num]
                stitched_hm = stitch_slide_heatmaps(
                    slide_path,
                    heatmaps,
                    coordinates,
                    output_dir_slide,
                    fname=f"rhead_{rhead_num}_phead_{phead_num}_thresh",
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    save_to_disk=True,
                )
                stitched_hms_thresh[rhead_num].append(stitched_hm)
        
        if cfg.display:
            for rhead_num, hms in stitched_hms.items():
                d = {f'Region Head {rhead_num} & Patch Head {phead}': hm for phead, hm in enumerate(hms)}
                display_stitched_heatmaps(
                    slide_path,
                    d,
                    output_dir_slide,
                    fname=f"rhead_{rhead_num}",
                    display_patching=True,
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                )

        if cfg.slide_weights:

            slide_weights = Path(cfg.slide_weights)
            sd = torch.load(slide_weights, map_location="cpu")
            start_idx = list(sd.keys()).index('global_phi.0.weight')
            end_idx = list(sd.keys()).index('global_rho.0.bias')
            sd = {k: v for i, (k,v) in enumerate(sd.items()) if i >= start_idx and i <= end_idx}
            print(f"Pretrained weights found at {slide_weights}")

            slide_model = get_slide_model(sd, device=device)

            hms, thresh_hms, coords = get_slide_level_heatmaps(
                slide_id,
                patch_model,
                region_model,
                slide_model,
                region_dir,
                region_size=cfg.region_size,
                patch_size=256,
                downscale=cfg.downscale,
                threshold=0.5,
                cmap=light_jet,
                region_fmt="jpg",
                device=torch.device("cuda:0"),
            )

            stitched_hms = {}
            stitched_hm = stitch_slide_heatmaps(
                slide_path,
                hms,
                coords,
                output_dir_slide,
                fname=f"wsi",
                downsample=cfg.downsample,
                downscale=cfg.downscale,
                save_to_disk=True,
            )
            stitched_hms[f"slide-level"] = stitched_hm

            if len(thresh_hms) > 0:

                stitched_thresh_hm = stitch_slide_heatmaps(
                    slide_path,
                    thresh_hms,
                    coords,
                    output_dir_slide,
                    fname=f"wsi_thresh",
                    downsample=cfg.downsample,
                    downscale=cfg.downscale,
                    save_to_disk=True,
                )
                stitched_hms[f"thresholded"] = stitched_thresh_hm

            if cfg.display:
                display_stitched_heatmaps(
                    slide_path,
                    stitched_hms,
                    output_dir,
                    fname=f"wsi",
                    display_patching=True,
                    region_dir=region_dir,
                    region_size=cfg.region_size,
                    downsample=cfg.downsample,
                    font_fp=cfg.font_fp,
                )


    if cfg.slide_csv:

        df = pd.read_csv("cfg.slide_csv")
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

            for i, fp in enumerate(t1):

                slide_path = Path(fp)
                slide_id = slide_path.stem
                region_dir = Path(cfg.region_dir)

                output_dir_slide = Path(output_dir, slide_id)

                hms, hms_thresh, coords = get_slide_region_level_heatmaps(
                    slide_id,
                    patch_model,
                    region_model,
                    region_dir,
                    output_dir_slide,
                    downscale=cfg.downscale,
                    cmap=light_jet,
                    save_to_disk=True,
                    granular=cfg.granular,
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
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        save_to_disk=True,
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
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            save_to_disk=True,
                        )
                        stitched_hms_thresh[f"Head {head_num}"] = stitched_hm

                if cfg.display:
                    display_stitched_heatmaps(
                        slide_path,
                        stitched_hms,
                        output_dir_slide,
                        fname=f"region",
                        display_patching=True,
                        region_dir=region_dir,
                        region_size=cfg.region_size,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                    )
                    if len(stitched_hms_thresh) > 0:
                        display_stitched_heatmaps(
                            slide_path,
                            stitched_hms_thresh,
                            output_dir_slide,
                            fname=f"region_thresh",
                            display_patching=True,
                            region_dir=region_dir,
                            region_size=cfg.region_size,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                        )

                hms, hms_thresh, coords = get_slide_patch_level_heatmaps(
                    slide_id,
                    patch_model,
                    region_model,
                    region_dir,
                    output_dir_slide,
                    downscale=cfg.downscale,
                    cmap=light_jet,
                    threshold=None,
                    save_to_disk=True,
                    granular=cfg.granular,
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
                        fname=f"patch_head_{head_num}",
                        downsample=cfg.downsample,
                        downscale=cfg.downscale,
                        save_to_disk=True,
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
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            save_to_disk=True,
                        )
                        stitched_hms_thresh[f"Head {head_num}"] = stitched_hm

                if cfg.display:
                    display_stitched_heatmaps(
                        slide_path,
                        stitched_hms,
                        output_dir_slide,
                        fname=f"patch",
                        display_patching=True,
                        region_dir=region_dir,
                        region_size=cfg.region_size,
                        downsample=cfg.downsample,
                        font_fp=cfg.font_fp,
                    )
                    if len(stitched_hms_thresh) > 0:
                        display_stitched_heatmaps(
                            slide_path,
                            stitched_hms_thresh,
                            output_dir_slide,
                            fname=f"patch_thresh",
                            display_patching=True,
                            region_dir=region_dir,
                            region_size=cfg.region_size,
                            downsample=cfg.downsample,
                            font_fp=cfg.font_fp,
                        )

                hms, hms_thresh, coords = get_slide_hierarchical_heatmaps(
                    slide_id,
                    patch_model,
                    region_model,
                    region_dir,
                    output_dir_slide,
                    downscale=cfg.downscale,
                    cmap=light_jet,
                    threshold=None,
                    save_to_disk=False,
                    granular=cfg.granular,
                    patch_device=device,
                    region_device=device,
                    main_process=is_main_process(),
                )

                stitched_hms = defaultdict(list)
                for rhead_num, hm_dict in hms.items():
                    coords_dict = coords[rhead_num]
                    for phead_num, heatmaps in hm_dict.items():
                        coordinates = coords_dict[phead_num]
                        stitched_hm = stitch_slide_heatmaps(
                            slide_path,
                            heatmaps,
                            coordinates,
                            output_dir_slide,
                            fname=f"hierarchcial_rhead_{rhead_num}_phead_{phead_num}",
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            save_to_disk=True,
                        )
                        stitched_hms[rhead_num].append(stitched_hm)
                
                stitched_hms_thresh = defaultdict(list)
                for rhead_num, hm_dict in hms_thresh.items():
                    coords_dict = coords[rhead_num]
                    for phead_num, heatmaps in hm_dict.items():
                        coordinates = coords_dict[phead_num]
                        stitched_hm = stitch_slide_heatmaps(
                            slide_path,
                            heatmaps,
                            coordinates,
                            output_dir_slide,
                            fname=f"hierarchcial_rhead_{rhead_num}_phead_{phead_num}_thresh",
                            downsample=cfg.downsample,
                            downscale=cfg.downscale,
                            save_to_disk=True,
                        )
                        stitched_hms_thresh[rhead_num].append(stitched_hm)

                if cfg.wandb.enable and not distributed:
                    wandb.log({"processed": i + 1})


if __name__ == "__main__":

    main()
