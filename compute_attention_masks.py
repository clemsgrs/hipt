import os
import tqdm
import time
import hydra
import datetime
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable, Dict, Optional, Any, Union
from omegaconf import DictConfig

from source.wsi import WholeSlideImage
from source.utils import initialize_wandb, compute_time


def prepare_df(df, region_dir):
    filtered_slide_ids = []
    for slide_id in df.slide_id:
        if Path(region_dir, f"{slide_id}").is_dir():
            filtered_slide_ids.append(slide_id)
    df_missing = df[~df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
    df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
    if len(df.slide_id) != len(df_filtered.slide_id):
        print(
            f"WARNING: {len(df.slide_id)-len(df_filtered.slide_id)} slides with no regions found!"
        )
    return df_filtered, df_missing


def split_into_blocks(arr, block_size):
    """
    Split a 2D array into smaller blocks of given block_size.
    """
    # split the array into sub-arrays of size block_size along the rows
    rows_split = np.split(arr, arr.shape[0] // block_size)
    # further split these sub-arrays along the columns
    blocks = [
        np.split(block, arr.shape[1] // block_size, axis=1) for block in rows_split
    ]
    # flatten the list of blocks
    flattened_blocks = [block for sublist in blocks for block in sublist]
    return np.array(flattened_blocks)


def compute_attention_mask_slide(
    slide_id: str,
    slide_fp: str,
    target_spacing: float,
    region_size: int,
    patch_size: int,
    mini_patch_size: int,
    region_dir: str,
    region_format: str = "jpg",
    segmentation_parameters: Dict[str, Union[bool, int]] = {},
    seg_mask_fp: Optional[str] = None,
    backend: str = "pyvips",
):
    # load the slide
    wsi_object = WholeSlideImage(Path(slide_fp), backend=backend)
    if seg_mask_fp:
        # load segmentation mask
        seg_level, seg_spacing = wsi_object.loadSegmentation(
            Path(seg_mask_fp),
            downsample=segmentation_parameters["downsample"],
        )
    else:
        segmentation_parameters["tissue_pixel_value"] = 1
        seg_level, seg_spacing = wsi_object.segmentTissue(
            downsample=segmentation_parameters["downsample"],
            sthresh=segmentation_parameters["sthresh"],
            mthresh=segmentation_parameters["mthresh"],
            close=segmentation_parameters["close"],
            use_otsu=segmentation_parameters["use_otsu"],
        )
    # scale coordinates from slide's level 0 to mask's level
    sx, sy = tuple(
        i / j
        for i, j in zip(wsi_object.wsi.shapes[seg_level], wsi_object.wsi.shapes[0])
    )
    # scale region size from target spacing to downsample spacing
    # we expect mask spacings to be a subset of slide spacings
    # the ratio should thus give an integer
    sr = round(
        wsi_object.wsi.get_real_spacing(seg_spacing)
        / wsi_object.wsi.get_real_spacing(target_spacing)
    )
    scaled_region_size = region_size // sr
    # scale patch_size and mini_patch_size
    scaled_patch_size = patch_size // sr
    scaled_mini_patch_size = mini_patch_size // sr
    # retrieve region's (x,y) coordinates
    # should appear in the same order as in the corresponding slide feature vector
    coordinates = sorted(
        [p.stem for p in Path(region_dir, slide_id, "imgs").glob(f"*.{region_format}")]
    )
    coordinates = [(int(p.split("_")[0]), int(p.split("_")[1])) for p in coordinates]
    tissue_pcts = []
    for x, y in coordinates:
        x_mask, y_mask = int(x * sx), int(y * sy)
        mask_region = np.zeros((scaled_region_size, scaled_region_size))
        # going from WholeSlideImage to numpy arrays: need to switch x & y axes
        sub_mask = wsi_object.binary_mask[
            y_mask : y_mask + scaled_region_size,
            x_mask : x_mask + scaled_region_size,
        ]
        mask_region[: sub_mask.shape[0], : sub_mask.shape[1]] = sub_mask

        # compute tissue percentage for each mini patch in each patch
        mask_patches = split_into_blocks(mask_region, scaled_patch_size)

        mask_mini_patches = []
        for p in mask_patches:
            mp = split_into_blocks(p, scaled_mini_patch_size)
            mask_mini_patches.append(mp)
        mask_mini_patches = np.stack(mask_mini_patches)
        tissue = mask_mini_patches == segmentation_parameters["tissue_pixel_value"]
        tissue_pct = np.sum(tissue, axis=(-2, -1)) / tissue[0][0].size
        tissue_pcts.append(tissue_pct)

    tissue_pcts = np.stack(tissue_pcts)  # (M, npatch**2, nminipatch**2)
    del wsi_object, mask_region, mask_patches, mask_mini_patches, tissue
    return tissue_pcts


def compute_attention_masks(
    df: pd.DataFrame,
    output_dir: str,
    target_spacing: float,
    region_size: int,
    patch_size: int,
    mini_patch_size: int,
    region_dir: str,
    region_format: str = "jpg",
    segmentation_parameters: Dict[str, Union[bool, int]] = {},
    backend: str = "pyvips",
    verbose: bool = True,
):
    if "segmentation_mask_path" in df.columns:
        id_path_mask = zip(
            df.slide_id.unique().tolist(),
            df.slide_path.unique().tolist(),
            df.segmentation_mask_path.unique().tolist(),
        )
    else:
        id_path_mask = zip(
            df.slide_id.unique().tolist(),
            df.slide_path.unique().tolist(),
            [None] * df.slide_id.nunique(),
        )
    with tqdm.tqdm(
        id_path_mask,
        desc=("Infering attention masks from tissue content"),
        unit=f" slide",
        total=df.slide_id.nunique(),
        leave=True,
        disable=(not verbose),
    ) as t:
        for (slide_id, slide_fp, seg_mask_fp) in t:
            mask = compute_attention_mask_slide(
                slide_id,
                slide_fp,
                target_spacing,
                region_size,
                patch_size,
                mini_patch_size,
                region_dir,
                region_format,
                segmentation_parameters,
                seg_mask_fp,
                backend,
            )
            # save the array
            save_path = Path(output_dir, f"{slide_id}.npy")
            np.save(save_path, mask)
            del mask


@hydra.main(
    version_base="1.2.0",
    config_path="config/misc",
    config_name="attention_masks",
)
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.data.csv)
    df_filtered, _ = prepare_df(df, cfg.data.region_dir)

    start_time = time.time()
    compute_attention_masks(
        df_filtered,
        output_dir,
        cfg.spacing,
        cfg.model.region_size,
        cfg.model.patch_size,
        cfg.model.mini_patch_size,
        cfg.data.region_dir,
        cfg.region_format,
        cfg.seg_params,
        cfg.backend,
        verbose=True,
    )
    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    main()
