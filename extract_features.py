import os
import tqdm
import wandb
import torch
import hydra
import shutil
import datetime
import pandas as pd
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from torchvision import transforms

from source.dataset import RegionFilepathsDataset
from source.models import GlobalFeatureExtractor, LocalFeatureExtractor
from source.utils import initialize_wandb, initialize_df, collate_region_filepaths


@hydra.main(
    version_base="1.2.0", config_path="config/feature_extraction", config_name="default"
)
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    slide_features_dir = Path(output_dir, "slide")
    region_features_dir = Path(output_dir, "region")
    if not cfg.resume:
        if output_dir.exists():
            print(f"{output_dir} already exists! deleting it...")
            shutil.rmtree(output_dir)
            print("done")
            output_dir.mkdir(parents=False)
            slide_features_dir.mkdir()
            if cfg.save_region_features:
                region_features_dir.mkdir()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            slide_features_dir.mkdir(exist_ok=True)
            if cfg.save_region_features:
                region_features_dir.mkdir(exist_ok=True)

    if cfg.level == "global":
        model = GlobalFeatureExtractor(
            region_size=cfg.region_size,
            patch_size=cfg.patch_size,
            mini_patch_size=cfg.mini_patch_size,
            pretrain_vit_patch=cfg.pretrain_vit_patch,
            pretrain_vit_region=cfg.pretrain_vit_region,
        )
    elif cfg.level == "local":
        model = LocalFeatureExtractor(
            patch_size=cfg.patch_size,
            mini_patch_size=cfg.mini_patch_size,
            pretrain_vit_patch=cfg.pretrain_vit_patch,
        )
    else:
        raise ValueError(f"cfg.level ({cfg.level}) not supported")

    region_dir = Path(cfg.region_dir)
    slide_ids = sorted([s.name for s in region_dir.iterdir()])
    print(f"{len(slide_ids)} slides with extracted patches found")

    if cfg.slide_list:
        with open(Path(cfg.slide_list), "r") as f:
            slide_ids = sorted([Path(x.strip()).stem for x in f.readlines()])
        print(f"restricting to {len(slide_ids)} slides from slide list .txt file")

    process_list_fp = None
    if (
        Path(output_dir, f"process_list_{cfg.level}.csv").is_file()
        and cfg.resume
    ):
        process_list_fp = Path(output_dir, f"process_list_{cfg.level}.csv")

    if process_list_fp is None:
        df = initialize_df(slide_ids)
    else:
        df = pd.read_csv(process_list_fp)

    mask = df["process"] == 1
    process_stack = df[mask]
    total = len(process_stack)
    already_processed = len(df) - total

    region_dataset = RegionFilepathsDataset(df, region_dir, cfg.format)
    region_subset = torch.utils.data.Subset(
        region_dataset, indices=process_stack.index.tolist()
    )
    loader = torch.utils.data.DataLoader(
        region_subset, batch_size=1, shuffle=False, collate_fn=collate_region_filepaths
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    slide_ids, region_slide_ids = [], []
    slide_feature_paths, region_feature_paths = [], []
    x_coords, y_coords = [], []

    with tqdm.tqdm(
        loader,
        desc="Slide Encoding",
        unit=" slide",
        initial=already_processed,
        total=total + already_processed,
        ncols=80,
        position=0,
        leave=True,
    ) as t1:

        with torch.no_grad():

            for i, batch in enumerate(t1):

                idx, region_fps, slide_id = batch
                slide_ids.append(slide_id)
                features = []

                with tqdm.tqdm(
                    region_fps,
                    desc=(f"{slide_id}"),
                    unit=" region",
                    ncols=80 + len(slide_id),
                    position=1,
                    leave=False,
                ) as t2:

                    for fp in t2:

                        x_y = Path(fp).stem
                        x, y = int(x_y.split('_')[0]), int(x_y.split('_')[1])
                        x_coords.append(x)
                        y_coords.append(y)
                        img = Image.open(fp)
                        img = transforms.functional.to_tensor(img)  # [3, 4096, 4096]
                        img = img.unsqueeze(0)  # [1, 3, 4096, 4096]
                        img = img.to(device, non_blocking=True)
                        feature = model(img)
                        if cfg.save_region_features:
                            save_path = Path(region_features_dir, f"{slide_id}_{x_y}.pt")
                            torch.save(feature, save_path)
                            region_feature_paths.append(save_path.resolve())
                            region_slide_ids.append(slide_id)
                        features.append(feature)

                stacked_features = torch.stack(features, dim=0).squeeze(1)
                save_path = Path(slide_features_dir, f"{slide_id}.pt")
                torch.save(stacked_features, save_path)
                slide_feature_paths.append(save_path.resolve())

                df.loc[idx, "process"] = 0
                df.loc[idx, "status"] = "processed"
                df.to_csv(
                    Path(output_dir, f"process_list_{cfg.level}.csv"),
                    index=False,
                )

                if cfg.wandb.enable:
                    wandb.log({"processed": already_processed + i + 1})

    slide_features_df = pd.DataFrame.from_dict({
        'feature_path': slide_feature_paths,
        'slide_id': slide_ids,
        'level': [f'{cfg.level}']*len(slide_ids),
        'tile_size': [cfg.region_size]*len(slide_ids),
    })
    slide_features_df.to_csv(Path(output_dir, 'slide_features.csv'), index=False)
    if cfg.save_region_features:
        region_features_df = pd.DataFrame.from_dict({
            'feature_path': region_feature_paths,
            'slide_id': region_slide_ids,
            'level': [f'{cfg.level}']*len(region_slide_ids),
            'tile_size': [cfg.region_size]*len(region_slide_ids),
            'x': x_coords,
            'y': y_coords,
        })
        region_features_df.to_csv(Path(output_dir, 'region_features.csv'), index=False)


if __name__ == "__main__":

    main()
