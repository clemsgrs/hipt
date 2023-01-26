import os
import tqdm
import wandb
import torch
import hydra
import shutil
import pandas as pd
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from torchvision import transforms

from source.dataset import RegionFilepathsDataset
from source.models import LocalFeatureExtractor
from source.utils import initialize_wandb, initialize_df, collate_region_filepaths


@hydra.main(
    version_base="1.2.0", config_path="../config/pre-training", config_name="extract_features"
)
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir, cfg.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(output_dir, "features")
    if not cfg.resume:
        if features_dir.exists():
            print(f"{features_dir} already exists! deleting it...")
            shutil.rmtree(features_dir)
            print("done")
            features_dir.mkdir(parents=False)
        else:
            features_dir.mkdir(parents=True, exist_ok=True)

    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("processed", summary="max")

    model = LocalFeatureExtractor(
        patch_size=cfg.patch_size,
        mini_patch_size=cfg.mini_patch_size,
        pretrain_vit_patch=cfg.pretrain_vit_patch,
    )

    region_dir = Path(cfg.region_dir)
    slide_ids = sorted([s.name for s in region_dir.iterdir()])
    print(f"{len(slide_ids)} slides with extracted patches found")

    if cfg.slide_list:
        with open(Path(cfg.slide_list), "r") as f:
            slide_ids = sorted([x.strip() for x in f.readlines()])
        print(f"restricting to {len(slide_ids)} slides from slide list .txt file")

    process_list_fp = None
    if (
        Path(features_dir.parent, f"process_list.csv").is_file()
        and cfg.resume
    ):
        process_list_fp = Path(output_dir, "features", f"process_list.csv")

    if process_list_fp is None:
        df = initialize_df(slide_ids)
    else:
        df = pd.read_csv(process_list_fp)

    mask = df["process"] == 1
    process_stack = df[mask]
    total = len(process_stack)
    already_processed = len(df) - total

    region_dataset = RegionFilepathsDataset(df, region_dir, cfg.region_size, cfg.format)
    region_subset = torch.utils.data.Subset(
        region_dataset, indices=process_stack.index.tolist()
    )
    loader = torch.utils.data.DataLoader(
        region_subset, batch_size=1, shuffle=False, collate_fn=collate_region_filepaths
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()

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

                idx, region_fps = batch
                slide_id = process_stack.loc[idx.item(), "slide_id"]

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
                        img = Image.open(fp)
                        img = transforms.functional.to_tensor(img)  # [3, 4096, 4096]
                        img = img.unsqueeze(0)  # [1, 3, 4096, 4096]
                        img = img.to(device, non_blocking=True)
                        feature = model(img)    # [256, 384]
                        save_path = Path(features_dir, f"{slide_id}_{x_y}.pt")
                        torch.save(feature, save_path)

                df.loc[idx, "process"] = 0
                df.loc[idx, "status"] = "processed"
                df.to_csv(
                    Path(features_dir.parent, f"process_list.csv"),
                    index=False,
                )

                if cfg.wandb.enable:
                    wandb.log({"processed": already_processed + i + 1})


if __name__ == "__main__":

    main()
