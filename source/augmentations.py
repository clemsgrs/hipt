import os
import tqdm
import torch
import hydra
import random
import datetime
import subprocess
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from functools import partial
from omegaconf import DictConfig
from typing import Optional, Union, Dict
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors

from source.wsi import WholeSlideImage


def add_random_noise(
    feature, gamma: float = 0.5, mean: float = 0.0, std: float = 1.0, seed: int = 21
):
    torch.manual_seed(seed)
    noise = torch.normal(mean, std, size=feature.shape)
    augm_feature = feature + gamma * noise
    return augm_feature


def interpolate_feature(ref_feature, neighbor_feature, lmbda: float = 0.5):
    augm_feature = lmbda * (neighbor_feature - ref_feature) + ref_feature
    return augm_feature


def extrapolate_feature(ref_feature, neighbor_feature, lmbda: float = 0.5):
    augm_feature = lmbda * (ref_feature - neighbor_feature) + ref_feature
    return augm_feature


def load_feature(feature_path):
    f = torch.load(feature_path)
    return f


def get_knn_features(
    feature,
    slide_id: str,
    fname: str,
    region_df: pd.DataFrame,
    label_df: pd.DataFrame,
    output_dir: Path,
    level: str,
    label_name: str = "label",
    K: int = 10,
    sim_threshold: Optional[float] = None,
    region_features_dir: Optional[Path] = None,
    num_workers: int = 0,
    process_id: Optional[int] = None,
    augmentation_dir: Optional[Path] = None,
):
    label = label_df[label_df.slide_id == slide_id][label_name].values[0]
    df = pd.merge(
        region_df, label_df[["slide_id", label_name]], on="slide_id", how="inner"
    )
    # grab all samples be longing to same class
    in_class_df = df[df[label_name] == label].reset_index(drop=True)

    # load features
    features = []
    in_class_feature_paths = list(in_class_df.feature_path.unique())
    if region_features_dir:
        in_class_feature_paths = [
            Path(region_features_dir, Path(fp).name) for fp in in_class_feature_paths
        ]
    if augmentation_dir:
        stacked_features_path = Path(augmentation_dir, f"in_class_features_{label}.pt")
    else:
        stacked_features_path = Path(output_dir, f"in_class_features_{label}.pt")
    if stacked_features_path.is_file():
        stacked_features = torch.load(
            stacked_features_path
        )  # (nregion, 192) or (nregion*npatch, 384) with npatch = 256 (resp. 64, 16) if region size = 4096 (resp. 2048, 1024)
    else:
        # multi-cpu support
        if num_workers > 1:
            features = []
            with mp.Pool(num_workers) as pool:
                for i, r in enumerate(pool.map(load_feature, in_class_feature_paths)):
                    features.append(r)
        else:
            enable_pbar = (process_id == mp.current_process().pid)
            with tqdm.tqdm(
                in_class_feature_paths,
                desc=f"Loading in class features (label={label})",
                unit=" region",
                leave=False,
                # disable=not enable_pbar,
            ) as t:
                for fp in t:
                    f = torch.load(fp)
                    features.append(f)
        assert len(features) == len(in_class_feature_paths)
        stacked_features = torch.cat(features, dim=0)
        # save it for later epochs
        torch.save(
            stacked_features, stacked_features_path
        )  # (nregion, 192) or (nregion*npatch, 384) with npatch = 256 (resp. 64, 16) if region size = 4096 (resp. 2048, 1024)
    # nearest neighbor search doesn't have to be conducted every epoch
    # only on first epoch
    knn_output_dir = Path(output_dir, "knn")
    knn_output_dir.mkdir(exist_ok=True)
    knn_csv_path = Path(knn_output_dir, f"{slide_id}_{fname}.csv")
    if knn_csv_path.is_file():
        knn_df = pd.read_csv(knn_csv_path)
    else:
        knn = NearestNeighbors(
            n_neighbors=K + 1,
            metric="cosine",
        )
        knn.fit(stacked_features.numpy())
        # retrieve K nearest neighbors
        distances, indices = knn.kneighbors(
            feature.numpy().reshape(1, -1), return_distance=True
        )
        # drop the first result which corresponds to the input feature
        distances, indices = distances.squeeze()[1:], indices.squeeze()[1:].tolist()
        # cosine distance is defined as (1 - cosine_similarity)
        similarities = 1 - distances
        # optional thresholding
        if sim_threshold:
            idx = (similarities > sim_threshold).nonzero()[0]
            indices = indices[idx]
            similarities = similarities[idx]
        if level == "local":
            npatch = 64
            region_indices, patch_indices = [i // npatch for i in indices], [
                i % npatch for i in indices
            ]
            knn_df = pd.DataFrame(
                {
                    "region_idx": region_indices,
                    "patch_idx": patch_indices,
                    "knn_idx": indices,
                    "sim": similarities,
                }
            )
        else:
            knn_df = pd.DataFrame(
                {
                    "knn_idx": indices,
                    "sim": similarities,
                }
            )
        knn_df.to_csv(knn_csv_path, index=False)
    indices = knn_df.knn_idx.values.tolist()
    knn_features = stacked_features[indices]
    return knn_features, indices


def random_augmentation(
    features,
    gamma: float = 0.5,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 21,
    slide_id: Optional[str] = None,
):
    augm_features = add_random_noise(features, gamma, mean, std, seed=seed)
    return augm_features


def simple_augmentation(
    features,
    slide_id,
    region_df,
    label_df,
    level: str,
    method: str,
    label_name: str = "label",
    output_dir: Path = Path(""),
    K: int = 10,
    sim_threshold: Optional[float] = None,
    lmbda: float = 0.5,
    region_features_dir: Optional[Path] = None,
    num_workers: int = 0,
    seed: int = 21,
):
    if level == "global":
        # features = (M, 192)
        augm_features = []
        for region_idx, feature in enumerate(features):
            # feature = (192)
            fname = f"{region_idx}"
            knn_features, _ = get_knn_features(
                feature,
                slide_id,
                fname,
                region_df,
                label_df,
                output_dir,
                level,
                label_name,
                K,
                sim_threshold,
                region_features_dir,
                num_workers,
            )
            # pick a random neighbor, compute augmented feature
            random.seed(seed)
            i = random.randint(0, K - 1)
            neighbor_feature = knn_features[i]
            if method == "interpolation":
                augm_feature = interpolate_feature(
                    feature, neighbor_feature, lmbda=lmbda
                )
            elif method == "extrapolation":
                augm_feature = extrapolate_feature(
                    feature, neighbor_feature, lmbda=lmbda
                )
            else:
                raise KeyError(
                    f"provided method '{method}' not suported ; chose among ['interpolation', 'extrapolation']"
                )
            augm_features.append(augm_feature.unsqueeze(0))
        stacked_augm_features = torch.cat(augm_features, dim=0)
    elif level == "local":
        # features = (M, npatch, 384)
        augm_slide_features = []
        for region_idx, region_features in enumerate(features):
            # region_features = (npatch, 384)
            augm_patch_features = []
            for patch_idx, patch_feature in enumerate(region_features):
                # patch_feature = (384)
                fname = f"{region_idx}_{patch_idx}"
                knn_features, _ = get_knn_features(
                    patch_feature,
                    slide_id,
                    fname,
                    region_df,
                    label_df,
                    output_dir,
                    level,
                    label_name,
                    K,
                    sim_threshold,
                    region_features_dir,
                    num_workers,
                )
                # pick a random neighbor, compute augmented feature
                random.seed(seed)
                i = random.randint(0, K - 1)
                neighbor_feature = knn_features[i]
                if method == "interpolation":
                    augm_patch_feature = interpolate_feature(
                        patch_feature, neighbor_feature, lmbda=lmbda
                    )
                elif method == "extrapolation":
                    augm_patch_feature = extrapolate_feature(
                        patch_feature, neighbor_feature, lmbda=lmbda
                    )
                else:
                    raise KeyError(
                        f"provided method '{method}' not suported ; chose among ['interpolation', 'extrapolation']"
                    )
                augm_patch_features.append(
                    augm_patch_feature.unsqueeze(0)
                )  # [(1, 384)] of len npatch
            stacked_augm_patch_features = torch.cat(
                augm_patch_features, dim=0
            )  # (npatch, 384)
            augm_slide_features.append(
                stacked_augm_patch_features.unsqueeze(0)
            )  # [(1, npatch, 384)] of len M
        stacked_augm_features = torch.cat(
            augm_slide_features, dim=0
        )  # (M, npatch, 384)

    return stacked_augm_features


def plot_knn_features(
    feature,
    x,
    y,
    slide_id,
    region_df,
    label_df,
    label_name: str = "label",
    K: int = 10,
    sim_threshold: Optional[float] = None,
    region_dir: Optional[str] = None,
    slide_dir: Optional[str] = None,
    spacing: Optional[float] = None,
    backend: str = "openslide",
    size: int = 256,
    region_fmt: str = "jpg",
    dpi: int = 300,
):
    label = label_df[label_df.slide_id == slide_id][label_name].values[0]
    df = pd.merge(
        region_df, label_df[["slide_id", label_name]], on="slide_id", how="inner"
    )
    # grab all samples be longing to same class
    in_class_df = df[df[label_name] == label].reset_index(drop=True)
    fname = f"{x}_{y}"
    _, knn_indices = get_knn_features(
        feature,
        slide_id,
        region_df,
        label_df,
        label_name=label_name,
        level="gobal",
        K=K,
        sim_threshold=sim_threshold,
    )
    knn_df = in_class_df.loc[knn_indices]
    fig, ax = plt.subplots(1, K + 1, dpi=dpi)

    # get reference region
    if region_dir:
        fname = Path(region_dir, sid, "imgs", f"{x}_{y}.{region_fmt}")
        ref_region = Image.open(fname)
    elif slide_dir:
        slide_path = [x for x in slide_dir.glob(f"{sid}*")][0]
        wsi_object = WholeSlideImage(slide_path, backend=backend)
        ref_region = wsi_object.wsi.get_patch(x, y, ts, ts, spacing=spacing, center=False)
        ref_region = Image.fromarray(region).convert("RGB")
    else:
        raise ValueError(
            "neither 'region_dir' nor 'slide_dir' was given ; at least one of them must be given"
        )
    ref_region = ref_region.resize((size, size))
    ax[0].imshow(ref_region)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(f"reference", size="xx-small")

    # iterate over knn regions
    for i, (_, sid, ts, x, y, sim) in knn_df.iterrows():
        if region_dir:
            fname = Path(region_dir, sid, "imgs", f"{x}_{y}.jpg")
            region = Image.open(fname)
        elif slide_dir:
            slide_path = [x for x in slide_dir.glob(f"{sid}*")][0]
            wsi_object = WholeSlideImage(slide_path, backend=backend)
            region = wsi_object.wsi.get_patch(x, y, ts, ts, spacing=spacing, center=False)
            region = Image.fromarray(region).convert("RGB")
        region = region.resize((size, size))
        ax[i].imshow(region)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f"sim: {sim:.3f}", size="xx-small")
    plt.tight_layout()
    return fig


@dataclass
class AugmentationOptions:
    name: str
    output_dir: Path
    region_features_dir: Path
    region_df: pd.DataFrame
    label_df: pd.DataFrame
    level: str
    label_name: str = "label"
    multiprocessing: Optional[bool] = True
    kwargs: Dict[str, Union[str, float, int]] = field(default_factory=lambda: {})


class FeatureSpaceAugmentation:
    def __init__(self, options: DictConfig):
        self.name = options.name
        if self.name == "random":
            self.aug = partial(random_augmentation, **options.kwargs)
        elif self.name in ["interpolation", "extrapolation"]:
            self.aug = partial(
                simple_augmentation,
                region_df=options.region_df,
                label_df=options.label_df,
                label_name=options.label_name,
                level=options.level,
                method=self.name,
                output_dir=options.output_dir,
                region_features_dir=options.region_features_dir,
                multiprocessing=options.multiprocessing,
                **options.kwargs,
            )
        else:
            raise KeyError(
                f"'{self.name}' not supported ; please chose among ['random', 'interpolation', 'extrapolation']"
            )

    def __call__(self, features, slide_id, seed):
        augm_features = self.aug(features, slide_id=slide_id, seed=seed)
        return augm_features


def save_slide_knn_df(
    slide_id: str,
    features_dir: Path,
    region_df: pd.DataFrame,
    label_df: pd.DataFrame,
    output_dir: Path,
    level: str,
    label_name: str = "label",
    region_features_dir: Optional[Path] = None,
    K: int = 10,
    sim_threshold: Optional[float] = None,
    num_workers: int = 0,
    process_id: Optional[int] = None,
    augmentation_dir: Optional[Path] = None,
):
    fp = Path(features_dir, f"{slide_id}.pt")
    features = torch.load(fp)
    if level == "global":
        # feature = (M, 192)
        for region_idx, feature in enumerate(features):
            fname = f"{region_idx}"
            _, _ = get_knn_features(
                feature,
                slide_id,
                fname,
                region_df,
                label_df,
                output_dir,
                level,
                label_name,
                region_features_dir=region_features_dir,
                K=K,
                sim_threshold=sim_threshold,
                num_workers=num_workers,
                process_id=process_id,
                augmentation_dir=augmentation_dir,
            )
    elif level == "local":
        # features = (M, npatch, 384)
        for region_idx, region_features in enumerate(features):
            # region_features = (npatch, 384)
            for patch_idx, patch_feature in enumerate(region_features):
                # patch_feature = (384)
                fname = f"{region_idx}_{patch_idx}"
                _, _ = get_knn_features(
                    patch_feature,
                    slide_id,
                    fname,
                    region_df,
                    label_df,
                    output_dir,
                    level,
                    label_name,
                    region_features_dir=region_features_dir,
                    K=K,
                    sim_threshold=sim_threshold,
                    num_workers=num_workers,
                    process_id=process_id,
                    augmentation_dir=augmentation_dir,
                )
    return slide_id


@hydra.main(
    version_base="1.2.0",
    config_path="../config/augmentations",
    config_name="default",
)
def main(cfg: DictConfig):

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        run_id = wandb_run.id

    features_root_dir = Path(cfg.features_root_dir)
    # slide_features_dir = Path(features_root_dir, f"slide_features")
    slide_features_dir = Path(features_root_dir, f"slide")
    region_features_dir = None

    df = pd.read_csv(cfg.label_csv)
    region_df = pd.read_csv(Path(features_root_dir, "region_features.csv"))

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_slide_ids = []
    for slide_id in df.slide_id:
        if Path(slide_features_dir, f"{slide_id}.pt").is_file():
            filtered_slide_ids.append(slide_id)
    df_missing = df[~df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
    df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
    if len(df.slide_id) != len(df_filtered.slide_id):
        print(
            f"WARNING: {len(df.slide_id)-len(df_filtered.slide_id)} slides dropped because .pt files missing"
        )

    slide_ids = df_filtered.slide_id.unique().tolist()

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    if cfg.speed.multiprocessing:

        args = [
            (
                slide_id,
                slide_features_dir,
                region_df,
                df_filtered,
                output_dir,
                cfg.level,
                cfg.label_name,
                region_features_dir,
                cfg.K,
                cfg.threshold,
                0,
                None,
                cfg.augmentation_dir,
            )
            for sid in slide_ids
        ]

        processed_sids = []
        with mp.Pool(num_workers) as pool:
            # # get all active child processes
            # children = mp.active_children()
            # # get the pid of each child process
            # pids = [child.pid for child in children]
            # args = [e + (pids[0],) for e in args]
            for sid in pool.starmap(save_slide_knn_df, args):
                processed_sids.append(sid)
        if cfg.wandb.enable:
            p = len(processed_sids)
            wandb.log({"processed": p})

    else:

        with tqdm.tqdm(
            slide_ids,
            desc="Gathering knn in feature space",
            unit=" slide",
            leave=True,
        ) as t:

            for i, slide_id in enumerate(t):

                save_slide_knn_df(
                    slide_id,
                    slide_features_dir,
                    region_df,
                    df_filtered,
                    output_dir,
                    cfg.level,
                    cfg.label_name,
                    region_features_dir,
                    K=cfg.K,
                    sim_threshold=cfg.threshold,
                    num_workers=num_workers,
                    augmentation_dir=cfg.augmentation_dir,
                )

                if cfg.wandb.enable:
                    wandb.log({"processed": i + 1})


if __name__ == "__main__":

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    main()
