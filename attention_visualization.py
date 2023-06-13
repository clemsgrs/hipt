import hydra
import torch
import matplotlib

from PIL import Image
from pathlib import Path
from omegaconf import DictConfig

from source.attention_visualization_utils import (
    cmap_map,
    get_patch_model,
    get_region_model,
    create_patch_heatmaps_indiv,
    create_patch_heatmaps_concat,
    create_hierarchical_heatmaps_indiv,
    create_hierarchical_heatmaps_concat,
)


@hydra.main(
    version_base="1.2.0", config_path="../config/heatmaps", config_name="default"
)
def main(cfg: DictConfig):

    patch = Image.open(cfg.patch_fp)
    region = Image.open(cfg.region_fp)

    patch_device = torch.device("cuda:0")
    region_device = torch.device("cuda:0")

    patch_weights = Path(cfg.patch_weights)
    patch_model = get_patch_model(pretrained_weights=patch_weights, device=patch_device)

    region_weights = Path(cfg.region_weights)
    region_model = get_region_model(
        pretrained_weights=region_weights, device=region_device
    )

    light_jet = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.jet)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

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
        patch_device=patch_device,
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
        patch_device=patch_device,
    )
    print("done!")

    print(f"Computing individual region-level attention heatmaps")
    output_dir_region = Path(output_dir, "region_indiv")
    output_dir_region.mkdir(exist_ok=True, parents=True)
    create_hierarchical_heatmaps_indiv(
        region,
        patch_model,
        region_model,
        output_dir_region,
        scale=2,
        threshold=0.5,
        alpha=0.5,
        cmap=light_jet,
        patch_device=patch_device,
        region_device=region_device,
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
        scale=2,
        alpha=0.5,
        cmap=light_jet,
        patch_device=patch_device,
        region_device=region_device,
    )
    print("done!")


if __name__ == "__main__":

    main()
