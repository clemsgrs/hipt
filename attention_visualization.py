import hydra
import torch
import matplotlib
import random
import numpy as np

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
    get_slide_patch_level_heatmaps,
    get_slide_region_level_heatmaps,
    get_slide_hierarchical_heatmaps,
    stitch_slide_heatmaps,
)


@hydra.main(
    version_base="1.2.0", config_path="config/heatmaps", config_name="default"
)
def main(cfg: DictConfig):

    # patch = Image.open(cfg.patch_fp)
    # region = Image.open(cfg.region_fp)

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

    # print(f"Computing indiviudal patch-level attention heatmaps")
    # output_dir_patch = Path(output_dir, "patch_indiv")
    # output_dir_patch.mkdir(exist_ok=True, parents=True)
    # create_patch_heatmaps_indiv(
    #     patch,
    #     patch_model,
    #     output_dir_patch,
    #     threshold=0.5,
    #     alpha=0.5,
    #     cmap=light_jet,
    #     patch_device=patch_device,
    # )
    # print("done!")

    # print(f"Computing concatenated patch-level attention heatmaps")
    # output_dir_patch_concat = Path(output_dir, "patch_concat")
    # output_dir_patch_concat.mkdir(exist_ok=True, parents=True)
    # create_patch_heatmaps_concat(
    #     patch,
    #     patch_model,
    #     output_dir_patch_concat,
    #     threshold=0.5,
    #     alpha=0.5,
    #     cmap=light_jet,
    #     patch_device=patch_device,
    # )
    # print("done!")

    # print(f"Computing individual region-level attention heatmaps")
    # output_dir_region = Path(output_dir, "region_indiv")
    # output_dir_region.mkdir(exist_ok=True, parents=True)
    # create_hierarchical_heatmaps_indiv(
    #     region,
    #     patch_model,
    #     region_model,
    #     output_dir_region,
    #     scale=2,
    #     threshold=0.5,
    #     alpha=0.5,
    #     cmap=light_jet,
    #     patch_device=patch_device,
    #     region_device=region_device,
    # )
    # print("done!")

    # print(f"Computing concatenated region-level attention heatmaps")
    # output_dir_region_concat = Path(output_dir, "region_concat")
    # output_dir_region_concat.mkdir(exist_ok=True, parents=True)
    # create_hierarchical_heatmaps_concat(
    #     region,
    #     patch_model,
    #     region_model,
    #     output_dir_region_concat,
    #     scale=2,
    #     alpha=0.5,
    #     cmap=light_jet,
    #     patch_device=patch_device,
    #     region_device=region_device,
    # )
    # print("done!")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    slide_path = Path(cfg.slide_fp)
    slide_id = slide_path.stem
    region_dir = Path(cfg.region_dir)

    output_dir_slide = Path(output_dir, slide_id)

    # hms, coords = get_slide_region_level_heatmaps(
    #     slide_id,
    #     patch_model,
    #     region_model,
    #     region_dir,
    #     output_dir_slide,
    #     scale=2,
    #     cmap=light_jet,
    #     save_to_disk=False,
    #     patch_device=torch.device("cuda:0"),
    #     region_device=torch.device("cuda:0"),
    # )
    # for head_num, heatmaps in hms.items():
    #     stitched_hm = stitch_slide_heatmaps(
    #         slide_path,
    #         heatmaps,
    #         coords[head_num],
    #         output_dir_slide,
    #         fname=f"region_head_{head_num}",
    #         downsample=32,
    #         scale=2,
    #         save_to_disk=True,
    #     )

    # hms, coords = get_slide_patch_level_heatmaps(
    #     slide_id,
    #     patch_model,
    #     region_model,
    #     region_dir,
    #     output_dir_slide,
    #     scale=2,
    #     cmap=light_jet,
    #     threshold=None,
    #     save_to_disk=False,
    #     patch_device=torch.device("cuda:0"),
    #     region_device=torch.device("cuda:0"),
    # )
    # for head_num, heatmaps in hms.items():
    #     stitched_hm = stitch_slide_heatmaps(
    #         slide_path,
    #         heatmaps,
    #         coords[head_num],
    #         output_dir_slide,
    #         fname=f"patch_head_{head_num}",
    #         downsample=32,
    #         scale=2,
    #         save_to_disk=True,
    #     )

    hms, coords = get_slide_hierarchical_heatmaps(
        slide_id,
        patch_model,
        region_model,
        region_dir,
        output_dir_slide,
        scale=2,
        cmap=light_jet,
        threshold=None,
        save_to_disk=False,
        patch_device=torch.device("cuda:0"),
        region_device=torch.device("cuda:0"),
    )
    for rhead_num, hm_dict in hms.items():
        coords_dict = coords[rhead_num]
        for phead_num, heatmaps in hm_dict.items():
            coordinates = coords_dict[phead_num]
            _ = stitch_slide_heatmaps(
                slide_path,
                heatmaps,
                coordinates,
                output_dir_slide,
                fname=f"hierarchcial_rhead_{rhead_num}_phead_{phead_num}",
                downsample=32,
                scale=2,
                save_to_disk=True,
            )

if __name__ == "__main__":

    main()
