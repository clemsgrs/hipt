import tqdm
import torch
import hydra
from PIL import Image
from pathlib import Path
from torchvision import transforms

from source.models import GlobalFeatureExtractor, LocalFeatureExtractor

@hydra.main(version_base='1.2.0', config_path='config', config_name='feature_extraction')
def main(cfg):

    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.level == 'global':
        model = GlobalFeatureExtractor(
            pretrain_256=cfg.pretrain_256,
            pretrain_4096=cfg.pretrain_4096,
        )
    elif cfg.level == 'local':
        model = LocalFeatureExtractor(
            pretrain_256=cfg.pretrain_256,
        )
    else:
        raise ValueError(f'cfg.level ({cfg.level} not supported')

    patch_dir = Path(cfg.data_dir, cfg.dataset_name, 'patches')
    slide_list = [s.stem for s in patch_dir.iterdir()]

    if Path(cfg.slide_list).is_file():
        with open(Path(cfg.slide_list), 'r') as f:
            slide_list = [Path(x.strip()).stem for x in f.readlines()]

    with tqdm.tqdm(
            slide_list,
            desc=(f'Slide Encoding'),
            unit=' slides',
            ncols=80,
            position=0,
            leave=True) as t1:

            for slide_id in t1:

                slide_patch_dir = Path(patch_dir, slide_id, str(cfg.region_size), cfg.format)
                tiles = [t for t in slide_patch_dir.glob(f'*.{cfg.format}')][:3]

                M = len(tiles)
                features = []

                with tqdm.tqdm(
                    tiles,
                    desc=(f'{slide_id}'),
                    unit=' tiles',
                    ncols=80+len(slide_id),
                    position=1,
                    leave=False) as t2:

                    for i, fp in enumerate(t2):

                        tile = Image.open(fp)
                        tile = transforms.functional.to_tensor(tile)
                        tile = tile.unsqueeze(0)
                        feature = model(tile)
                        features.append(feature)

                stacked_features = torch.stack(features, dim=0)
                save_path = Path(output_dir, 'features', cfg.level, f'{slide_id}.pt')
                save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(stacked_features, save_path)



if __name__ == '__main__':

    # python3 extract_features.py
    main()



