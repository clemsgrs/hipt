
import tqdm
import torch
import hydra
from PIL import Image
from pathlib import Path
from torchvision import transforms

from models import GlobalFeatureExtractor, LocalFeatureExtractor

@hydra.main(version_base='1.2.0', config_path='config', config_name='feature_extraction')
def main(cfg):

    output_dir = Path(cfg.output_dir)
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

    patch_dir = Path(cfg.data_dir, 'patches')
    slide_list = [s.stem for s in patch_dir.iterdir()]

    with tqdm.tqdm(
            slide_list,
            desc=(f'Slide Encoding'),
            unit=' slides',
            ncols=80,
            position=1,
            leave=True) as t1:

            for slide_id in t1:

                slide_patch_dir = Path(patch_dir, slide_id, str(cfg.region_size), 'imgs')
                tiles = list(slide_patch_dir.glob('*.png'))

                M = len(tiles)
                features = []

                with tqdm.tqdm(
                    tiles,
                    desc=(f'{slide_id}'),
                    unit=' tiles',
                    ncols=80,
                    position=2,
                    leave=False) as t2:

                    for i, fp in enumerate(t2):

                        tile = Image.open(fp)
                        tile = transforms.functional.to_tensor(tile)
                        tile = tile.unsqueeze(0)
                        feature = model(tile)
                        features.append(feature)

                stacked_features = torch.stack(features, dim=0)
                save_path = Path(cfg.output_dir, 'features', cfg.level, f'{slide_id}.pt')
                save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(stacked_features, save_path)



if __name__ == '__main__':

    # python3 extract_features.py
    main()



