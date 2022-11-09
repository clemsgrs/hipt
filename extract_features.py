import os
import sys
import tqdm
import wandb
import torch
import hydra
import shutil
import pandas as pd
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from source.models import GlobalFeatureExtractor, LocalFeatureExtractor
from source.utils import initialize_wandb, initialize_df


@hydra.main(version_base='1.2.0', config_path='config', config_name='feature_extraction')
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(output_dir, 'features', cfg.level)
    if not cfg.resume:
        if features_dir.exists():
            shutil.rmtree(features_dir)
            features_dir.mkdir(parents=False)
        else:
            features_dir.mkdir(parents=False, exist_ok=True)

    # set up wandb
    key = os.environ.get('WANDB_API_KEY')
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_run = initialize_wandb(project=cfg.wandb.project, exp_name=cfg.wandb.exp_name, entity=cfg.wandb.username, config=config, key=key)
    wandb_run.define_metric('processed', summary='max')

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
    slide_ids = sorted([s.name for s in patch_dir.iterdir()])

    if cfg.slide_list:
        with open(Path(cfg.slide_list), 'r') as f:
            slide_ids = sorted([Path(x.strip()).stem for x in f.readlines()])

    process_list_fp = None
    if Path(output_dir, 'features', f'process_list_{cfg.level}.csv').is_file() and cfg.resume:
        process_list_fp = Path(output_dir, 'features', f'process_list_{cfg.level}.csv')

    if process_list_fp is None:
        df = initialize_df(slide_ids)
    else:
        df = pd.read_csv(process_list_fp)

    mask = df['process'] == 1
    process_stack = df[mask]
    total = len(process_stack)
    already_processed = len(df)-total

    print()

    tqdm_output_fp = None
    tqdm_output_fp = Path(f'tqdm_{wandb.run.id}.log')
    tqdm_output_fp.unlink(missing_ok=True)
    tqdm_file = open(tqdm_output_fp, 'a+') if tqdm_output_fp is not None else sys.stderr

    with tqdm.tqdm(
        range(total),
        desc=(f'Slide Encoding'),
        unit=' slide',
        initial=already_processed,
        total=total+already_processed,
        ncols=80,
        position=0,
        leave=True,
        file=tqdm_file) as t1:

            for i in t1:

                idx = process_stack.index[i]
                slide_id = process_stack.loc[idx, 'slide_id']

                slide_patch_dir = Path(patch_dir, slide_id, str(cfg.region_size), cfg.format)
                tiles = [t for t in slide_patch_dir.glob(f'*.{cfg.format}')]

                M = len(tiles)
                features = []

                with tqdm.tqdm(
                    tiles,
                    desc=(f'{slide_id}'),
                    unit=' tiles',
                    ncols=80+len(slide_id),
                    position=1,
                    leave=False,
                    file=tqdm_file) as t2:

                    for fp in t2:

                        tile = Image.open(fp)
                        tile = transforms.functional.to_tensor(tile)
                        tile = tile.unsqueeze(0)
                        feature = model(tile)
                        features.append(feature)

                stacked_features = torch.stack(features, dim=0).squeeze(1)
                save_path = Path(features_dir, f'{slide_id}.pt')
                torch.save(stacked_features, save_path)

                df.loc[idx, 'process'] = 0
                df.loc[idx, 'status'] = 'processed'
                df.to_csv(Path(output_dir, 'features', f'process_list_{cfg.level}.csv'), index=False)

                wandb.log({'processed': already_processed+i+1})

    tqdm_file.close()
    tqdm_output_fp.unlink()



if __name__ == '__main__':

    # python3 extract_features.py
    main()

