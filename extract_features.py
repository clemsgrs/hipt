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

from source.dataset import RegionDataset
from source.models import GlobalFeatureExtractor, LocalFeatureExtractor
from source.utils import initialize_wandb, initialize_df, collate_regions


@hydra.main(version_base='1.2.0', config_path='config', config_name='feature_extraction')
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(output_dir, 'features', cfg.level)
    if not cfg.resume:
        if features_dir.exists():
            print(f'{features_dir} already exists! deleting it...')
            shutil.rmtree(features_dir)
            print('done')
            features_dir.mkdir(parents=False)
        else:
            features_dir.mkdir(parents=True, exist_ok=True)

    # set up wandb
    key = os.environ.get('WANDB_API_KEY')
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _ = initialize_wandb(project=cfg.wandb.project, exp_name=cfg.wandb.exp_name, entity=cfg.wandb.username, config=config, key=key)
    wandb.define_metric('processed', summary='max')

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

    region_dir = Path(cfg.data_dir, cfg.dataset_name, 'patches')
    if cfg.region_dir:
        region_dir = Path(cfg.region_dir)
    slide_ids = sorted([s.name for s in region_dir.iterdir()])

    if cfg.slide_list:
        with open(Path(cfg.slide_list), 'r') as f:
            slide_ids = sorted([x.strip() for x in f.readlines()])

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

    label_df = pd.read_csv(cfg.data_csv)[['slide_id', 'label']]
    df = df.merge(label_df, how='left', on='slide_id')

    stacked_region_dataset = RegionDataset(df, region_dir, cfg.region_size, cfg.format)
    stacked_region_subset = torch.utils.data.Subset(stacked_region_dataset, indices=process_stack.index.tolist())
    loader = torch.utils.data.DataLoader(stacked_region_subset, batch_size=1, shuffle=False, collate_fn=collate_regions)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()

    tqdm_output_fp = None
    tqdm_output_fp = Path(f'tqdm_{wandb.run.id}.log')
    tqdm_output_fp.unlink(missing_ok=True)
    tqdm_file = open(tqdm_output_fp, 'a+') if tqdm_output_fp is not None else sys.stderr

    with tqdm.tqdm(
        loader,
        desc='Slide Encoding',
        unit=' slide',
        initial=already_processed,
        total=total+already_processed,
        ncols=80,
        position=0,
        leave=True,
        file=tqdm_file) as t1:

            for i, batch in enumerate(t1):

                idx, region_fps, _ = batch
                region_fps = region_fps[0]
                slide_id = process_stack.loc[idx.item(), 'slide_id']

                features = []

                with tqdm.tqdm(
                    region_fps,
                    desc=(f'{slide_id}'),
                    unit=' tiles',
                    ncols=80+len(slide_id),
                    position=1,
                    leave=False,
                    file=tqdm_file) as t2:

                    for fp in t2:

                        img = Image.open(fp)
                        img = transforms.functional.to_tensor(img)      # [3, 4096, 4096]
                        img = img.unsqueeze(0)                          # [1, 3, 4096, 4096]
                        img = img.to(device, non_blocking=True)
                        feature = model(img)
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

