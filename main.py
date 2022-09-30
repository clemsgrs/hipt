
import tqdm
import torch
import torch.nn.functional as F
import hydra
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from models import HIPT_4096
from dataset import StackedTilesDataset
from utils import create_train_tune_test_df

@hydra.main(version_base='1.2.0', config_path='config', config_name='default')
def main(cfg):

    # hipt = HIPT_4096(
    #     size_arg=cfg.size,
    #     dropout=cfg.dropout,
    #     num_classes=cfg.num_classes,
    #     pretrain_256=cfg.pretrain_256,
    #     freeze_256=cfg.freeze_256,
    #     pretrain_4096=cfg.pretrain_4096,
    #     freeze_4096=cfg.freeze_4096,
    # )
    
    # x = [M, 3, 4096, 4096]
    # wsi_name = 'PANDA_C1_4251'
    # patch_4096_dir = Path(cfg.data_dir, 'patches', wsi_name, '4096', 'imgs')
    # patch_4096_list = list(patch_4096_dir.glob('*.png'))

    # M = len(patch_4096_list)
    # print(f'Found {M} [4096,4096] patches for slide {wsi_name}')
    # print('Loading patches...')

    # stacked_patches = np.zeros((M, 3, 4096, 4096))
    # with tqdm.tqdm(
    #     patch_4096_list,
    #     desc=(f'{wsi_name}'),
    #     unit=' patch',
    #     ncols=100,
    # ) as t:

    #     for i, path in enumerate(t):

    #         patch = Image.open(path)
    #         patch_arr = np.asarray(patch)
    #         patch_arr = np.moveaxis(patch_arr, -1, 0)
    #         patch_arr = patch_arr[np.newaxis, :]
    #         stacked_patches[i] = patch_arr
    
    # print(f'stacked_patches.shape: {stacked_patches.shape}')

    # logits = hipt(torch.from_numpy(stacked_patches))
    # print(f'logits.shape: {logits.shape}')
    # preds = F.softmax(logits, dim=1).detach().cpu()
    # y_hat = torch.topk(logits, 1, dim = 1)[1]
    # print(preds)
    # print(y_hat)

    if Path(cfg.data_dir, 'train.csv').exists() and Path(cfg.data_dir, 'tune.csv').exists() and Path(cfg.data_dir, 'test.csv').exists():
        train_df_path = Path(cfg.data_dir, 'train.csv')
        tune_df_path = Path(cfg.data_dir, 'tune.csv')
        test_df_path = Path(cfg.data_dir, 'test.csv')
        train_df = pd.read_csv(train_df_path)
        tune_df = pd.read_csv(tune_df_path)
        test_df = pd.read_csv(test_df_path)
    else:
        label_df_path = Path(cfg.data_dir, 'labels.csv')
        label_df = pd.read_csv(label_df_path)
        train_df, tune_df, test_df = create_train_tune_test_df(label_df, save_csv=True, output_dir=cfg.data_dir)
    
    tiles_dir = Path(cfg.data_dir, 'patches')
    train_dataset = StackedTilesDataset(train_df, tiles_dir, tile_size=4096)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=False)
    index, stacked_tiles, label = next(train_loader)
    print(stacked_tiles.shape)



if __name__ == '__main__':
	
    # python3 main.py
    main()



