
import tqdm
import hydra
import numpy as np
from PIL import Image
from pathlib import Path

from models import HIPT_4096
from utils import extract_coord_from_path

@hydra.main(version_base='1.2.0', config_path='config', config_name='default')
def main(cfg):

    hipt = HIPT_4096(
        size_arg=cfg.size,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
        pretrain_256=cfg.pretrain_256,
        freeze_256=cfg.freeze_256,
        pretrain_4096=cfg.pretrain_4096,
        freeze_4096=cfg.freeze_256,
    )
    
    # x = [M, 3, 4096, 4096]
    wsi_name = 'PANDA_C1_4251'
    patch_4096_dir = Path(cfg.data_dir, 'patches', wsi_name, '4096', 'imgs')
    patch_4096_list = list(patch_4096_dir.glob('*.png'))

    M = len(patch_4096_list)
    print(f'Found {M} [4096,4096] patches for slide {wsi_name}')
    print('Loading patches...')

    stacked_patches = np.zeros((M, 3, 4096, 4096))
    with tqdm.tqdm(
        patch_4096_list,
        desc=(f'{wsi_name}'),
        unit=' patch',
        ncols=100,
    ) as t:

        for i, path in enumerate(t):

            x, y = extract_coord_from_path(path)
            patch = Image.open(path)
            patch_arr = np.asarray(patch)
            patch_arr = np.moveaxis(patch_arr, -1, 0)
            patch_arr = patch_arr[np.newaxis, :]
            stacked_patches[i] = patch_arr
    
    print(f'stacked_patches.shape: {stacked_patches.shape}')

if __name__ == '__main__':
	
    # python3 main.py
    main()



