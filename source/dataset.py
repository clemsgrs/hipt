import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Callable
from collections import defaultdict


def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)


class StackedTilesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tiles_dir: Path,
        tile_size: int = 256,
        training: bool = True,
        transform: Callable = None
    ):
        self.df = df
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.training = training
        self.transform = transform

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.id
        tiles_dir = Path(self.tiles_dir, slide_id, str(self.tile_size), 'imgs')
        tiles_list = list(tiles_dir.glob('*.png'))
        M = len(tiles_list)

        stacked_tiles = torch.zeros((M, 3, self.tile_size, self.tile_size))

        with tqdm.tqdm(
            tiles_list,
            desc=(f'{slide_id}'),
            unit=' tiles',
            ncols=80,
            position=2,
            leave=False) as t:

            for i, tp in enumerate(t):

                tile = Image.open(tp)
                if self.transform:
                    tile = self.transform(tile)
                else:
                    tile = transforms.functional.to_tensor(tile)
                tile = tile.unsqueeze(0)
                stacked_tiles[i] = tile

            label = np.array([row.label]).astype(float) if self.training else np.array([-1])

            return idx, stacked_tiles, label

    def __len__(self):
        return len(self.df)


class ExtractedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_root_dir: Path,
        training: bool = True,
        transform: Callable = None
    ):
        self.df = df
        self.features_root_dir = features_root_dir
        self.training = training
        self.transform = transform

        self.num_classes = len(self.df.label.value_counts(dropna=True))
        self.map_class_to_slide_ids()

    def map_class_to_slide_ids(self):
        # map each class to corresponding slide ids
        self.class_2_id = defaultdict(list)
        for i in range(self.num_classes):
            self.class_2_id[i] = np.asarray(self.df.label == i).nonzero()[0]

    def get_label(self, idx):
        return self.df.label[idx]

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        fp = Path(self.features_dir, f'{slide_id}.pt')
        features = torch.load(fp)

        label = row.label

        return idx, features, label

    def __len__(self):
        return len(self.df)