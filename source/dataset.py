import tqdm
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Callable, Dict, Optional
from collections import defaultdict


def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)


class ExtractedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        label_name: str = 'label',
        label_mapping: Dict[int,int] = {},
        label_encoding: Optional[str] = None,
    ):
        self.features_dir = features_dir
        self.label_name = label_name
        self.label_mapping = label_mapping
        self.label_encoding = label_encoding

        self.df = self.prepare_data(df)

        self.num_classes = len(self.df.label.value_counts(dropna=True))
        self.map_class_to_slide_ids()

    def prepare_data(self, df):
        if self.label_mapping:
            df['label'] = df[self.label_name].apply(lambda x: self.label_mapping[x])
        elif self.label_name != 'label':
            df['label'] = df[self.label_name]
        filtered_slide_ids = []
        for slide_id in df.slide_id:
            if Path(self.features_dir, f'{slide_id}.pt').is_file():
                filtered_slide_ids.append(slide_id)
        df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        return df_filtered

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
        if self.label_encoding == 'ordinal':
            label = [1]*(label+1) + [0]*(self.num_classes-label-1)

        return idx, features, label

    def __len__(self):
        return len(self.df)


class StackedRegionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        region_dir: Path,
        region_size: int = 256,
        fmt: str = 'jpg',
        label_name: str = 'label',
        label_mapping: Dict[int,int] = {},
        transform: Callable = None,
        M_max: int = -1,
    ):
        self.region_dir = region_dir
        self.region_size = region_size
        self.format = fmt
        self.label_name = label_name
        self.label_mapping = label_mapping
        self.transform = transform
        self.M_max = M_max

        self.df = self.prepare_data(df)

        self.num_classes = len(self.df.label.value_counts(dropna=True))
        self.map_class_to_slide_ids()

    def prepare_data(self, df):
        if self.label_mapping:
            df['label'] = df[self.label_name].apply(lambda x: self.label_mapping[x])
        elif self.label_name != 'label':
            df['label'] = df[self.label_name]
        return df

    def map_class_to_slide_ids(self):
        # map each class to corresponding slide ids
        self.class_2_id = defaultdict(list)
        for i in range(self.num_classes):
            self.class_2_id[i] = np.asarray(self.df[self.label_name] == i).nonzero()[0]

    def get_label(self, idx):
        return self.df[self.label_name][idx]

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        region_dir = Path(self.region_dir, slide_id, str(self.region_size), self.format)
        regions_fp = [fp for fp in region_dir.glob(f'*.{self.format}')]
        M = len(regions_fp)
        if self.M_max > -1 and M > self.M_max:
            regions_fp = random.sample([x for x in regions_fp], self.M_max)
            M = self.M_max

        stacked_regions = torch.zeros((M, 3, self.region_size, self.region_size))

        with tqdm.tqdm(
            regions_fp,
            desc=(f'{slide_id}'),
            unit=' tiles',
            ncols=80,
            position=2,
            leave=False,
            disable=True) as t:

            for i, fp in enumerate(t):

                img = Image.open(fp)
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.functional.to_tensor(img)
                img = img.unsqueeze(0)
                stacked_regions[i] = img

            label = row[self.label_name]

            return idx, stacked_regions, label

    def __len__(self):
        return len(self.df)


class RegionFilepathsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        region_root_dir: Path,
        region_size: int,
        fmt: str,
    ):
        self.df = df
        self.region_root_dir = region_root_dir
        self.region_size = region_size
        self.format = fmt

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        region_dir = Path(self.region_root_dir, slide_id, str(self.region_size), self.format)
        regions = [str(fp) for fp in region_dir.glob(f'*.{self.format}')]
        return idx, regions

    def __len__(self):
        return len(self.df)