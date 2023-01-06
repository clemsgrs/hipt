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
        label_name: str = "label",
        label_mapping: Dict[int, int] = {},
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
            df["label"] = df[self.label_name].apply(lambda x: self.label_mapping[x])
        elif self.label_name != "label":
            df["label"] = df[self.label_name]
        filtered_slide_ids = []
        for slide_id in df.slide_id:
            if Path(self.features_dir, f"{slide_id}.pt").is_file():
                filtered_slide_ids.append(slide_id)
        df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        if len(df.slide_id) != len(df_filtered.slide_id):
            print(f"WARNING: {len(df.slide_id)-len(df_filtered.slide_id)} slides dropped because missing on disk")
        return df_filtered

    def map_class_to_slide_ids(self):
        # map each class to corresponding slide ids
        self.class_2_id = defaultdict(list)
        for i in range(self.num_classes):
            class_idxs = np.asarray(self.df.label == i).nonzero()[0]
            #TODO: make sure to add a drop_duplicates(['slide_id']) somewhere before
            self.class_2_id[i] = self.df.loc[class_idxs, 'slide_id'].values.tolist()

    def get_label(self, idx):
        return self.df.label[idx]

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        fp = Path(self.features_dir, f"{slide_id}.pt")
        features = torch.load(fp)

        label = row.label
        if self.label_encoding == "ordinal":
            label = [1] * (label + 1) + [0] * (self.num_classes - label - 1)

        return idx, features, label

    def __len__(self):
        return len(self.df)


class ExtractedFeaturesSurvivalSlideLevelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
        nbins: int = 4,
        eps: float = 1e-6,
    ):

        self.features_dir = features_dir
        self.label_name = label_name
        self.nbins = nbins
        self.eps = eps

        if "IDC" in df['oncotree_code'].values:
            # must be BRCA (and if so, use only IDCs)
            df = df[df['oncotree_code'] == 'IDC']

        patient_df, slide_df = self.prepare_data(df)
        self.patient_df, self.slide_df = self.filter_df(patient_df), self.filter_df(slide_df)

        self.map_class_to_slide_ids()
        self.map_class_to_patient_ids()

    def prepare_data(self, df):

        patient_df = df.drop_duplicates(['case_id']).copy()
        uncensored_df = patient_df[patient_df['censorship'] < 1]

        _, q_bins = pd.qcut(uncensored_df[self.label_name], q=self.nbins, retbins=True, labels=False)
        q_bins[-1] = df[self.label_name].max() + self.eps
        q_bins[0] = df[self.label_name].min() - self.eps

        disc_labels, q_bins = pd.cut(patient_df[self.label_name], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patient_df.insert(2, "disc_label", disc_labels.values.astype(int))

        self.patient_id_2_slide_id = defaultdict(list)
        for patient_id in patient_df['case_id']:
            slide_ids = df[df['case_id'] == patient_id]['slide_id'].values.tolist()
            self.patient_id_2_slide_id[patient_id] = slide_ids

        self.label_dict = {}
        label_count = 0
        for label in range(len(q_bins)-1):
            for censorship in [0, 1]:
                self.label_dict.update({(label, censorship): label_count})
                label_count += 1

        self.num_classes = len(self.label_dict)

        patient_df.reset_index(drop=True, inplace=True)
        for i in patient_df.index:
            disc_label = patient_df.loc[i, "disc_label"]
            censorship = patient_df.loc[i, "censorship"]
            key = (disc_label, int(censorship))
            patient_df.at[i, "label"] = self.label_dict[key]

        slide_df = pd.merge(df, patient_df[["case_id", "label"]], how="left", on="case_id")

        return patient_df, slide_df

    def filter_df(self, df):
        missing_slide_ids = []
        for slide_id in df.slide_id:
            if not Path(self.features_dir, f"{slide_id}.pt").is_file():
                missing_slide_ids.append(slide_id)
        if len(missing_slide_ids) > 0:
            print(f"WARNING: {len(missing_slide_ids)} slides dropped because missing on disk")
        filtered_df = df[~df.slide_id.isin(missing_slide_ids)].reset_index(drop=True)
        return filtered_df

    def map_class_to_slide_ids(self):
        # map each class to corresponding slide ids
        self.class_2_slide_id = defaultdict(list)
        for i in range(self.num_classes):
            class_idxs = np.asarray(self.slide_df.label == i).nonzero()[0]
            self.class_2_slide_id[i] = self.slide_df.loc[class_idxs, 'slide_id'].values.tolist()

    def map_class_to_patient_ids(self):
        # map each class to corresponding patient ids
        self.class_2_patient_id = defaultdict(list)
        for i in range(self.num_classes):
            class_idxs = np.asarray(self.patient_df.label == i).nonzero()[0]
            self.class_2_patient_id[i] = self.patient_df.loc[class_idxs, 'case_id'].values.tolist()

    def get_label(self, idx):
        return self.slide_df.label[idx]

    def __getitem__(self, idx: int):

        row = self.slide_df.loc[idx]
        slide_id = row.slide_id
        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship

        fp = Path(self.features_dir, f"{slide_id}.pt")
        features = torch.load(fp)

        return idx, features, label, event_time, c

    def __len__(self):
        return len(self.slide_df)


class ExtractedFeaturesSurvivalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
        nbins: int = 4,
        eps: float = 1e-6,
    ):

        self.features_dir = features_dir
        self.label_name = label_name
        self.nbins = nbins
        self.eps = eps

        if "IDC" in df['oncotree_code'].values:
            # must be BRCA (and if so, use only IDCs)
            df = df[df['oncotree_code'] == 'IDC']

        patient_df, slide_df = self.prepare_data(df)
        self.patient_df, self.slide_df = self.filter_df(patient_df), self.filter_df(slide_df)

        self.map_class_to_slide_ids()
        self.map_class_to_patient_ids()

    def prepare_data(self, df):

        patient_df = df.drop_duplicates(['case_id']).copy()
        uncensored_df = patient_df[patient_df['censorship'] < 1]

        _, q_bins = pd.qcut(uncensored_df[self.label_name], q=self.nbins, retbins=True, labels=False)
        q_bins[-1] = df[self.label_name].max() + self.eps
        q_bins[0] = df[self.label_name].min() - self.eps

        disc_labels, q_bins = pd.cut(patient_df[self.label_name], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patient_df.insert(2, "disc_label", disc_labels.values.astype(int))

        self.patient_id_2_slide_id = defaultdict(list)
        for patient_id in patient_df['case_id']:
            slide_ids = df[df['case_id'] == patient_id]['slide_id'].values.tolist()
            self.patient_id_2_slide_id[patient_id] = slide_ids

        self.label_dict = {}
        label_count = 0
        for label in range(len(q_bins)-1):
            for censorship in [0, 1]:
                self.label_dict.update({(label, censorship): label_count})
                label_count += 1

        self.num_classes = len(self.label_dict)

        patient_df.reset_index(drop=True, inplace=True)
        for i in patient_df.index:
            disc_label = patient_df.loc[i, "disc_label"]
            censorship = patient_df.loc[i, "censorship"]
            key = (disc_label, int(censorship))
            patient_df.at[i, "label"] = self.label_dict[key]

        slide_df = pd.merge(df, patient_df[["case_id", "label"]], how="left", on="case_id")

        return patient_df, slide_df

    def filter_df(self, df):
        missing_slide_ids = []
        for slide_id in df.slide_id:
            if not Path(self.features_dir, f"{slide_id}.pt").is_file():
                missing_slide_ids.append(slide_id)
        if len(missing_slide_ids) > 0:
            print(f"WARNING: {len(missing_slide_ids)} slides dropped because missing on disk")
        filtered_df = df[~df.slide_id.isin(missing_slide_ids)].reset_index(drop=True)
        return filtered_df

    def map_class_to_slide_ids(self):
        # map each class to corresponding slide ids
        self.class_2_slide_id = defaultdict(list)
        for i in range(self.num_classes):
            class_idxs = np.asarray(self.slide_df.label == i).nonzero()[0]
            self.class_2_slide_id[i] = self.slide_df.loc[class_idxs, 'slide_id'].values.tolist()

    def map_class_to_patient_ids(self):
        # map each class to corresponding patient ids
        self.class_2_patient_id = defaultdict(list)
        for i in range(self.num_classes):
            class_idxs = np.asarray(self.patient_df.label == i).nonzero()[0]
            self.class_2_patient_id[i] = self.patient_df.loc[class_idxs, 'case_id'].values.tolist()

    def get_label(self, idx):
        return self.patient_df.label[idx]

    def __getitem__(self, idx: int):

        row = self.patient_df.loc[idx]
        case_id = row.slide_id
        slide_ids = self.patient_id_2_slide_id[case_id]
        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship

        features = []
        for slide_id in slide_ids:
            fp = Path(self.features_dir, f"{slide_id}.pt")
            f = torch.load(fp)
            features.append(f)
        features = torch.cat(features, dim=0)

        return idx, features, label, event_time, c

    def __len__(self):
        len(self.patient_df)


class StackedRegionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        region_dir: Path,
        region_size: int = 256,
        fmt: str = "jpg",
        label_name: str = "label",
        label_mapping: Dict[int, int] = {},
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
            df["label"] = df[self.label_name].apply(lambda x: self.label_mapping[x])
        elif self.label_name != "label":
            df["label"] = df[self.label_name]
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
        regions_fp = [fp for fp in region_dir.glob(f"*.{self.format}")]
        M = len(regions_fp)
        if self.M_max > -1 and M > self.M_max:
            regions_fp = random.sample([x for x in regions_fp], self.M_max)
            M = self.M_max

        stacked_regions = torch.zeros((M, 3, self.region_size, self.region_size))

        with tqdm.tqdm(
            regions_fp,
            desc=(f"{slide_id}"),
            unit=" tiles",
            ncols=80,
            position=2,
            leave=False,
            disable=True,
        ) as t:

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
        region_dir = Path(
            self.region_root_dir, slide_id, str(self.region_size), self.format
        )
        regions = [str(fp) for fp in region_dir.glob(f"*.{self.format}")]
        return idx, regions

    def __len__(self):
        return len(self.df)
