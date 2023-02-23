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
from omegaconf import DictConfig


def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)


def ppcess_tcga_survival_data(
    df,
    label_name: str = "label",
    nbins: int = 4,
    eps: float = 1e-6,
):
    if "IDC" in df["oncotree_code"].values:
        # must be BRCA (and if so, use only IDCs)
        df = df[df["oncotree_code"] == "IDC"]

    patient_df = df.drop_duplicates(["case_id"])
    patient_df = patient_df.drop("slide_id", axis=1)
    uncensored_df = patient_df[patient_df["censorship"] < 1]

    _, q_bins = pd.qcut(uncensored_df[label_name], q=nbins, retbins=True, labels=False)
    q_bins[-1] = df[label_name].max() + eps
    q_bins[0] = df[label_name].min() - eps

    disc_labels, bins = pd.cut(
        patient_df[label_name],
        bins=q_bins,
        retbins=True,
        labels=False,
        right=False,
        include_lowest=True,
    )
    patient_df.insert(2, "disc_label", disc_labels.values.astype(int))

    label_dict = {}
    label_count = 0
    for label in range(len(bins) - 1):
        for censorship in [0, 1]:
            label_dict.update({(label, censorship): label_count})
            label_count += 1

    patient_df.reset_index(drop=True, inplace=True)
    for i in patient_df.index:
        disc_label = patient_df.loc[i, "disc_label"]
        censorship = patient_df.loc[i, "censorship"]
        key = (disc_label, int(censorship))
        patient_df.at[i, "label"] = label_dict[key]

    slide_df = pd.merge(
        df, patient_df[["case_id", "disc_label", "label"]], how="left", on="case_id"
    )

    return patient_df, slide_df


def ppcess_survival_data(
    df,
    label_name: str = "label",
    nbins: int = 4,
    eps: float = 1e-6,
):
    patient_df = df.drop_duplicates(["case_id"])
    patient_df = patient_df.drop("slide_id", axis=1)
    uncensored_df = patient_df[patient_df["censorship"] < 1]

    _, q_bins = pd.qcut(uncensored_df[label_name], q=nbins, retbins=True, labels=False)
    q_bins[-1] = df[label_name].max() + eps
    q_bins[0] = df[label_name].min() - eps

    disc_labels, bins = pd.cut(
        patient_df[label_name],
        bins=q_bins,
        retbins=True,
        labels=False,
        right=False,
        include_lowest=True,
    )
    patient_df.insert(2, "disc_label", disc_labels.values.astype(int))

    label_dict = {}
    label_count = 0
    for label in range(len(bins) - 1):
        for censorship in [0, 1]:
            label_dict.update({(label, censorship): label_count})
            label_count += 1

    patient_df.reset_index(drop=True, inplace=True)
    for i in patient_df.index:
        disc_label = patient_df.loc[i, "disc_label"]
        censorship = patient_df.loc[i, "censorship"]
        key = (disc_label, int(censorship))
        patient_df.at[i, "label"] = label_dict[key]

    slide_df = pd.merge(
        df, patient_df[["case_id", "disc_label", "label"]], how="left", on="case_id"
    )

    return patient_df, slide_df


class DatasetFactory:
    def __init__(
        self,
        task: str,
        options: DictConfig,
        agg_method: str = "concat",
    ):

        if task == "subtyping":
            self.dataset = ExtractedFeaturesDataset(
                options.df,
                options.features_dir,
                options.label_name,
                options.label_mapping,
                options.label_encoding,
            )
        elif task == "survival":
            if options.tiles_df:
                if agg_method == "concat":
                    self.dataset = ExtractedFeaturesCoordsSurvivalDataset(
                        options.patient_df,
                        options.slide_df,
                        options.tiles_df,
                        options.features_dir,
                        options.label_name,
                    )
                else:
                    self.dataset = ExtractedFeaturesPatientLevelCoordsSurvivalDataset(
                        options.patient_df,
                        options.slide_df,
                        options.tiles_df,
                        options.features_dir,
                        options.label_name,
                    )
            else:
                if agg_method == "concat":
                    self.dataset = ExtractedFeaturesSurvivalDataset(
                        options.patient_df,
                        options.slide_df,
                        options.features_dir,
                        options.label_name,
                    )
                else:
                    self.dataset = ExtractedFeaturesPatientLevelSurvivalDataset(
                        options.patient_df,
                        options.slide_df,
                        options.features_dir,
                        options.label_name,
                    )
        else:
            raise ValueError(
                f"task ({task}) not supported"
            )

    def get_dataset(self):
        return self.dataset


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
            df["label"] = df.loc[:, self.label_name]
        filtered_slide_ids = []
        for slide_id in df.slide_id:
            if Path(self.features_dir, f"{slide_id}.pt").is_file():
                filtered_slide_ids.append(slide_id)
        df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        if len(df.slide_id) != len(df_filtered.slide_id):
            print(
                f"WARNING: {len(df.slide_id)-len(df_filtered.slide_id)} slides dropped because .pt files missing"
            )
        return df_filtered

    def map_class_to_slide_ids(self):
        # map each class to corresponding slide ids
        self.class_2_id = defaultdict(list)
        for i in range(self.num_classes):
            class_idxs = np.asarray(self.df.label == i).nonzero()[0]
            # TODO: make sure to add a drop_duplicates(['slide_id']) somewhere before
            self.class_2_id[i] = self.df.loc[class_idxs, "slide_id"].values.tolist()

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


class ExtractedFeaturesSurvivalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
    ):

        self.features_dir = features_dir
        self.label_name = label_name
        self.use_coords = False

        self.slide_df = self.filter_df(slide_df)
        self.patient_df = patient_df

    def filter_df(self, df):
        missing_slide_ids = []
        for slide_id in df.slide_id:
            if not Path(self.features_dir, f"{slide_id}.pt").is_file():
                missing_slide_ids.append(slide_id)
        if len(missing_slide_ids) > 0:
            print(
                f"WARNING: {len(missing_slide_ids)} slides dropped because missing on disk"
            )
        filtered_df = df[~df.slide_id.isin(missing_slide_ids)].reset_index(drop=True)
        return filtered_df

    def __getitem__(self, idx: int):

        row = self.patient_df.loc[idx]
        case_id = row.case_id
        slide_ids = self.slide_df[
            self.slide_df.case_id == case_id
        ].slide_id.values.tolist()

        assert len(slide_ids) == len(set(slide_ids))

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship

        features = []
        for slide_id in slide_ids:
            fp = Path(self.features_dir, f"{slide_id}.pt")
            f = torch.load(fp)
            features.append(f)

        # when multiple slides, concatenate region features
        features = torch.cat(features, dim=0)

        return idx, features, label, event_time, c

    def __len__(self):
        return len(self.patient_df)


class ExtractedFeaturesPatientLevelSurvivalDataset(ExtractedFeaturesSurvivalDataset):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
    ):

        super().__init__(patient_df, slide_df, features_dir, label_name)

    def __getitem__(self, idx: int):

        row = self.patient_df.loc[idx]
        case_id = row.case_id
        slide_ids = self.slide_df[
            self.slide_df.case_id == case_id
        ].slide_id.values.tolist()

        assert len(slide_ids) == len(set(slide_ids))

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship

        features = []
        for slide_id in slide_ids:
            fp = Path(self.features_dir, f"{slide_id}.pt")
            f = torch.load(fp)
            features.append(f)

        return idx, features, label, event_time, c


class ExtractedFeaturesCoordsSurvivalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        tiles_df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
    ):

        self.features_dir = features_dir
        self.label_name = label_name
        self.use_coords = True

        self.slide_df = slide_df
        self.patient_df = patient_df
        self.tiles_df = tiles_df

        tmp = pd.merge(slide_df, tiles_df, on="slide_id")
        # tmp = self.filter_df(tmp)
        self.tmp = tmp.groupby("slide_id").size().to_frame(name="ntile").reset_index()

    def filter_df(self, df):
        def infer_tile_path(slide_id, x, y):
            return 1 - int(Path(self.features_dir, f"{slide_id}_{x}_{y}.pt").exists())

        df["missing"] = df.apply(
            lambda x: infer_tile_path(x.slide_id, x.x, x.y), axis=1
        )
        tmp = df.groubpy("slide_id")
        missing_slide_ids = []
        for slide_id, sub_df in tmp:
            if sub_df.missing.sum() > 0:
                missing_slide_ids.append(slide_id)
        if len(missing_slide_ids) > 0:
            print(
                f"WARNING: {len(missing_slide_ids)} patients dropped because some features were missing on disk"
            )
        filtered_df = df[~df.slide_id.isin(missing_slide_ids)].reset_index(drop=True)
        return filtered_df

    def get_slide_id_with_max_ntile(self, case_id: str):
        slide_ids = self.slide_df[
            self.slide_df.case_id == case_id
        ].slide_id.values.tolist()
        tmp = self.tmp[self.tmp.slide_id.isin(slide_ids)]
        max_id = tmp[tmp.ntile == tmp.ntile.max()]["slide_id"].unique().tolist()
        # assert len(max_id) == 1, f"{case_id} has {len(max_id)} slides with same max ntile: {max_id}"
        return max_id[0]

    def __getitem__(self, idx: int):

        row = self.patient_df.loc[idx]
        case_id = row.case_id
        # slide_ids = self.slide_df[self.slide_df.case_id == case_id].slide_id.values.tolist()

        slide_id = self.get_slide_id_with_max_ntile(case_id)

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship

        features = []
        # coordinates = []
        # for slide_id in slide_ids:
        #     coords = self.tiles_df[self.tiles_df.slide_id == slide_id][['x','y']].values
        #     for x,y in coords:
        #         fp = Path(self.features_dir, f"{slide_id}_{x}_{y}.pt")
        #         f = torch.load(fp)
        #         features.append(f)
        #         coordinates.append((x,y))
        coordinates = self.tiles_df[self.tiles_df.slide_id == slide_id][
            ["x", "y"]
        ].values
        for x, y in coordinates:
            fp = Path(self.features_dir, f"{slide_id}_{x}_{y}.pt")
            f = torch.load(fp)
            features.append(f)

        features = torch.cat(features, dim=0)

        return idx, features, coordinates, label, event_time, c

    def __len__(self):
        return len(self.patient_df)


class ExtractedFeaturesPatientLevelCoordsSurvivalDataset(
    ExtractedFeaturesCoordsSurvivalDataset
):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        tiles_df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
    ):

        super().__init__(patient_df, slide_df, tiles_df, features_dir, label_name)

    def __getitem__(self, idx: int):

        row = self.patient_df.loc[idx]
        case_id = row.case_id
        slide_ids = self.slide_df[
            self.slide_df.case_id == case_id
        ].slide_id.values.tolist()

        assert len(slide_ids) == len(set(slide_ids))

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship

        features = []
        coordinates = []
        for slide_id in slide_ids:
            feats = []
            coords = self.tiles_df[self.tiles_df.slide_id == slide_id][
                ["x", "y"]
            ].values
            for x, y in coords:
                fp = Path(self.features_dir, f"{slide_id}_{x}_{y}.pt")
                f = torch.load(fp)
                feats.append(f)
            feats = torch.cat(feats, dim=0)
            features.append(feats)
            coordinates.append(coords)

        return idx, features, coordinates, label, event_time, c


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
            df["label"] = df.loc[:, self.label_name]
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
        region_dir: Path,
        fmt: str,
    ):
        self.df = df
        self.region_dir = region_dir
        self.format = fmt

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        slide_dir = Path(self.region_dir, slide_id, "imgs")
        regions = [str(fp) for fp in slide_dir.glob(f"*.{self.format}")]
        return idx, regions

    def __len__(self):
        return len(self.df)


class HierarchicalPretrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features_dir: str,
        transform: Callable,
    ):
        self.features_list = [f for f in Path(features_dir).glob("*.pt")]
        self.transform = transform

    def __getitem__(self, idx: int):
        f = torch.load(self.features_list[idx])
        f = self.transform(f)
        label = torch.zeros(1, 1)
        return f, label

    def __len__(self):
        return len(self.features_list)
