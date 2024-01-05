import tqdm
import torch
import random
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from torchvision import transforms, datasets
from typing import Callable, Dict, Optional, Any, Union
from collections import defaultdict
from omegaconf import DictConfig
from dataclasses import dataclass, field
from torchvision.datasets.folder import default_loader

from source.wsi import WholeSlideImage


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


@dataclass
class ClassificationDatasetOptions:
    df: pd.DataFrame
    features_dir: Path
    label_name: str
    label_mapping: Dict[int, int] = field(default_factory=lambda: {})
    label_encoding: Optional[str] = None
    transform: Optional[Callable] = None
    blinded: bool = False
    num_classes: Optional[int] = None
    mask_attention: bool = False
    region_dir: Optional[Path] = None
    spacing: float = 0.5
    region_size: int = 4096
    patch_size: int = 256
    mini_patch_size: int = 16
    backend: str = "pyvips"
    region_format: str = "jpg"
    segmentation_parameters: Dict[str, Union[bool, int]] = field(
        default_factory=lambda: {}
    )
    tissue_pct: float = 0.0


@dataclass
class SurvivalDatasetOptions:
    patient_df: pd.DataFrame
    slide_df: pd.DataFrame
    tiles_df: pd.DataFrame
    features_dir: Path
    label_name: str
    transform: Optional[Callable] = (None,)


class DatasetFactory:
    def __init__(
        self,
        task: str,
        options: DictConfig,
        agg_method: str = "concat",
    ):
        if task in ["classification", "regression"]:
            if options.blinded:
                self.dataset = BlindedExtractedFeaturesDataset(
                    options.df,
                    options.features_dir,
                    options.transform,
                )
            elif options.mask_attention:
                self.dataset = ExtractedFeaturesMaskedDataset(
                    options.df,
                    options.features_dir,
                    options.region_dir,
                    options.spacing,
                    options.region_size,
                    options.patch_size,
                    options.mini_patch_size,
                    options.backend,
                    options.region_format,
                    options.segmentation_parameters,
                    options.tissue_pct,
                    options.transform,
                    options.label_name,
                    options.label_mapping,
                )
            elif options.label_encoding == "ordinal":
                self.dataset = ExtractedFeaturesOrdinalDataset(
                    options.df,
                    options.features_dir,
                    options.transform,
                    options.label_name,
                    options.label_mapping,
                )
            else:
                self.dataset = ExtractedFeaturesDataset(
                    options.df,
                    options.features_dir,
                    options.transform,
                    options.label_name,
                    options.label_mapping,
                )
        elif task == "survival":
            if options.tiles_df is not None:
                if agg_method == "concat":
                    self.dataset = ExtractedFeaturesCoordsSurvivalDataset(
                        options.patient_df,
                        options.slide_df,
                        options.tiles_df,
                        options.features_dir,
                        options.label_name,
                    )
                elif agg_method == "self_att":
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
                elif agg_method == "self_att":
                    self.dataset = ExtractedFeaturesPatientLevelSurvivalDataset(
                        options.patient_df,
                        options.slide_df,
                        options.features_dir,
                        options.label_name,
                    )
                elif not agg_method:
                    self.dataset = ExtractedFeaturesSlideLevelSurvivalDataset(
                        options.slide_df,
                        options.features_dir,
                        options.label_name,
                    )
        else:
            raise ValueError(f"task ({task}) not supported")

    def get_dataset(self):
        return self.dataset


class ExtractedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        transform: Optional[Callable] = None,
        label_name: str = "label",
        label_mapping: Dict[int, int] = {},
    ):
        self.features_dir = features_dir
        self.label_name = label_name
        self.label_mapping = label_mapping
        self.transform = transform

        self.seed = 0

        self.df, _ = self.prepare_data(df)

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
        df_missing = df[~df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        if len(df.slide_id) != len(df_filtered.slide_id):
            print(
                f"WARNING: {len(df.slide_id)-len(df_filtered.slide_id)} slides dropped because .pt files missing"
            )
        return df_filtered, df_missing

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
        if self.transform:
            features = self.transform(features, slide_id, self.seed)
        return idx, features, label

    def __len__(self):
        return len(self.df)


class ExtractedFeaturesOrdinalDataset(ExtractedFeaturesDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        transform: Optional[Callable] = None,
        label_name: str = "label",
        label_mapping: Dict[int, int] = {},
    ):
        super().__init__(df, features_dir, transform, label_name, label_mapping)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        fp = Path(self.features_dir, f"{slide_id}.pt")
        features = torch.load(fp)
        label = np.zeros(self.num_classes - 1).astype(np.float32)
        label[: row.label] = 1.0
        if self.transform:
            features = self.transform(features, slide_id, self.seed)
        return idx, features, label


class BlindedExtractedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        transform: Optional[Callable] = None,
        num_classes: Optional[int] = None,
    ):
        self.features_dir = features_dir
        self.transform = transform
        self.num_classes = num_classes

        self.seed = 0
        self.df, _ = self.prepare_data(df)

    def prepare_data(self, df):
        filtered_slide_ids = []
        for slide_id in df.slide_id:
            if Path(self.features_dir, f"{slide_id}.pt").is_file():
                filtered_slide_ids.append(slide_id)
        df_missing = df[~df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        df_filtered = df[df.slide_id.isin(filtered_slide_ids)].reset_index(drop=True)
        if len(df.slide_id) != len(df_filtered.slide_id):
            print(
                f"WARNING: {len(df.slide_id)-len(df_filtered.slide_id)} slides dropped because .pt files missing"
            )
        return df_filtered, df_missing

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        fp = Path(self.features_dir, f"{slide_id}.pt")
        features = torch.load(fp)
        if self.transform:
            features = self.transform(features, slide_id, self.seed)
        return idx, features, _

    def __len__(self):
        return len(self.df)


class ExtractedFeaturesSurvivalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patient_df: pd.DataFrame,
        slide_df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
        nfeats_max: Optional[int] = 1000,
    ):
        self.features_dir = features_dir
        self.label_name = label_name
        self.use_coords = False
        self.agg_level = "patient"
        self.nfeats_max = nfeats_max

        self.seed = 0

        self.slide_df, case_ids = self.filter_df(slide_df)
        self.df = patient_df[patient_df.case_id.isin(case_ids)].reset_index(drop=True)

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
        remaining_case_ids = filtered_df.case_id.unique().tolist()
        return filtered_df, remaining_case_ids

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
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

        # if more than nfeats_max features, randomly sample nfeats_max features
        if self.nfeats_max and len(features) > self.nfeats_max:
            torch.manual_seed(self.seed)
            features = features[torch.randperm(len(features))[: self.nfeats_max]]

        return idx, features, label, event_time, c

    def __len__(self):
        return len(self.df)


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
        row = self.df.loc[idx]
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
        self.agg_level = "patient"

        self.seed = 0

        self.slide_df = slide_df
        self.df = patient_df
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
                f"WARNING: {len(missing_slide_ids)} patients dropped because some tiles features were missing on disk"
            )
        filtered_df = df[~df.slide_id.isin(missing_slide_ids)].reset_index(drop=True)
        return filtered_df

    def get_slide_id_with_max_ntile(self, case_id: str):
        slide_ids = self.slide_df[
            self.slide_df.case_id == case_id
        ].slide_id.values.tolist()
        tmp = self.tmp[self.tmp.slide_id.isin(slide_ids)]
        max_id = tmp[tmp.ntile == tmp.ntile.max()]["slide_id"].unique().tolist()
        # if len(max_id) > 1, the given case_id has len(max_id) slides with same max ntile
        # in that case, we "randomly" pick the first slide
        return max_id[0]

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        case_id = row.case_id
        slide_ids = self.slide_df[
            self.slide_df.case_id == case_id
        ].slide_id.values.tolist()

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship

        features = []
        coordinates = []
        for i, slide_id in enumerate(slide_ids):
            coords = self.tiles_df[self.tiles_df.slide_id == slide_id][
                ["x", "y"]
            ].values
            for x, y in coords:
                fp = Path(self.features_dir, f"{slide_id}_{x}_{y}.pt")
                f = torch.load(fp)
                features.append(f)
                coordinates.append((i, x, y))
        coordinates = np.array(coordinates)

        # slide_id = self.get_slide_id_with_max_ntile(case_id)
        # coordinates = self.tiles_df[self.tiles_df.slide_id == slide_id][
        #     ["x", "y"]
        # ].values
        # for x, y in coordinates:
        #     fp = Path(self.features_dir, f"{slide_id}_{x}_{y}.pt")
        #     f = torch.load(fp)
        #     features.append(f)

        features = torch.cat(features, dim=0)

        return idx, features, coordinates, label, event_time, c

    def __len__(self):
        return len(self.df)


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
        row = self.df.loc[idx]
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


class ExtractedFeaturesSlideLevelSurvivalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        slide_df: pd.DataFrame,
        features_dir: Path,
        label_name: str = "label",
    ):
        self.features_dir = features_dir
        self.label_name = label_name
        self.use_coords = False
        self.agg_level = "slide"

        self.seed = 0

        self.df = self.filter_df(slide_df)

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
        row = self.df.loc[idx]
        slide_id = row.slide_id

        label = row.disc_label
        event_time = row[self.label_name]
        c = row.censorship

        fp = Path(self.features_dir, f"{slide_id}.pt")
        feature = torch.load(fp)

        return idx, feature, label, event_time, c

    def __len__(self):
        return len(self.df)


class ExtractedFeaturesMaskedDataset(ExtractedFeaturesDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        region_dir: Path,
        spacing: float,
        region_size: int,
        patch_size: int = 256,
        mini_patch_size: int = 16,
        backend: str = "pyvips",
        region_format: str = "jpg",
        segmentation_parameters: Dict[str, Union[bool, int]] = {},
        tissue_pct: float = 0.0,
        transform: Optional[Callable] = None,
        label_name: str = "label",
        label_mapping: Dict[int, int] = {},
        verbose: bool = True,
    ):
        super().__init__(df, features_dir, transform, label_name, label_mapping)
        self.region_dir = region_dir
        self.region_format = region_format
        self.spacing = spacing
        self.region_size = region_size
        self.patch_size = patch_size
        self.mini_patch_size = mini_patch_size
        self.backend = backend
        self.segmentation_parameters = segmentation_parameters
        self.tissue_pct = tissue_pct
        self.verbose = verbose

        self.masks = self.generate_masks()

    def generate_masks(self):
        self.slide_id_to_tissue_pct = {}
        if "segmentation_mask_path" in self.df.columns:
            id_path_mask = zip(
                self.df.slide_id.unique().tolist(),
                self.df.slide_path.unique().tolist(),
                self.df.segmentation_mask_path.unique().tolist(),
            )
        else:
            id_path_mask = zip(
                self.df.slide_id.unique().tolist(),
                self.df.slide_path.unique().tolist(),
                [None] * self.df.slide_id.nunique(),
            )
        with tqdm.tqdm(
            id_path_mask,
            desc=(f"Infering attention masks from tissue content"),
            unit=f" slide",
            total=self.df.slide_id.nunique(),
            leave=True,
            disable=(not self.verbose),
        ) as t:
            for (sid, slide_fp, mask_fp) in t:
                # load the slide
                wsi_object = WholeSlideImage(Path(slide_fp), backend=self.backend)
                if mask_fp:
                    # load segmentation mask
                    seg_level, seg_spacing = wsi_object.loadSegmentation(
                        Path(mask_fp),
                        downsample=self.segmentation_parameters["downsample"],
                    )
                else:
                    self.segmentation_parameters["tissue_pixel_value"] = 1
                    seg_level, seg_spacing = wsi_object.segmentTissue(
                        downsample=self.segmentation_parameters["downsample"],
                        sthresh=self.segmentation_parameters["sthresh"],
                        mthresh=self.segmentation_parameters["mthresh"],
                        close=self.segmentation_parameters["close"],
                        use_otsu=self.segmentation_parameters["use_otsu"],
                    )
                # scale coordinates from slide's level 0 to mask's level
                sx, sy = tuple(
                    i / j
                    for i, j in zip(
                        wsi_object.wsi.shapes[seg_level], wsi_object.wsi.shapes[0]
                    )
                )
                # scale region size from true spacing to downsample spacing
                # we excepct mask spacings to be a subset of slide spacings
                # the ratio should thus give an integer
                sr = round(wsi_object.wsi.get_real_spacing(seg_spacing) / wsi_object.wsi.get_real_spacing(self.spacing))
                scaled_region_size = self.region_size // sr
                # scale patch_size and mini_patch_size
                scaled_patch_size = self.patch_size // sr
                scaled_mini_patch_size = self.mini_patch_size // sr
                # retrieve region's (x,y) coordinates
                # should appear in the same order as in the corresponding slide feature vector
                coordinates = sorted(
                    [
                        p.stem
                        for p in Path(self.region_dir, sid, "imgs").glob(
                            f"*.{self.region_format}"
                        )
                    ]
                )
                coordinates = [
                    (int(p.split("_")[0]), int(p.split("_")[1])) for p in coordinates
                ]
                tissue_pcts = []
                for x, y in coordinates:
                    x_mask, y_mask = int(x * sx), int(y * sy)
                    mask_region = np.zeros((scaled_region_size, scaled_region_size))
                    # going from WholeSlideImage to numpy arrays: need to switch x & y axes
                    sub_mask = wsi_object.binary_mask[
                        y_mask : y_mask + scaled_region_size,
                        x_mask : x_mask + scaled_region_size,
                    ]
                    mask_region[:sub_mask.shape[0],:sub_mask.shape[1]] = sub_mask

                    def split_into_blocks(arr, block_size):
                        """
                        Split a 2D array into smaller blocks of given block_size.
                        """
                        # Split the array into sub-arrays of size block_size along the rows
                        rows_split = np.split(arr, arr.shape[0] // block_size)
                        # Further split these sub-arrays along the columns
                        blocks = [
                            np.split(block, arr.shape[1] // block_size, axis=1)
                            for block in rows_split
                        ]
                        # Flatten the list of blocks
                        flattened_blocks = [
                            block for sublist in blocks for block in sublist
                        ]
                        return np.array(flattened_blocks)

                    # compute tissue percentage for each mini patch in each patch
                    mask_patches = split_into_blocks(mask_region, scaled_patch_size)

                    mask_mini_patches = []
                    for p in mask_patches:
                        mp = split_into_blocks(p, scaled_mini_patch_size)
                        mask_mini_patches.append(mp)
                    mask_mini_patches = np.stack(mask_mini_patches)
                    tissue = (
                        mask_mini_patches
                        == self.segmentation_parameters["tissue_pixel_value"]
                    )
                    tissue_pct = np.sum(tissue, axis=(-2, -1)) / tissue[0][0].size
                    tissue_pcts.append(tissue_pct)

                tissue_pcts = np.stack(tissue_pcts)  # (M, npatch**2, nminipatch**2)
                self.slide_id_to_tissue_pct[sid] = tissue_pcts

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_id = row.slide_id
        fp = Path(self.features_dir, f"{slide_id}.pt")
        features = torch.load(fp)
        label = row.label
        if self.transform:
            features = self.transform(features, slide_id, self.seed)
        tissue_pct = self.slide_id_to_tissue_pct[
            slide_id
        ]  # (M, npatch**2, nminipatch**2)
        tissue_pct = torch.Tensor(tissue_pct)
        return idx, features, label, tissue_pct


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

        self.seed = 0

        self.df = self.prepare_data(df)

        self.num_classes = len(self.df.label.value_counts(dropna=True))
        self.map_class_to_slide_ids()

    def filter_df(self, df):
        missing_slide_ids = []
        for slide_id in df.slide_id:
            if not Path(self.region_dir, f"{slide_id}").is_dir():
                missing_slide_ids.append(slide_id)
        if len(missing_slide_ids) > 0:
            print(
                f"WARNING: {len(missing_slide_ids)} slides dropped because 0 regions found on disk"
            )
        filtered_df = df[~df.slide_id.isin(missing_slide_ids)].reset_index(drop=True)
        return filtered_df

    def prepare_data(self, df):
        df = self.filter_df(df)
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
        region_dir = Path(self.region_dir, slide_id, "imgs")
        regions_fp = [fp for fp in region_dir.glob(f"*.{self.format}")]
        M = len(regions_fp)

        if self.M_max > -1 and M > self.M_max:
            random.seed(self.seed)
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

        return idx, stacked_regions, M, label

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
        return idx, regions, slide_id, None

    def __len__(self):
        return len(self.df)


class SlideFilepathsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        slide_fp = row.slide_path
        return idx, slide_fp

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


class ImagePretrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tiles_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        label_name: Optional[str] = None,
    ):
        self.df = tiles_df
        self.transform = transform
        self.loader = loader
        self.label_name = label_name

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        path = row.tile_path
        tile = self.loader(path)
        if self.transform is not None:
            tile = self.transform(tile)
        label = -1
        if self.label_name is not None:
            label = row[self.label_name]
        return tile, label

    def __len__(self):
        return len(self.df)


class ImageFolderWithNameDataset(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            transform,
        )

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[idx]
        fname = Path(path).stem
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, fname
