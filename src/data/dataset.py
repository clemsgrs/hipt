import torch
import pandas as pd

from pathlib import Path
from dataclasses import dataclass


@dataclass
class DatasetOptions:
    df: pd.DataFrame
    features_dir: Path
    label_name: str
    label_mapping: dict[int, int] | None = None
    seed: int = 0


class ExtractedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        options: DatasetOptions,
    ):
        self.options = options
        self.seed = self.options.seed

        self.df = self.prepare_data(self.options.df)
        self.num_classes = len(
            self.options.df[self.options.label_name].value_counts(dropna=True)
        )

    def prepare_data(self, df):
        if self.options.label_mapping:
            df[self.options.label_name] = df[self.options.label_name].apply(
                lambda x: self.options.label_mapping[x]
            )
        # filter out case_ids that do not have corresponding .pt files
        filtered_case_ids = []
        for case_id in df.case_id:
            if Path(self.options.features_dir, f"{case_id}.pt").is_file():
                filtered_case_ids.append(case_id)
        df_filtered = df[df.case_id.isin(filtered_case_ids)].reset_index(drop=True)
        if len(df.case_id) != len(df_filtered.case_id):
            print(
                f"WARNING: {len(df.case_id)-len(df_filtered.case_id)} slides dropped because .pt files missing"
            )
        return df_filtered

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        case_id = row.case_id
        fp = Path(self.options.features_dir, f"{case_id}.pt")
        feature = torch.load(fp, map_location="cpu")
        label = row[self.options.label_name]
        return idx, feature, label

    def __len__(self):
        return len(self.df)


class ExtractedFeaturesSurvivalDataset(ExtractedFeaturesDataset):
    def __init__(
        self,
        options: DatasetOptions,
    ):
        self.options = options
        self.seed = self.options.seed

        self.df = self.prepare_data(self.options.df)
        self.num_classes = len(
            self.options.df.discrete_label.value_counts(dropna=True)
        )

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        case_id = row.case_id
        event_time = row[self.options.label_name]
        censored = row.censored

        fp = Path(self.options.features_dir, f"{case_id}.pt")
        feature = torch.load(fp, map_location='cpu')

        label = row.discrete_label
        return idx, feature, label, event_time, censored