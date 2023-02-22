import os
import time
import tqdm
import wandb
import torch
import hydra
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig

from source.components import LossFactory
from source.dataset import (
    ExtractedFeaturesCoordsSurvivalDataset,
    ExtractedFeaturesPatientLevelCoordsSurvivalDataset,
    ppcess_survival_data,
)
from source.utils import (
    initialize_wandb,
    compute_time,
    update_log_dict,
    get_cumulative_dynamic_auc,
    plot_cumulative_dynamic_auc,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)

import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from sksurv.metrics import concordance_index_censored
from source.model_utils import Attn_Net_Gated


class GlobalHIPT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        d_model: int = 192,
        tile_size: int = 4096,
        dropout: float = 0.25,
        pos_encoding: bool = False,
    ):

        super(GlobalHIPT, self).__init__()
        self.num_classes = num_classes
        self.tile_size = tile_size
        self.use_pos_encoding = pos_encoding

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )
        if self.use_pos_encoding:
            self.pos_embedder_x = nn.Embedding(512, d_model // 2)
            self.pos_embedder_y = nn.Embedding(512, d_model // 2)
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192,
                nhead=3,
                dim_feedforward=192,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=192, D=192, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(192, 192), nn.ReLU(), nn.Dropout(dropout)]
        )

        self.classifier = nn.Linear(192, num_classes)

    def get_grid_values(self, coords: np.ndarray, tile_size: int):
        m = coords.min()
        grid_coords = torch.div(coords - m, tile_size, rounding_mode="floor")
        return grid_coords

    def forward(self, x, coords):

        # x = [M, 192]
        x = self.global_phi(x)

        if self.use_pos_encoding:
            coords = coords.squeeze(0)
            x_values = self.get_grid_values(coords[:, 0], self.tile_size)
            y_values = self.get_grid_values(coords[:, 1], self.tile_size)
            position_embeddings_x = self.pos_embedder_x(x_values)
            position_embeddings_y = self.pos_embedder_y(y_values)
            position_embeddings = torch.cat(
                [position_embeddings_x, position_embeddings_y], dim=1
            )
            x += position_embeddings

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)

        return logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_phi = self.global_phi.to(device)
        if self.use_pos_encoding:
            self.pos_embedder_x = self.pos_embedder_x.to(device)
            self.pos_embedder_y = self.pos_embedder_y.to(device)
        self.global_transformer = self.global_transformer.to(device)
        self.global_attn_pool = self.global_attn_pool.to(device)
        self.global_rho = self.global_rho.to(device)

        self.classifier = self.classifier.to(device)

    def __repr__(self) -> str:
        num_params = 0
        num_params_train = 0
        for param in self.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        main_str = f"Total number of parameters: {num_params}\n"
        main_str += f"Total number of trainable parameters: {num_params_train}"
        return main_str


class GlobalPatientLevelHIPT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        embed_dim_slide: int = 192,
        embed_dim_patient: int = 192,
        tile_size: int = 4096,
        dropout: float = 0.25,
        pos_encoding: bool = False,
    ):

        super(GlobalPatientLevelHIPT, self).__init__()
        self.num_classes = num_classes
        self.tile_size = tile_size
        self.use_pos_encoding = pos_encoding

        # from region to slide aggregation
        self.global_phi_slide = nn.Sequential(
            nn.Linear(embed_dim_region, embed_dim_slide), nn.ReLU(), nn.Dropout(dropout)
        )
        if self.use_pos_encoding:
            self.pos_embedder_x = nn.Embedding(512, embed_dim_slide // 2)
            self.pos_embedder_y = nn.Embedding(512, embed_dim_slide // 2)
        self.global_transformer_slide = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim_slide,
                nhead=3,
                dim_feedforward=embed_dim_slide,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool_slide = Attn_Net_Gated(
            L=embed_dim_slide, D=embed_dim_slide, dropout=dropout, num_classes=1
        )
        self.global_rho_slide = nn.Sequential(
            *[
                nn.Linear(embed_dim_slide, embed_dim_slide),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        # from slide to patient aggregation
        self.global_phi_patient = nn.Sequential(
            nn.Linear(embed_dim_slide, embed_dim_patient),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_transformer_patient = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim_patient,
                nhead=3,
                dim_feedforward=embed_dim_patient,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool_patient = Attn_Net_Gated(
            L=embed_dim_patient, D=embed_dim_patient, dropout=dropout, num_classes=1
        )
        self.global_rho_patient = nn.Sequential(
            *[
                nn.Linear(embed_dim_patient, embed_dim_patient),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        self.classifier = nn.Linear(embed_dim_patient, num_classes)

    def get_grid_values(self, coords: np.ndarray, tile_size: int):
        m = coords.min()
        grid_coords = torch.div(coords - m, tile_size, rounding_mode="floor")
        return grid_coords

    def forward(self, x, coords):

        # x = [ [M1, 192], [M2, 192], ...]
        N = len(x)
        slide_seq = []
        for n in range(N):
            y = x[n]
            coord = coords[n]
            y = self.global_phi_slide(y)
            if self.use_pos_encoding:
                x_values = self.get_grid_values(coord[:, 0], self.tile_size)
                y_values = self.get_grid_values(coord[:, 1], self.tile_size)
                position_embeddings_x = self.pos_embedder_x(x_values)
                position_embeddings_y = self.pos_embedder_y(y_values)
                position_embeddings = torch.cat(
                    [position_embeddings_x, position_embeddings_y], dim=1
                )
                y += position_embeddings
            # in nn.TransformerEncoderLayer, batch_first defaults to False
            # hence, input is expected to be of shape (seq_length, batch, emb_size)
            y = self.global_transformer_slide(y.unsqueeze(1)).squeeze(1)
            att_slide, y = self.global_attn_pool_slide(y)
            att_slide = torch.transpose(att_slide, 1, 0)
            att_slide = F.softmax(att_slide, dim=1)
            y_att = torch.mm(att_slide, y)
            y_slide = self.global_rho_slide(y_att)
            slide_seq.append(y_slide)

        slide_seq = torch.cat(slide_seq, dim=0)
        # slide_seq = [N, 192]
        z = self.global_phi_patient(slide_seq)

        z = self.global_transformer_patient(z.unsqueeze(1)).squeeze(1)
        att_patient, z = self.global_attn_pool_patient(z)
        att_patient = torch.transpose(att_patient, 1, 0)
        att_patient = F.softmax(att_patient, dim=1)
        z_att = torch.mm(att_patient, z)
        z_patient = self.global_rho_patient(z_att)

        logits = self.classifier(z_patient)

        return logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_phi_slide = self.global_phi_slide.to(device)
        if self.use_pos_encoding:
            self.pos_embedder_x = self.pos_embedder_x.to(device)
            self.pos_embedder_y = self.pos_embedder_y.to(device)
        self.global_transformer_slide = self.global_transformer_slide.to(device)
        self.global_attn_pool_slide = self.global_attn_pool_slide.to(device)
        self.global_rho_slide = self.global_rho_slide.to(device)

        self.global_phi_patient = self.global_phi_patient.to(device)
        self.global_transformer_patient = self.global_transformer_patient.to(device)
        self.global_attn_pool_patient = self.global_attn_pool_patient.to(device)
        self.global_rho_patient = self.global_rho_patient.to(device)

        self.classifier = self.classifier.to(device)

    def __repr__(self) -> str:
        num_params = 0
        num_params_train = 0
        for param in self.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        main_str = f"Total number of parameters: {num_params}\n"
        main_str += f"Total number of trainable parameters: {num_params_train}"
        return main_str


def collate_features(batch, label_type: str = "int", agg_method: str = "concat"):
    idx = torch.LongTensor([item[0] for item in batch])
    if agg_method == "concat":
        feature = torch.cat([item[1] for item in batch], dim=0)
        coords = torch.LongTensor(np.array([item[2] for item in batch]))
    elif agg_method == "self_att":
        feature = [item[1] for item in batch]
        coords = [
            [torch.LongTensor(item[2][i]) for i in range(len(item[2]))]
            for item in batch
        ]
    if label_type == "float":
        label = torch.FloatTensor([item[3] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[3] for item in batch])
    event_time = torch.FloatTensor([item[4] for item in batch])
    censorship = torch.FloatTensor([item[5] for item in batch])
    return [idx, feature, coords, label, event_time, censorship]


def train_survival(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    agg_method: str = "concat",
    batch_size: Optional[int] = 1,
    gradient_accumulation: Optional[int] = 1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
    censorships, event_times = [], []
    risk_scores, labels = [], []
    idxs = []

    sampler = torch.utils.data.RandomSampler(dataset)
    collate_fn = partial(collate_features, label_type="int", agg_method=agg_method)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Train - Epoch {epoch}"),
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        for i, batch in enumerate(t):

            idx, x, coords, label, event_time, c = batch
            label, c = label.to(device, non_blocking=True), c.to(
                device, non_blocking=True
            )
            if agg_method == "concat":
                x, coords = x.to(device, non_blocking=True), coords.to(
                    device, non_blocking=True
                )
            elif agg_method == "self_att":
                x = [f.to(device, non_blocking=True) for f in x[0]]
                coords = [c.to(device, non_blocking=True) for c in coords[0]]

            logits = model(x, coords)  # [1, nbins]
            hazards = torch.sigmoid(logits)  # [1, nbins]
            surv = torch.cumprod(1 - hazards, dim=1)  # [1, nbins]

            loss = criterion(hazards, surv, label, c)

            loss_value = loss.item()
            epoch_loss += loss_value

            risk = -torch.sum(surv, dim=1).detach()  # [1]
            risk_scores.append(risk.item())
            censorships.append(c.item())
            event_times.append(event_time.item())

            if gradient_accumulation:
                loss = loss / gradient_accumulation

            loss.backward()

            if (i + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    dataset.patient_df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1 - c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune_survival(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    agg_method: str = "concat",
    batch_size: Optional[int] = 1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss = 0
    censorships, event_times = [], []
    risk_scores, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)
    collate_fn = partial(collate_features, label_type="int", agg_method=agg_method)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Tune - Epoch {epoch}"),
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, x, coords, label, event_time, c = batch
                label, c = label.to(device, non_blocking=True), c.to(
                    device, non_blocking=True
                )
                if agg_method == "concat":
                    x, coords = x.to(device, non_blocking=True), coords.to(
                        device, non_blocking=True
                    )
                elif agg_method == "self_att":
                    x = [f.to(device, non_blocking=True) for f in x[0]]
                    coords = [c.to(device, non_blocking=True) for c in coords[0]]

                logits = model(x, coords)
                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1 - hazards, dim=1)

                loss = criterion(hazards, surv, label, c, alpha=0)
                epoch_loss += loss.item()

                risk = -torch.sum(surv, dim=1).detach()
                risk_scores.append(risk.item())
                censorships.append(c.item())
                event_times.append(event_time.item())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    dataset.patient_df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1 - c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index
    results["risks"] = risk_scores

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test_survival(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    agg_method: str = "concat",
    batch_size: Optional[int] = 1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    censorships, event_times = [], []
    risk_scores, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)
    collate_fn = partial(collate_features, label_type="int", agg_method=agg_method)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Test"),
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, x, coords, label, event_time, c = batch
                label, c = label.to(device, non_blocking=True), c.to(
                    device, non_blocking=True
                )
                if agg_method == "concat":
                    x, coords = x.to(device, non_blocking=True), coords.to(
                        device, non_blocking=True
                    )
                elif agg_method == "self_att":
                    x = [f.to(device, non_blocking=True) for f in x[0]]
                    coords = [c.to(device, non_blocking=True) for c in coords[0]]

                logits = model(x, coords)
                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1 - hazards, dim=1)

                risk = -torch.sum(surv, dim=1).detach()
                risk_scores.append(risk.item())
                censorships.append(c.item())
                event_times.append(event_time.item())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    dataset.patient_df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1 - c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index

    return results


@hydra.main(
    version_base="1.2.0", config_path="../config/training/survival", config_name="tmp"
)
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir, cfg.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_root_dir = Path(output_dir, "checkpoints", cfg.level)
    checkpoint_root_dir.mkdir(parents=True, exist_ok=True)

    result_root_dir = Path(output_dir, "results", cfg.level)
    result_root_dir.mkdir(parents=True, exist_ok=True)

    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        _ = initialize_wandb(cfg, key=key)

    features_dir = Path(output_dir, "features", cfg.level)
    if cfg.features_dir:
        features_dir = Path(cfg.features_dir)

    tiles_df = pd.read_csv(cfg.data.tiles_csv)

    fold_root_dir = Path(cfg.data.fold_dir)
    nfold = len([_ for _ in fold_root_dir.glob(f"fold_*")])
    print(f"Training on {nfold} folds")

    test_metrics = []

    start_time = time.time()
    for i in range(nfold):

        fold_dir = Path(fold_root_dir, f"fold_{i}")
        checkpoint_dir = Path(checkpoint_root_dir, f"fold_{i}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir = Path(result_root_dir, f"fold_{i}")
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading data for fold {i}")
        dfs = {}
        for p in ["train", "tune", "test"]:
            df_path = Path(fold_dir, f"{p}.csv")
            df = pd.read_csv(df_path)
            df["partition"] = [p] * len(df)
            dfs[p] = df

        if cfg.training.pct:
            print(f"Training on {cfg.training.pct*100}% of the data")
            dfs["train"] = (
                dfs["train"].sample(frac=cfg.training.pct).reset_index(drop=True)
            )

        df = pd.concat([df for df in dfs.values()], ignore_index=True)
        patient_df, slide_df = ppcess_survival_data(df, cfg.label_name, nbins=cfg.nbins)

        patient_dfs, slide_dfs = {}, {}
        for p in ["train", "tune", "test"]:
            patient_dfs[p] = patient_df[patient_df.partition == p].reset_index(
                drop=True
            )
            slide_dfs[p] = slide_df[slide_df.partition == p]

        if cfg.model.agg_method == "concat":
            train_dataset = ExtractedFeaturesCoordsSurvivalDataset(
                patient_dfs["train"],
                slide_dfs["train"],
                tiles_df,
                features_dir,
                cfg.label_name,
            )
            tune_dataset = ExtractedFeaturesCoordsSurvivalDataset(
                patient_dfs["tune"],
                slide_dfs["tune"],
                tiles_df,
                features_dir,
                cfg.label_name,
            )
            test_dataset = ExtractedFeaturesCoordsSurvivalDataset(
                patient_dfs["test"],
                slide_dfs["test"],
                tiles_df,
                features_dir,
                cfg.label_name,
            )
        elif cfg.model.agg_method == "self_att":
            train_dataset = ExtractedFeaturesPatientLevelCoordsSurvivalDataset(
                patient_dfs["train"],
                slide_dfs["train"],
                tiles_df,
                features_dir,
                cfg.label_name,
            )
            tune_dataset = ExtractedFeaturesPatientLevelCoordsSurvivalDataset(
                patient_dfs["tune"],
                slide_dfs["tune"],
                tiles_df,
                features_dir,
                cfg.label_name,
            )
            test_dataset = ExtractedFeaturesPatientLevelCoordsSurvivalDataset(
                patient_dfs["test"],
                slide_dfs["test"],
                tiles_df,
                features_dir,
                cfg.label_name,
            )

        # model = ModelFactory(cfg.level, cfg.nbins, cfg.model).get_model()
        if cfg.model.agg_method == "concat":
            model = GlobalHIPT(
                cfg.nbins,
                dropout=cfg.model.dropout,
                pos_encoding=cfg.model.pos_encoding,
            )
        elif cfg.model.agg_method == "self_att":
            model = GlobalPatientLevelHIPT(
                cfg.nbins,
                dropout=cfg.model.dropout,
                pos_encoding=cfg.model.pos_encoding,
            )
        model.relocate()
        print(model)

        print("Configuring optimmizer & scheduler")
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = OptimizerFactory(
            cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
        ).get_optimizer()
        scheduler = SchedulerFactory(optimizer, cfg.optim.lr_scheduler).get_scheduler()

        criterion = LossFactory(cfg.task, cfg.loss).get_loss()

        early_stopping = EarlyStopping(
            cfg.early_stopping.tracking,
            cfg.early_stopping.min_max,
            cfg.early_stopping.patience,
            cfg.early_stopping.min_epoch,
            checkpoint_dir=checkpoint_dir,
            save_all=cfg.early_stopping.save_all,
        )

        stop = False
        fold_start_time = time.time()

        if cfg.wandb.enable:
            wandb.define_metric(f"train/fold_{i}/epoch", summary="max")

        with tqdm.tqdm(
            range(cfg.nepochs),
            desc=(f"Fold {i} Training"),
            unit=" patient",
            ncols=100,
            leave=True,
        ) as t:

            for epoch in t:

                epoch_start_time = time.time()
                if cfg.wandb.enable:
                    log_dict = {f"train/fold_{i}/epoch": epoch + 1}

                train_results = train_survival(
                    epoch + 1,
                    model,
                    train_dataset,
                    optimizer,
                    criterion,
                    agg_method=cfg.model.agg_method,
                    batch_size=cfg.training.batch_size,
                    gradient_accumulation=cfg.training.gradient_accumulation,
                )

                if cfg.wandb.enable:
                    update_log_dict(
                        f"train/fold_{i}",
                        train_results,
                        log_dict,
                        step=f"train/fold_{i}/epoch",
                        to_log=cfg.wandb.to_log,
                    )
                train_dataset.patient_df.to_csv(
                    Path(result_dir, f"train_{epoch}.csv"), index=False
                )

                if epoch % cfg.tuning.tune_every == 0:

                    tune_results = tune_survival(
                        epoch + 1,
                        model,
                        tune_dataset,
                        criterion,
                        agg_method=cfg.model.agg_method,
                        batch_size=cfg.tuning.batch_size,
                    )

                    auc, mean_auc, times = get_cumulative_dynamic_auc(
                        patient_dfs["train"],
                        patient_dfs["tune"],
                        tune_results["risks"],
                        cfg.label_name,
                    )
                    if cfg.wandb.enable:
                        update_log_dict(
                            f"tune/fold_{i}",
                            tune_results,
                            log_dict,
                            step=f"train/fold_{i}/epoch",
                            to_log=cfg.wandb.to_log,
                        )
                        if auc is not None:
                            fig = plot_cumulative_dynamic_auc(
                                auc, mean_auc, times, epoch
                            )
                            log_dict.update(
                                {
                                    f"tune/fold_{i}/cumulative_dynamic_auc": wandb.Image(
                                        fig
                                    )
                                }
                            )
                            plt.close(fig)
                    tune_dataset.patient_df.to_csv(
                        Path(result_dir, f"tune_{epoch}.csv"), index=False
                    )

                    early_stopping(epoch, model, tune_results)
                    if early_stopping.early_stop and cfg.early_stopping.enable:
                        stop = True

                lr = cfg.optim.lr
                if scheduler:
                    lr = scheduler.get_last_lr()[0]
                    scheduler.step()
                if cfg.wandb.enable:
                    wandb.define_metric(
                        f"train/fold_{i}/lr", step_metric=f"train/fold_{i}/epoch"
                    )
                    log_dict.update({f"train/fold_{i}/lr": lr})

                # logging
                if cfg.wandb.enable:
                    wandb.log(log_dict)

                epoch_end_time = time.time()
                epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
                tqdm.tqdm.write(
                    f"End of epoch {epoch+1} / {cfg.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
                )

                if stop:
                    tqdm.tqdm.write(
                        f"Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago"
                    )
                    break

        fold_end_time = time.time()
        fold_mins, fold_secs = compute_time(fold_start_time, fold_end_time)
        print(f"Total time taken for fold {i}: {fold_mins}m {fold_secs}s")

        # load best model
        best_model_fp = Path(
            checkpoint_dir, f"{cfg.testing.retrieve_checkpoint}_model.pt"
        )
        if cfg.wandb.enable:
            wandb.save(str(best_model_fp))
        best_model_sd = torch.load(best_model_fp)
        model.load_state_dict(best_model_sd)

        test_results = test_survival(
            model,
            test_dataset,
            agg_method=cfg.model.agg_method,
            batch_size=1,
        )
        test_dataset.patient_df.to_csv(Path(result_dir, f"test.csv"), index=False)

        for r, v in test_results.items():
            if r == "c-index":
                test_metrics.append(v)
                v = round(v, 3)
            if r in cfg.wandb.to_log and cfg.wandb.enable:
                wandb.log({f"test/fold_{i}/{r}": v})

    mean_test_metric = round(np.mean(test_metrics), 3)
    std_test_metric = round(statistics.stdev(test_metrics), 3)
    if cfg.wandb.enable:
        wandb.log({f"test/c-index_mean": mean_test_metric})
        wandb.log({f"test/c-index_std": std_test_metric})

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken ({nfold} folds): {mins}m {secs}s")


if __name__ == "__main__":

    main()
