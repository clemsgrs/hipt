import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Optional
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from source.vision_transformer import vit_small, vit4k_xs
from source.model_utils import Attn_Net_Gated, PositionalEncoderFactory
from source.utils import update_state_dict, get_device


class ModelFactory:
    def __init__(
        self,
        level: str,
        num_classes: int = 2,
        task: str = "classification",
        loss: Optional[str] = None,
        label_encoding: Optional[str] = None,
        model_options: Optional[DictConfig] = None,
    ):

        if task in ["classification", "survival"]:
            if level == "global":
                if model_options.agg_method == "self_att":
                    if model_options.slide_pos_embed.type == '2d' and model_options.slide_pos_embed.use:
                        self.model = GlobalPatientLevelCoordsHIPT(
                            num_classes=num_classes,
                            dropout=model_options.dropout,
                            slide_pos_embed=model_options.slide_pos_embed,
                        )
                    else:
                        self.model = GlobalPatientLevelHIPT(
                            num_classes=num_classes,
                            dropout=model_options.dropout,
                            slide_pos_embed=model_options.slide_pos_embed,
                        )
                elif model_options.agg_method == "concat" or not model_options.agg_method:
                    if model_options.slide_pos_embed.type == '2d' and model_options.slide_pos_embed.use:
                        self.model = GlobalCoordsHIPT(
                            num_classes=num_classes,
                            dropout=model_options.dropout,
                            slide_pos_embed=model_options.slide_pos_embed,
                        )
                    elif label_encoding == "ordinal":
                        if loss == 'coral':
                            self.model = GlobalCoralHIPT(
                                num_classes=num_classes,
                                dropout=model_options.dropout,
                                slide_pos_embed=model_options.slide_pos_embed,
                            )
                        else:
                            self.model = GlobalOrdinalHIPT(
                                num_classes=num_classes,
                                dropout=model_options.dropout,
                                slide_pos_embed=model_options.slide_pos_embed,
                            )
                    else:
                        self.model = GlobalHIPT(
                            num_classes=num_classes,
                            dropout=model_options.dropout,
                            slide_pos_embed=model_options.slide_pos_embed,
                        )
                else:
                    raise ValueError(
                        f"cfg.model.agg_method ({model_options.agg_method}) not supported"
                    )
            elif level == "local":
                if label_encoding == "ordinal":
                    if loss == 'coral':
                        self.model = LocalGlobalCoralHIPT(
                            num_classes=num_classes,
                            region_size=model_options.region_size,
                            patch_size=model_options.patch_size,
                            pretrain_vit_region=model_options.pretrain_vit_region,
                            freeze_vit_region=model_options.freeze_vit_region,
                            freeze_vit_region_pos_embed=model_options.freeze_vit_region_pos_embed,
                            dropout=model_options.dropout,
                            slide_pos_embed=model_options.slide_pos_embed,
                        )
                    else:
                        self.model = LocalGlobalOrdinalHIPT(
                            num_classes=num_classes,
                            region_size=model_options.region_size,
                            patch_size=model_options.patch_size,
                            pretrain_vit_region=model_options.pretrain_vit_region,
                            freeze_vit_region=model_options.freeze_vit_region,
                            freeze_vit_region_pos_embed=model_options.freeze_vit_region_pos_embed,
                            dropout=model_options.dropout,
                            slide_pos_embed=model_options.slide_pos_embed,
                        )
                else:
                    self.model = LocalGlobalHIPT(
                        num_classes=num_classes,
                        region_size=model_options.region_size,
                        patch_size=model_options.patch_size,
                        embed_dim_patch=model_options.embed_dim_patch,
                        embed_dim_region=model_options.embed_dim_region,
                        pretrain_vit_region=model_options.pretrain_vit_region,
                        freeze_vit_region=model_options.freeze_vit_region,
                        freeze_vit_region_pos_embed=model_options.freeze_vit_region_pos_embed,
                        dropout=model_options.dropout,
                        slide_pos_embed=model_options.slide_pos_embed,
                    )
            else:
                self.model = HIPT(
                    num_classes=num_classes,
                    region_size=model_options.region_size,
                    patch_size=model_options.patch_size,
                    mini_patch_size=model_options.mini_patch_size,
                    pretrain_vit_patch=model_options.pretrain_vit_patch,
                    freeze_vit_patch=model_options.freeze_vit_patch,
                    freeze_vit_patch_pos_embed=model_options.freeze_vit_patch_pos_embed,
                    pretrain_vit_region=model_options.pretrain_vit_region,
                    freeze_vit_region=model_options.freeze_vit_region,
                    freeze_vit_region_pos_embed=model_options.freeze_vit_region_pos_embed,
                    dropout=model_options.dropout,
                    slide_pos_embed=model_options.slide_pos_embed,
                )
        elif task == "regression":
            if level == "global":
                if model_options.agg_method == "self_att":
                    raise KeyError(f"aggregation method '{model_options.agg_method}' is not supported yet for {task} task")
                elif model_options.agg_method == "concat" or not model_options.agg_method:
                    self.model = GlobalRegressionHIPT(
                            num_classes=num_classes,
                            dropout=model_options.dropout,
                            slide_pos_embed=model_options.slide_pos_embed,
                        )
            elif level == "local":
                if model_options.agg_method == "self_att":
                    raise KeyError(f"aggregation method '{model_options.agg_method}' is not supported yet for {task} task")
                elif model_options.agg_method == "concat" or not model_options.agg_method:
                    self.model = LocalGlobalRegressionHIPT(
                            num_classes=num_classes,
                            region_size=model_options.region_size,
                            patch_size=model_options.patch_size,
                            pretrain_vit_region=model_options.pretrain_vit_region,
                            freeze_vit_region=model_options.freeze_vit_region,
                            freeze_vit_region_pos_embed=model_options.freeze_vit_region_pos_embed,
                            dropout=model_options.dropout,
                            slide_pos_embed=model_options.slide_pos_embed,
                        )


    def get_model(self):
        return self.model


class GlobalHIPT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        d_model: int = 192,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super(GlobalHIPT, self).__init__()
        self.slide_pos_embed = slide_pos_embed

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )

        if self.slide_pos_embed.use:
            pos_encoding_options = OmegaConf.create(
                {
                    "agg_method": "concat",
                    "dim": d_model,
                    "dropout": dropout,
                    "max_seq_len": slide_pos_embed.max_seq_len,
                    "max_nslide": slide_pos_embed.max_nslide,
                    "tile_size": slide_pos_embed.tile_size,
                }
            )
            self.pos_encoder = PositionalEncoderFactory(
                slide_pos_embed.type, slide_pos_embed.learned, pos_encoding_options
            ).get_pos_encoder()

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

    def forward(self, x):

        # x = [M, 192]
        x = self.global_phi(x)

        if self.slide_pos_embed.use:
            x = self.pos_encoder(x)

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

    def relocate(self, gpu_id: int = -1):
        device = get_device(gpu_id)
        self.global_phi = self.global_phi.to(device)
        if self.slide_pos_embed.use:
            self.pos_encoder = self.pos_encoder.to(device)
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


class LocalGlobalHIPT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        region_size: int = 4096,
        patch_size: int = 256,
        pretrain_vit_region: Optional[str] = None,
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        freeze_vit_region: bool = True,
        freeze_vit_region_pos_embed: bool = True,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super(LocalGlobalHIPT, self).__init__()
        self.npatch = int(region_size // patch_size)
        self.slide_pos_embed = slide_pos_embed

        checkpoint_key = "teacher"

        self.vit_region = vit4k_xs(
            img_size=region_size,
            patch_size=patch_size,
            input_embed_dim=embed_dim_patch,
            output_embed_dim=embed_dim_region,
        )

        if pretrain_vit_region and Path(pretrain_vit_region).is_file():
            print("Loading pretrained weights for region-level Transformer...")
            state_dict = torch.load(pretrain_vit_region, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(
                self.vit_region.state_dict(), state_dict
            )
            self.vit_region.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_vit_region}")
            print(msg)

        elif pretrain_vit_region:
            print(
                f"{pretrain_vit_region} doesnt exist ; please provide path to existing file"
            )

        if pretrain_vit_region and freeze_vit_region:
            print("Freezing pretrained region-level Transformer")
            for name, param in self.vit_region.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_vit_region_pos_embed)
            print(
                f"Region-level Transformer positional embedding layer frozen: {freeze_vit_region_pos_embed}"
            )
            print("Done")

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )

        if self.slide_pos_embed.use:
            pos_encoding_options = OmegaConf.create(
                {
                    "agg_method": "concat",
                    "dim": embed_dim_region,
                    "dropout": dropout,
                    "max_seq_len": slide_pos_embed.max_seq_len,
                    "max_nslide": slide_pos_embed.max_nslide,
                    "tile_size": slide_pos_embed.tile_size,
                }
            )
            self.pos_encoder = PositionalEncoderFactory(
                slide_pos_embed.type, slide_pos_embed.learned, pos_encoding_options
            ).get_pos_encoder()

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

    def forward(self, x):

        # x = [M, 256, 384]
        x = self.vit_region(
            x.unfold(1, self.npatch, self.npatch).transpose(1, 2)
        )  # [M, 192]
        x = self.global_phi(x)  # [M, 192]

        if self.slide_pos_embed.use:
            x = self.pos_encoder(x)

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

    def relocate(self, gpu_id: int = -1):
        device = get_device(gpu_id)
        self.vit_region = self.vit_region.to(device)

        self.global_phi = self.global_phi.to(device)
        if self.slide_pos_embed.use:
            self.pos_encoder = self.pos_encoder.to(device)
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


class HIPT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        region_size: int = 4096,
        patch_size: int = 256,
        mini_patch_size: int = 16,
        pretrain_vit_patch: str = "path/to/pretrained/vit_patch/weights.pth",
        freeze_vit_patch: bool = True,
        pretrain_vit_region: str = "path/to/pretrained/vit_region/weights.pth",
        freeze_vit_region: bool = True,
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        freeze_vit_patch_pos_embed: bool = True,
        freeze_vit_region_pos_embed: bool = True,
        split_across_gpus: bool = False,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super(HIPT, self).__init__()
        self.npatch = int(region_size // patch_size)
        self.num_patches = self.npatch**2
        self.ps = patch_size
        self.slide_pos_embed = slide_pos_embed

        self.split_across_gpus = split_across_gpus
        #TODO: if split_across_gpus:
            self.device_patch = torch.device("cuda:0")
            self.device_region = torch.device("cuda:1")

        checkpoint_key = "teacher"

        self.vit_patch = vit_small(
            img_size=patch_size,
            patch_size=mini_patch_size,
            embed_dim=embed_dim_patch,
        )

        if pretrain_vit_patch and Path(pretrain_vit_patch).is_file():
            print("Loading pretrained weights for patch-level Transformer...")
            state_dict = torch.load(pretrain_vit_patch, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_patch.state_dict(), state_dict)
            self.vit_patch.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_vit_patch}")
            print(msg)

        elif pretrain_vit_patch:
            print(
                f"{pretrain_vit_patch} doesnt exist ; please provide path to existing file"
            )

        if pretrain_vit_patch and freeze_vit_patch:
            print("Freezing pretrained patch-level Transformer")
            for name, param in self.vit_patch.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_vit_patch_pos_embed)
            print(
                f"Patch-level Transformer positional embedding layer frozen: {freeze_vit_patch_pos_embed}"
            )
            print("Done")

        if self.split_across_gpus:
            self.vit_patch.to(self.device_patch)

        self.vit_region = vit4k_xs(
            img_size=region_size,
            patch_size=patch_size,
            input_embed_dim=embed_dim_patch,
            output_embed_dim=embed_dim_region,
        )

        if pretrain_vit_region and Path(pretrain_vit_region).is_file():
            print("Loading pretrained weights for region-level Transformer...")
            state_dict = torch.load(pretrain_vit_region, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(
                self.vit_region.state_dict(), state_dict
            )
            self.vit_region.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_vit_region}")
            print(msg)

        elif pretrain_vit_region:
            print(
                f"{pretrain_vit_region} doesnt exist ; please provide path to existing file"
            )

        if pretrain_vit_region and freeze_vit_region:
            print("Freezing pretrained region-level Transformer")
            for name, param in self.vit_region.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_vit_region_pos_embed)
            print(
                f"Region-level Transformer positional embedding layer frozen: {freeze_vit_region_pos_embed}"
            )
            print("Done")

        if self.split_across_gpus:
            self.vit_region.to(self.device_region)

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )

        if self.slide_pos_embed.use:
            pos_encoding_options = OmegaConf.create(
                {
                    "agg_method": "concat",
                    "dim": embed_dim_region,
                    "dropout": dropout,
                    "max_seq_len": slide_pos_embed.max_seq_len,
                    "max_nslide": slide_pos_embed.max_nslide,
                    "tile_size": slide_pos_embed.tile_size,
                }
            )
            self.pos_encoder = PositionalEncoderFactory(
                slide_pos_embed.type, slide_pos_embed.learned, pos_encoding_options
            ).get_pos_encoder()

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

    def forward(self, x):

        # x = [M, 3, region_size, region_size]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, self.ps, self.ps).unfold(
            3, self.ps, self.ps
        )  # [M, 3, npatch, npatch, ps, ps]
        x = rearrange(
            x, "b c p1 p2 w h -> (b p1 p2) c w h"
        )  # [M*npatch*npatch, 3, ps, ps]
        if self.split_across_gpus:
            x = x.to(self.device_patch, non_blocking=True)  # [M*num_patches, 3, ps, ps]

        patch_features = []
        for mini_bs in range(0, x.shape[0], self.num_patches):
            minibatch = x[mini_bs : mini_bs + self.num_patches]
            f = self.vit_patch(minibatch).detach()  # [num_patches, 384]
            patch_features.append(f.unsqueeze(0))

        x = torch.vstack(patch_features)  # [M, num_patches, 384]
        if self.split_across_gpus:
            x = x.to(self.device_region, non_blocking=True)
        x = self.vit_region(
            x.unfold(1, self.npatch, self.npatch).transpose(1, 2)
        )  # x = [M, npatch, npatch, 384] -> [M, 192]

        x = self.global_phi(x)

        if self.slide_pos_embed.use:
            x = self.pos_encoder(x)

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

    def relocate(self, gpu_id: int = -1):
        device = get_device(gpu_id)
        if self.split_across_gpus:
            device = self.device_region
            self.vit_patch = self.vit_patch.to(self.device_patch)
        else:
            self.vit_patch = self.vit_patch.to(device)
        self.vit_region = self.vit_region.to(device)
        self.global_phi = self.global_phi.to(device)
        if self.slide_pos_embed.use:
            self.pos_encoder = self.pos_encoder.to(device)
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


class GlobalFeatureExtractor(nn.Module):
    def __init__(
        self,
        region_size: int = 4096,
        patch_size: int = 256,
        mini_patch_size: int = 16,
        pretrain_vit_patch: str = "path/to/pretrained/vit_patch/weights.pth",
        pretrain_vit_region: str = "path/to/pretrained/vit_region/weights.pth",
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        split_across_gpus: bool = False,
        verbose: bool = True,
    ):

        super(GlobalFeatureExtractor, self).__init__()
        checkpoint_key = "teacher"

        self.npatch = int(region_size // patch_size)
        self.ps = patch_size
        self.split_across_gpus = split_across_gpus

        if split_across_gpus:
            self.device_patch = torch.device("cuda:0")
            self.device_region = torch.device("cuda:1")

        self.vit_patch = vit_small(
            img_size=patch_size,
            patch_size=mini_patch_size,
            embed_dim=embed_dim_patch,
        )

        if Path(pretrain_vit_patch).is_file():
            if verbose:
                print("Loading pretrained weights for patch-level Transformer...")
            state_dict = torch.load(pretrain_vit_patch, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                if verbose:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_patch.state_dict(), state_dict)
            self.vit_patch.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"Pretrained weights found at {pretrain_vit_patch}")
                print(msg)

        elif verbose:
            print(
                f"{pretrain_vit_patch} doesnt exist ; please provide path to existing file"
            )

        if verbose:
            print("Freezing pretrained patch-level Transformer")
        for param in self.vit_patch.parameters():
            param.requires_grad = False
        if verbose:
            print("Done")

        if split_across_gpus:
            self.vit_patch.to(self.device_patch)

        self.vit_region = vit4k_xs(
            img_size=region_size,
            patch_size=patch_size,
            input_embed_dim=embed_dim_patch,
            output_embed_dim=embed_dim_region,
        )

        if Path(pretrain_vit_region).is_file():
            if verbose:
                print("Loading pretrained weights for region-level Transformer...")
            state_dict = torch.load(pretrain_vit_region, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                if verbose:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(
                self.vit_region.state_dict(), state_dict
            )
            self.vit_region.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"Pretrained weights found at {pretrain_vit_region}")
                print(msg)

        elif verbose:
            print(
                f"{pretrain_vit_region} doesnt exist ; please provide path to existing file"
            )

        if verbose:
            print("Freezing pretrained region-level Transformer")
        for param in self.vit_region.parameters():
            param.requires_grad = False
        if verbose:
            print("Done")

        if split_across_gpus:
            self.vit_region.to(self.device_region)

    def forward(self, x):

        # x = [1, 3, region_size, region_size]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, self.ps, self.ps).unfold(
            3, self.ps, self.ps
        )  # [1, 3, npatch, npatch, ps, ps]
        x = rearrange(
            x, "b c p1 p2 w h -> (b p1 p2) c w h"
        )  # [1*npatch*npatch, 3, ps, ps]
        if self.split_across_gpus:
            x = x.to(self.device_patch, non_blocking=True)  # [num_patches, 3, ps, ps]

        patch_features = self.vit_patch(x)  # [num_patches, 384]
        patch_features = patch_features.unsqueeze(0)  # [1, num_patches, 384]
        patch_features = patch_features.unfold(1, self.npatch, self.npatch).transpose(
            1, 2
        )  # [1, 384, npatch, npatch]
        if self.split_across_gpus:
            patch_features = patch_features.to(self.device_region, non_blocking=True)

        region_feature = self.vit_region(patch_features).cpu()  # [1, 192]

        return region_feature


class LocalFeatureExtractor(nn.Module):
    def __init__(
        self,
        patch_size: int = 256,
        mini_patch_size: int = 16,
        pretrain_vit_patch: str = "path/to/pretrained/vit_patch/weights.pth",
        embed_dim_patch: int = 384,
        verbose: bool = True,
    ):

        super(LocalFeatureExtractor, self).__init__()
        checkpoint_key = "teacher"

        self.ps = patch_size

        self.vit_patch = vit_small(
            img_size=patch_size,
            patch_size=mini_patch_size,
            embed_dim=embed_dim_patch,
        )

        if Path(pretrain_vit_patch).is_file():
            if verbose:
                print("Loading pretrained weights for patch-level Transformer")
            state_dict = torch.load(pretrain_vit_patch, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                if verbose:
                    print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_patch.state_dict(), state_dict)
            self.vit_patch.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"Pretrained weights found at {pretrain_vit_patch}")
                print(msg)

        elif verbose:
            print(
                f"{pretrain_vit_patch} doesnt exist ; please provide path to existing file"
            )

        if verbose:
            print("Freezing pretrained patch-level Transformer")
        for param in self.vit_patch.parameters():
            param.requires_grad = False
        if verbose:
            print("Done")

    def forward(self, x):

        # x = [1, 3, region_size, region_size]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, self.ps, self.ps).unfold(
            3, self.ps, self.ps
        )  # [1, 3, npatch, region_size, ps] -> [1, 3, npatch, npatch, ps, ps]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [num_patches, 3, ps, ps]

        patch_feature = self.vit_patch(x).detach().cpu()  # [num_patches, 384]

        return patch_feature


class GlobalPatientLevelHIPT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        embed_dim_slide: int = 192,
        embed_dim_patient: int = 192,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super(GlobalPatientLevelHIPT, self).__init__()
        self.slide_pos_embed = slide_pos_embed

        # from region to slide aggregation
        self.global_phi_slide = nn.Sequential(
            nn.Linear(embed_dim_region, embed_dim_slide), nn.ReLU(), nn.Dropout(dropout)
        )

        if self.slide_pos_embed.use:
            pos_encoding_options = OmegaConf.create(
                {
                    "agg_method": "self_att",
                    "dim": embed_dim_slide,
                    "dropout": dropout,
                    "max_seq_len": slide_pos_embed.max_seq_len,
                    "max_nslide": slide_pos_embed.max_nslide,
                    "tile_size": slide_pos_embed.tile_size,
                }
            )
            self.pos_encoder = PositionalEncoderFactory(
                slide_pos_embed.type, slide_pos_embed.learned, pos_encoding_options
            ).get_pos_encoder()

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

    def forward(self, x):

        # x = [ [M1, 192], [M2, 192], ...]
        N = len(x)
        slide_seq = []
        for n in range(N):
            y = x[n]
            y = self.global_phi_slide(y)
            if self.slide_pos_embed.use:
                y = self.pos_encoder(y)
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
        if self.slide_pos_embed.use:
            self.pos_encoder = self.pos_encoder.to(device)
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


class GlobalCoordsHIPT(GlobalHIPT):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        d_model: int = 192,
        tile_size: int = 4096,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super().__init__(num_classes, embed_dim_region, d_model, dropout, slide_pos_embed)
        self.tile_size = tile_size

    def forward(self, x, coords):

        # x = [M, 192]
        x = self.global_phi(x)

        if self.slide_pos_embed.use:
            coords = coords.squeeze(0)
            x = self.pos_encoder(x, coords)

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


class GlobalPatientLevelCoordsHIPT(GlobalPatientLevelHIPT):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        embed_dim_slide: int = 192,
        embed_dim_patient: int = 192,
        tile_size: int = 4096,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super().__init__(num_classes, embed_dim_region, embed_dim_slide, embed_dim_patient, dropout, slide_pos_embed)
        self.tile_size = tile_size

    def forward(self, x, coords):

        # x = [ [M1, 192], [M2, 192], ...]
        N = len(x)
        slide_seq = []
        for n in range(N):
            y = x[n]
            coord = coords[n]
            y = self.global_phi_slide(y)
            if self.slide_pos_embed.use:
                y = self.pos_encoder(y, coord)
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


class GlobalRegressionHIPT(GlobalHIPT):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        d_model: int = 192,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super().__init__(num_classes, embed_dim_region, d_model, dropout, slide_pos_embed)
        self.classifier = nn.Linear(192, 1)


class GlobalOrdinalHIPT(GlobalHIPT):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        d_model: int = 192,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super().__init__(num_classes, embed_dim_region, d_model, dropout, slide_pos_embed)
        self.classifier = nn.Linear(192, num_classes-1)


class GlobalCoralHIPT(GlobalHIPT):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_region: int = 192,
        d_model: int = 192,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super().__init__(num_classes, embed_dim_region, d_model, dropout, slide_pos_embed)
        self.classifier = nn.Linear(192, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(num_classes-1).float()).cuda()

    def forward(self, x):

        # x = [M, 192]
        x = self.global_phi(x)

        if self.slide_pos_embed.use:
            x = self.pos_encoder(x)

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)
        logits = logits + self.b

        return logits


class LocalGlobalOrdinalHIPT(LocalGlobalHIPT):
    def __init__(
        self,
        num_classes: int = 2,
        region_size: int = 4096,
        patch_size: int = 256,
        pretrain_vit_region: Optional[str] = None,
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        freeze_vit_region: bool = True,
        freeze_vit_region_pos_embed: bool = True,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super().__init__(num_classes, region_size, patch_size, pretrain_vit_region, embed_dim_patch, embed_dim_region,freeze_vit_region, freeze_vit_region_pos_embed, dropout, slide_pos_embed)
        self.classifier = nn.Linear(192, num_classes-1)


class LocalGlobalCoralHIPT(LocalGlobalHIPT):
    def __init__(
        self,
        num_classes: int = 2,
        region_size: int = 4096,
        patch_size: int = 256,
        pretrain_vit_region: Optional[str] = None,
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        freeze_vit_region: bool = True,
        freeze_vit_region_pos_embed: bool = True,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super().__init__(num_classes, region_size, patch_size, pretrain_vit_region, embed_dim_patch, embed_dim_region,freeze_vit_region, freeze_vit_region_pos_embed, dropout, slide_pos_embed)
        self.classifier = nn.Linear(192, 1, bias=False)
        self.num_classes = num_classes
        self.b = nn.Parameter(torch.zeros(self.num_classes-1, device='cuda').float())

    def forward(self, x):

        # x = [M, 256, 384]
        x = self.vit_region(
            x.unfold(1, self.npatch, self.npatch).transpose(1, 2)
        )  # [M, 192]
        x = self.global_phi(x)  # [M, 192]

        if self.slide_pos_embed.use:
            x = self.pos_encoder(x)

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)
        logits = logits + self.b

        return logits

    def relocate(self, gpu_id: int = -1):
        device = get_device(gpu_id)
        self.vit_region = self.vit_region.to(device)

        self.global_phi = self.global_phi.to(device)
        if self.slide_pos_embed.use:
            self.pos_encoder = self.pos_encoder.to(device)
        self.global_transformer = self.global_transformer.to(device)
        self.global_attn_pool = self.global_attn_pool.to(device)
        self.global_rho = self.global_rho.to(device)

        self.classifier = self.classifier.to(device)
        self.b = nn.Parameter(torch.zeros(self.num_classes-1, device=device).float())


class LocalGlobalRegressionHIPT(LocalGlobalHIPT):
    def __init__(
        self,
        num_classes: int = 2,
        region_size: int = 4096,
        patch_size: int = 256,
        pretrain_vit_region: Optional[str] = None,
        embed_dim_patch: int = 384,
        embed_dim_region: int = 192,
        freeze_vit_region: bool = True,
        freeze_vit_region_pos_embed: bool = True,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super().__init__(num_classes, region_size, patch_size, pretrain_vit_region, embed_dim_patch, embed_dim_region,freeze_vit_region, freeze_vit_region_pos_embed, dropout, slide_pos_embed)
        self.classifier = nn.Linear(192, 1)