import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Optional
from einops import rearrange
from omegaconf import DictConfig

from source.vision_transformer import vit_small, vit4k_xs
from source.model_utils import Attn_Net_Gated
from source.utils import update_state_dict


class ModelFactory:
    def __init__(
        self,
        level: str,
        num_classes: int = 2,
        model_opt: Optional[DictConfig] = None,
    ):

        if level == "global":
            self.model = GlobalHIPT(
                num_classes=num_classes,
                dropout=model_opt.dropout,
            )
        elif level == "local":
            self.model = LocalGlobalHIPT(
                num_classes=num_classes,
                img_size_4096=model_opt.img_size_4096,
                patch_size_4096=model_opt.patch_size_4096,
                pretrain_4096=model_opt.pretrain_4096,
                freeze_4096=model_opt.freeze_4096,
                freeze_4096_pos_embed=model_opt.freeze_4096_pos_embed,
                dropout=model_opt.dropout,
            )
        else:
            self.model = HIPT(
                num_classes=num_classes,
                pretrain_256=model_opt.pretrain_256,
                freeze_256=model_opt.freeze_256,
                freeze_256_pos_embed=model_opt.freeze_256_pos_embed,
                img_size_256=model_opt.img_size_256,
                patch_size_256=model_opt.patch_size_256,
                pretrain_4096=model_opt.pretrain_4096,
                freeze_4096=model_opt.freeze_4096,
                freeze_4096_pos_embed=model_opt.freeze_4096_pos_embed,
                img_size_4096=model_opt.img_size_4096,
                patch_size_4096=model_opt.patch_size_4096,
                dropout=model_opt.dropout,
            )

    def get_model(self):
        return self.model


class GlobalHIPT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim_4096: int = 192,
        dropout: float = 0.25,
    ):

        super(GlobalHIPT, self).__init__()
        self.num_classes = num_classes

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_4096, 192), nn.ReLU(), nn.Dropout(dropout)
        )
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
        pretrain_4096: str = "path/to/pretrained/vit_4096/weights.pth",
        embed_dim_256: int = 384,
        embed_dim_4096: int = 192,
        img_size_4096: int = 3584,
        patch_size_4096: int = 256,
        freeze_4096: bool = True,
        freeze_4096_pos_embed: bool = True,
        dropout: float = 0.25,
    ):

        super(LocalGlobalHIPT, self).__init__()
        self.num_classes = num_classes

        checkpoint_key = "teacher"

        self.vit_4096 = vit4k_xs(
            img_size=img_size_4096,
            patch_size=patch_size_4096,
            input_embed_dim=embed_dim_256,
            output_embed_dim=embed_dim_4096,
        )

        if Path(pretrain_4096).is_file():
            print("Loading pretrained weights for ViT_4096 model...")
            state_dict = torch.load(pretrain_4096, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_4096.state_dict(), state_dict)
            self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_4096}")
            print(msg)

        else:
            print(
                f"{pretrain_4096} doesnt exist ; please provide path to existing file"
            )

        if freeze_4096:
            print("Freezing pretrained ViT_4096 model")
            for name, param in self.vit_4096.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_4096_pos_embed)
            print(
                f"ViT_4096 positional embedding layer frozen: {freeze_4096_pos_embed}"
            )
            print("Done")

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_4096, 192), nn.ReLU(), nn.Dropout(dropout)
        )
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
        x = self.vit_4096(x.unfold(1, 16, 16).transpose(1, 2))  # [M, 192]
        x = self.global_phi(x)  # [M, 192]

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
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.vit_4096 = nn.DataParallel(self.vit_4096, device_ids=device_ids).to(
                "cuda:0"
            )

        self.global_phi = self.global_phi.to(device)
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
        pretrain_256: str = "path/to/pretrained/vit_256/weights.pth",
        freeze_256: bool = True,
        pretrain_4096: str = "path/to/pretrained/vit_4096/weights.pth",
        freeze_4096: bool = True,
        img_size_256: int = 224,
        patch_size_256: int = 16,
        embed_dim_256: int = 384,
        img_size_4096: int = 3584,
        patch_size_4096: int = 256,
        embed_dim_4096: int = 192,
        freeze_256_pos_embed: bool = True,
        freeze_4096_pos_embed: bool = True,
        dropout: float = 0.25,
    ):

        super(HIPT, self).__init__()
        self.num_classes = num_classes

        checkpoint_key = "teacher"

        self.vit_256 = vit_small(
            img_size=img_size_256,
            patch_size=patch_size_256,
            embed_dim=embed_dim_256,
        )

        if Path(pretrain_256).is_file():
            print("Loading pretrained weights for ViT_256 model...")
            state_dict = torch.load(pretrain_256, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_256}")
            print(msg)

        else:
            print(f"{pretrain_256} doesnt exist ; please provide path to existing file")

        if freeze_256:
            print("Freezing pretrained ViT_256 model")
            for name, param in self.vit_256.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_256_pos_embed)
            print("Done")

        self.vit_4096 = vit4k_xs(
            img_size=img_size_4096,
            patch_size=patch_size_4096,
            input_embed_dim=embed_dim_256,
            output_embed_dim=embed_dim_4096,
        )

        if Path(pretrain_4096).is_file():
            print("Loading pretrained weights for ViT_4096 model...")
            state_dict = torch.load(pretrain_4096, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_4096.state_dict(), state_dict)
            self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_4096}")
            print(msg)

        else:
            print(
                f"{pretrain_4096} doesnt exist ; please provide path to existing file"
            )

        if freeze_4096:
            print("Freezing pretrained ViT_4096 model")
            for name, param in self.vit_4096.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_4096_pos_embed)
            print("Done")

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_4096, 192), nn.ReLU(), nn.Dropout(dropout)
        )
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

        # x = [M, 3, 4096, 4096]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, 256, 256).unfold(3, 256, 256)  # [M, 3, 16, 16, 256, 256]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [M*16*16, 3, 256, 256]
        x = x.to(self.device_256, non_blocking=True)  # [M*256, 3, 256, 256]

        # x = self.vit_256(x)                                     # [M, 256, 384]
        features_256 = []
        for mini_bs in range(0, x.shape[0], 256):
            minibatch = x[mini_bs : mini_bs + 256]
            f = self.vit_256(minibatch).detach()  # [256, 384]
            features_256.append(f.unsqueeze(0))

        x = torch.vstack(features_256)  # [M, 256, 384]
        x = x.to(self.device_4096, non_blocking=True)
        x = self.vit_4096(
            x.unfold(1, 16, 16).transpose(1, 2)
        )  # x = [M, 16, 16, 384] -> [M, 192]

        x = x.to(self.device_256, non_blocking=True)
        x = self.global_phi(x)

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
        if device == torch.device("cuda"):
            assert torch.cuda.device_count() >= 2
            self.device_256 = torch.device("cuda:0")
            self.device_4096 = torch.device("cuda:1")
            device = self.device_256
        else:
            self.device_256 = device
            self.device_4096 = device

        self.vit_256 = self.vit_256.to(self.device_256)
        self.vit_4096 = self.vit_4096.to(self.device_4096)

        self.global_phi = self.global_phi.to(device)
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
        pretrain_256: str = "path/to/pretrained/vit_256/weights.pth",
        pretrain_4096: str = "path/to/pretrained/vit_4096/weights.pth",
        embed_dim_256: int = 384,
        embed_dim_4096: int = 192,
    ):

        super(GlobalFeatureExtractor, self).__init__()
        checkpoint_key = "teacher"

        self.device_256 = torch.device("cuda:0")
        self.device_4096 = torch.device("cuda:1")

        self.vit_256 = vit_small(
            img_size=224,
            patch_size=16,
            embed_dim=embed_dim_256,
        )

        if Path(pretrain_256).is_file():
            print("Loading pretrained weights for ViT_256 model...")
            state_dict = torch.load(pretrain_256, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_256}")
            print(msg)

        else:
            print(f"{pretrain_256} doesnt exist ; please provide path to existing file")

        print("Freezing pretrained ViT_256 model")
        for param in self.vit_256.parameters():
            param.requires_grad = False
        print("Done")

        self.vit_256.to(self.device_256)

        self.vit_4096 = vit4k_xs(
            img_size=3584,
            patch_size=256,
            input_embed_dim=embed_dim_256,
            output_embed_dim=embed_dim_4096,
        )

        if Path(pretrain_4096).is_file():
            print("Loading pretrained weights for ViT_4096 model...")
            state_dict = torch.load(pretrain_4096, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_4096.state_dict(), state_dict)
            self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_4096}")
            print(msg)

        else:
            print(
                f"{pretrain_4096} doesnt exist ; please provide path to existing file"
            )

        print("Freezing pretrained ViT_4096 model")
        for param in self.vit_4096.parameters():
            param.requires_grad = False
        print("Done")

        self.vit_4096.to(self.device_4096)

    def forward(self, x):

        # x = [1, 3, 4096, 4096]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, 256, 256).unfold(3, 256, 256)  # [1, 3, 16, 16, 256, 256]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [1*16*16, 3, 256, 256]
        x = x.to(self.device_256, non_blocking=True)  # [256, 3, 256, 256]

        features_256 = self.vit_256(x)  # [256, 384]
        features_256 = features_256.unsqueeze(0)  # [1, 256, 384]
        features_256 = features_256.unfold(1, 16, 16).transpose(
            1, 2
        )  # [1, 384, 16, 16]
        features_256 = features_256.to(self.device_4096, non_blocking=True)

        feature_4096 = self.vit_4096(features_256).cpu()  # [1, 192]

        return feature_4096


class LocalFeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrain_256: str = "path/to/pretrained/vit_256/weights.pth",
        embed_dim_256: int = 384,
    ):

        super(LocalFeatureExtractor, self).__init__()
        checkpoint_key = "teacher"

        self.device_256 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vit_256 = vit_small(
            img_size=224,
            patch_size=16,
            embed_dim=embed_dim_256,
        )

        if Path(pretrain_256).is_file():
            print("Loading pretrained weights for ViT_256 model...")
            state_dict = torch.load(pretrain_256, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_256}")
            print(msg)

        else:
            print(f"{pretrain_256} doesnt exist ; please provide path to existing file")

        print("Freezing pretrained ViT_256 model")
        for param in self.vit_256.parameters():
            param.requires_grad = False
        print("Done")

        self.vit_256.to(self.device_256)

    def forward(self, x):

        # x = [1, 3, 4096, 4096]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, 256, 256).unfold(
            3, 256, 256
        )  # [1, 3, 16, 4096, 256] -> [1, 3, 16, 16, 256, 256]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [256, 3, 256, 256]
        x = x.to(self.device_256, non_blocking=True)

        feature_256 = self.vit_256(x).detach().cpu()  # [256, 384]

        return feature_256
