import logging

import timm
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms

import source.distributed as distributed
import source.vision_transformer as vits
from source.augmentations import MaybeToTensor, make_normalize_transform
from source.utils import update_state_dict

logger = logging.getLogger("slide2vec")


class FoundationModelFactory:
    def __init__(
        self,
        options: DictConfig,
    ):
        if options.level == "tile":
            if options.name == "virchow":
                model = Virchow(input_size=options.tile_size)
            elif options.name == "virchow2":
                model = Virchow2(input_size=options.tile_size)
            elif options.name == "uni":
                model = UNI(input_size=options.tile_size)
            elif options.name == "prov-gigapath":
                model = ProvGigaPath(input_size=options.tile_size)
            elif options.name == "h-optimus-0":
                model = Hoptimus0(input_size=options.tile_size)
            elif options.name is None and options.arch:
                model = DINOViT(
                    arch=options.arch,
                    pretrained_weights=options.pretrained_weights,
                    input_size=options.tile_size,
                )
        elif options.level == "region":
            if options.name == "virchow":
                tile_encoder = Virchow(input_size=options.patch_size)
            elif options.name == "virchow2":
                tile_encoder = Virchow2(input_size=options.patch_size)
            elif options.name == "uni":
                tile_encoder = UNI(input_size=options.patch_size)
            elif options.name == "prov-gigapath":
                tile_encoder = ProvGigaPath(input_size=options.patch_size)
            elif options.name == "h-optimus-0":
                tile_encoder = Hoptimus0(input_size=options.patch_size)
            elif options.name is None and options.arch:
                tile_encoder = DINOViT(
                    arch=options.arch,
                    pretrained_weights=options.pretrained_weights,
                    input_size=options.patch_size,
                )
            model = RegionFeatureExtractor(tile_encoder)
        else:
            raise ValueError(f"{options.level} not supported")

        self.model = model.eval()
        self.model = self.model.to(self.model.device)

    def get_model(self):
        return self.model


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.encoder = self.build_encoder()
        self.set_device()

    def set_device(self):
        if distributed.is_enabled():
            self.device = torch.device(f"cuda:{distributed.get_local_rank()}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    def build_encoder(self):
        raise NotImplementedError

    def get_transforms(self):
        data_config = resolve_data_config(
            self.encoder.pretrained_cfg, model=self.encoder
        )
        transform = create_transform(**data_config)
        return transform

    def forward(self, x, **kwargs):
        raise NotImplementedError


class DINOViT(FeatureExtractor):
    def __init__(
        self,
        arch: str,
        pretrained_weights: str,
        input_size: int = 256,
        patch_size: int = 14,
        ckpt_key: str = "teacher",
    ):
        self.arch = arch
        self.pretrained_weights = pretrained_weights
        self.input_size = input_size
        self.patch_size = patch_size
        self.ckpt_key = ckpt_key
        arch2dim = {"vit_large": 1024, "vit_base": 768, "vit_small": 384}
        self.features_dim = arch2dim[arch]
        super(DINOViT, self).__init__()
        self.load_weights()

    def load_weights(self):
        if distributed.is_main_process():
            print(f"Loading pretrained weights from: {self.pretrained_weights}")
        state_dict = torch.load(self.pretrained_weights, map_location="cpu")
        if self.ckpt_key:
            state_dict = state_dict[self.ckpt_key]
        nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix="module."
        )
        nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix="backbone."
        )
        state_dict, msg = update_state_dict(self.encoder.state_dict(), state_dict)
        if distributed.is_main_process():
            print(msg)
        self.encoder.load_state_dict(state_dict, strict=True)

    def build_encoder(self):
        encoder = vits.__dict__[self.arch](
            img_size=self.input_size, patch_size=self.patch_size
        )
        return encoder

    def get_transforms(self):
        if self.input_size > 224:
            transform = transforms.Compose(
                [
                    MaybeToTensor(),
                    transforms.CenterCrop(224),
                    make_normalize_transform(),
                ]
            )
        else:
            transforms.Compose(
                [
                    MaybeToTensor(),
                    make_normalize_transform(),
                ]
            )
        return transform

    def forward(self, x, **kwargs):
        return self.encoder(x)


class UNI(FeatureExtractor):
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.features_dim = 1024
        super(UNI, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        if self.input_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        encoder.pretrained_cfg[
            "interpolation"
        ] = "bicubic"  # Match interpolation if needed
        return encoder

    def forward(self, x, **kwargs):
        return self.encoder(x)


class Virchow(FeatureExtractor):
    def __init__(self, input_size: int = 224, mode: str = "cls"):
        self.input_size = input_size
        self.mode = mode
        self.features_dim = 1280
        if mode == "full":
            self.features_dim = 2560
        super(Virchow, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        if self.input_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        return encoder

    def forward(self, x, **kwargs):
        output = self.encoder(x)
        class_token = output[:, 0]  # size: 1 x 1280
        patch_tokens = output[
            :, 1:
        ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        if self.mode == "cls":
            return class_token
        elif self.mode == "full":
            embedding = torch.cat(
                [class_token, patch_tokens.mean(1)], dim=-1
            )  # size: 1 x 2560
            return embedding


class Virchow2(FeatureExtractor):
    def __init__(self, input_size: int = 224, mode: str = "cls"):
        self.input_size = input_size
        self.mode = mode
        self.features_dim = 1280
        if mode == "full":
            self.features_dim = 2560
        super(Virchow2, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        if self.input_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        return encoder

    def forward(self, x, **kwargs):
        output = self.encoder(x)
        class_token = output[:, 0]  # size: 1 x 1280
        patch_tokens = output[
            :, 5:
        ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        if self.mode == "cls":
            return class_token
        elif self.mode == "full":
            embedding = torch.cat(
                [class_token, patch_tokens.mean(1)], dim=-1
            )  # size: 1 x 2560
            return embedding


class ProvGigaPath(FeatureExtractor):
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.features_dim = 1536
        super(ProvGigaPath, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
        )
        if self.input_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        return encoder

    def forward(self, x, **kwargs):
        return self.encoder(x)


class Hoptimus0(FeatureExtractor):
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.features_dim = 1536
        super(Hoptimus0, self).__init__()

    def build_encoder(self):
        encoder = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        if self.input_size == 256:
            encoder.pretrained_cfg["input_size"] = [3, 224, 224]
            encoder.pretrained_cfg["crop_pct"] = 224 / 256  # ensure Resize is 256
        return encoder

    def forward(self, x, **kwargs):
        return self.encoder(x)


class RegionFeatureExtractor(nn.Module):
    def __init__(
        self,
        tile_encoder: nn.Module,
        tile_size: int = 256,
    ):
        super(RegionFeatureExtractor, self).__init__()
        self.tile_encoder = tile_encoder
        self.tile_size = tile_size
        self.device = self.tile_encoder.device
        self.features_dim = self.tile_encoder.features_dim

    def get_transforms(self):
        return self.tile_encoder.get_transforms()

    def forward(self, x, **kwargs):
        # x = [num_tiles, 3, 224, 224]
        output = self.tile_encoder(x)  # [num_tiles, features_dim]
        return output