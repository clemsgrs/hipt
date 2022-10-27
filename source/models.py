import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Optional
from einops import rearrange
from functools import partial
from torchvision import transforms

from source.vision_transformer import vit_small, vit4k_xs
from source.model_utils import Attn_Net_Gated
from source.utils import update_state_dict


class GlobalHIPT(nn.Module):
    def __init__(
        self,
        size_arg: str = 'small',
        num_classes: int = 1,
        dropout: float = 0.25,
        embed_dim: int = 192,
        hidden_dim: int = 192,
    ):

        super(GlobalHIPT, self).__init__()
        self.size_dict_path = {'small': [384, 192, 192], 'big': [1024, 512, 384]}
        size = self.size_dict_path[size_arg]

        # Global aggregation
        self.global_phi = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=3, dim_feedforward=embed_dim, dropout=dropout, activation='relu'
            ),
            num_layers=2
        )
        self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=dropout, num_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[1], num_classes)


    def forward(self, x):

        # x = [B, M, 192]
        x = self.global_phi(x)
        x = self.global_transformer(x).squeeze(0)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)

        return logits


class LocalGlobalHIPT(nn.Module):
    def __init__(
        self,
        path_input_dim=384,
        size_arg = "small",
        dropout=0.25,
        num_classes=4,
        pretrain_4k='None',
        freeze_4k=False,
        pretrain_WSI='None',
        freeze_WSI=False,
    ):

        super(LocalGlobalHIPT, self).__init__()
        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_vit = vit4k_xs()
        if pretrain_4k != 'None':
            print("Loading Pretrained Local VIT model...",)
            state_dict = torch.load('../../HIPT_4K/Checkpoints/%s.pth' % pretrain_4k, map_location='cpu')['teacher']
            state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
            state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.local_vit.load_state_dict(state_dict, strict=False)
            print("Done!")
        if freeze_4k:
            print("Freezing Pretrained Local VIT model")
            for param in self.local_vit.parameters():
                param.requires_grad = False
            print("Done")

        ### Global Aggregation
        self.pretrain_WSI = pretrain_WSI
        if pretrain_WSI != 'None':
            pass
        else:
            self.global_phi = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Dropout(0.25))
            self.global_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=192, nhead=3, dim_feedforward=192, dropout=0.25, activation='relu'
                ),
                num_layers=2
            )
            self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, num_classes=1)
            self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])

        self.classifier = nn.Linear(size[1], num_classes)


    def forward(self, x):

        # for a given WSI, x should be a tensor of shape [M, 256, 384]
        # where M = number for [4096,4096] regions in the WSI

        ### Local
        h_4096 = self.local_vit(x.unfold(1, 16, 16).transpose(1,2))

        ### Global
        if self.pretrain_WSI != 'None':
            h_WSI = self.global_vit(x.unsqueeze(dim=0))
        else:
            h_4096 = self.global_phi(x)
            h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
            att_4096, h_4096 = self.global_attn_pool(h_4096)
            att_4096 = torch.transpose(att_4096, 1, 0)
            att_4096 = F.softmax(att_4096, dim=1)
            h_4096_att = torch.mm(att_4096, h_4096)
            h_WSI = self.global_rho(h_4096_att)

        logits = self.classifier(h_WSI)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits, F.softmax(logits, dim=1), Y_hat


class HIPT(nn.Module):
    def __init__(
        self,
        size_arg='small',
        dropout: float = 0.25,
        num_classes: int = 2,
        pretrain_256: Optional[str] = None,
        freeze_256: bool = False,
        pretrain_4096: Optional[str] = None,
        freeze_4096: bool = False,
    ):

        super(HIPT, self).__init__()
        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        size = self.size_dict_path[size_arg]

        device_256 = torch.device('cuda:0')
        device_4096 = torch.device('cuda:1')

        self.vit_256 = vit_small(
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        if Path(pretrain_256).isfile():
            print('Loading pretrained weights for ViT_256 model...')
            state_dict = torch.load(pretrain_256, map_location='cpu')
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f'Take key {checkpoint_key} in provided checkpoint dict')
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            msg = self.vit_256.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {pretrain_256} and loaded with msg: {msg}')

        if freeze_256:
            print('Freezing pretrained ViT_256 model')
            for param in self.vit_256.parameters():
                param.requires_grad = False
            print('Done')

        self.vit_4096 = vit4k_xs(
            num_classes=0,
            patch_size=256,
            img_size=[4096],
            input_embed_dim=384,
            output_embed_dim=192,
            depth=6,
            num_heads=6,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        if Path(pretrain_4096).isfile():
            print('Loading pretrained weights for ViT_4096 model...')
            state_dict = torch.load(pretrain_4096, map_location='cpu')
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f'Take key {checkpoint_key} in provided checkpoint dict')
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            msg = self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {pretrain_4096} and loaded with msg: {msg}')

        if freeze_4096:
            print('Freezing pretrained ViT_4096 model')
            for param in self.vit_4096.parameters():
                param.requires_grad = False
            print('Done')

        # Global aggregation

        self.global_phi = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Dropout(0.25))
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192, nhead=3, dim_feedforward=192, dropout=0.25, activation='relu'
            ),
            num_layers=2
        )
        self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, num_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])

        self.classifier = nn.Linear(size[1], num_classes)


    def forward(self, x):

        # x = [M, 256, 3, 256, 256]
        M = x.shape[0]

        features_4096 = []
        for m in range(M):
            # region = [256, 3, 256, 256]
            region = x[m].to(self.device_256, non_blocking=True)
            features_256 = self.vit_256(region).detach().cpu() # [256, 384]

            features_256 = features_256.reshape(16, 16, 384)
            features_256 = features_256.transpose(0,1)
            features_256 = features_256.transpose(0,2)
            features_256 = features_256.unsqueeze(0)
            features_256 = features_256.to(self.device_4096, non_blocking=True)

            feature_4096 = self.vit_4096.forward(features_256)
            features_4096.append(feature_4096) # [1, 192]

        features_4096 = torch.vstack(features_4096) # [M, 192]

        features_4096 = self.global_phi(features_4096)
        features_4096 = self.global_transformer(features_4096.unsqueeze(1)).squeeze(1)
        att_4096, features_4096 = self.global_attn_pool(features_4096)
        att_4096 = torch.transpose(att_4096, 1, 0)
        att_4096 = F.softmax(att_4096, dim=1)
        features_4096_att = torch.mm(att_4096, features_4096)
        features_WSI = self.global_rho(features_4096_att)

        logits = self.classifier(features_WSI)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits, F.softmax(logits, dim=1), Y_hat


class HIPT_4096(nn.Module):
    def __init__(
        self,
        size_arg='small',
        dropout: float = 0.25,
        num_classes: int = 2,
        pretrain_256: Optional[str] = None,
        freeze_256: bool = False,
        pretrain_4096: Optional[str] = None,
        freeze_4096: bool = False,
    ):

        super(HIPT_4096, self).__init__()
        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        size = self.size_dict_path[size_arg]
        checkpoint_key = 'teacher'

        self.device_256 = torch.device('cuda:0')
        self.device_4096 = torch.device('cuda:1')

        self.vit_256 = vit_small(img_size=256, patch_size=16, embed_dim=384)

        if Path(pretrain_256).is_file():
            print('Loading pretrained weights for ViT_256 model...')
            state_dict = torch.load(pretrain_256, map_location='cpu')
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f'Take key {checkpoint_key} in provided checkpoint dict')
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {pretrain_256}')
            print(msg)

        if freeze_256:
            print('Freezing pretrained ViT_256 model')
            for param in self.vit_256.parameters():
                param.requires_grad = False
            print('Done')

        self.vit_256.to(self.device_256)

        self.vit_4096 = vit4k_xs(
            img_size=4096,
            patch_size=256,
            input_embed_dim=384,
            output_embed_dim=192,
            num_classes=0,
        )

        if Path(pretrain_4096).is_file():
            print('Loading pretrained weights for ViT_4096 model...')
            state_dict = torch.load(pretrain_4096, map_location='cpu')
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f'Take key {checkpoint_key} in provided checkpoint dict')
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_4096.state_dict(), state_dict)
            self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {pretrain_4096}')
            print(msg)

        if freeze_4096:
            print('Freezing pretrained ViT_4096 model')
            for param in self.vit_4096.parameters():
                param.requires_grad = False
            print('Done')

        self.vit_4096.to(self.device_4096)

        # Global aggregation

        self.global_phi = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Dropout(0.25)).to(self.device_256)
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192, nhead=3, dim_feedforward=192, dropout=0.25, activation='relu'
            ),
            num_layers=2
        ).to(self.device_256)
        self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, num_classes=1).to(self.device_256)
        self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)]).to(self.device_256)

        self.classifier = nn.Linear(size[1], num_classes).to(self.device_256)


    def forward(self, x):

        # x = [B, M, 3, 4096, 4096]
        M = x.shape[1]
        # print(f'x.shape: {x.shape}')

        features_4096 = []
        for m in range(M):
            batch_256 = x[:, m, ...]                                        # 1. [B, 3, 4096, 4096]
            # print(f'batch_256.shape: {batch_256.shape}')
            batch_256, _, _ = self.prepare_img_tensor(batch_256)            # 1. [B, 3, 4096, 4096]
            # print(f'batch_256.shape: {batch_256.shape}')
            batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)           # 2. [B, 3, 16, 16, 256, 256]
            # print(f'batch_256.shape: {batch_256.shape}')
            batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 3. [B*256, 3, 256, 256]
            # print(f'batch_256.shape: {batch_256.shape}')
            batch_256 = batch_256.to(self.device_256, non_blocking=True)

            features_256 = self.vit_256(batch_256).detach().cpu()   # [B*256, 384]
            # print(f'features_256.shape: {features_256.shape}')

            features_256 = features_256.reshape(16, 16, 384)        # [16, 16, 384] (we assume B=1)
            features_256 = features_256.transpose(0,1)
            features_256 = features_256.transpose(0,2)              # [384, 16, 16]
            features_256 = features_256.unsqueeze(0)                # [1, 384, 16, 16]
            # print(f'features_256.shape: {features_256.shape}')
            features_256 = features_256.to(self.device_4096, non_blocking=True)

            feature_4096 = self.vit_4096.forward(features_256)      # [1, 192]
            # print(f'feature_4096.shape: {feature_4096.shape}')
            features_4096.append(feature_4096)

        features_4096 = torch.stack(features_4096, dim=1)           # [1, M, 192]
        # print(f'features_4096.shape: {features_4096.shape}')

        features_4096 = features_4096.to(self.device_256, non_blocking=True)
        features_4096 = self.global_phi(features_4096.squeeze(0))   # [M, 192]
        # print(f'features_4096.shape: {features_4096.shape}')
        features_4096 = self.global_transformer(features_4096.unsqueeze(1)).squeeze(1)
        # print(f'features_4096.shape: {features_4096.shape}')
        att_4096, features_4096 = self.global_attn_pool(features_4096)
        att_4096 = torch.transpose(att_4096, 1, 0)
        att_4096 = F.softmax(att_4096, dim=1)
        features_4096_att = torch.mm(att_4096, features_4096)
        features_WSI = self.global_rho(features_4096_att)

        logits = self.classifier(features_WSI)

        return logits

    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)
        return img_new, w_256, h_256


class GlobalFeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrain_256: str = 'path/to/pretrained/vit_256/weights.pth',
        pretrain_4096: str = 'path/to/pretrained/vit_4096/weights.pth',
        embed_dim_256: int = 384,
        embed_dim_4096: int = 192,
    ):

        super(GlobalFeatureExtractor, self).__init__()
        checkpoint_key = 'teacher'

        self.device_256 = torch.device('cuda:0')
        self.device_4096 = torch.device('cuda:1')

        self.vit_256 = vit_small(img_size=256, patch_size=16, embed_dim=embed_dim_256)

        if Path(pretrain_256).is_file():
            print('Loading pretrained weights for ViT_256 model...')
            state_dict = torch.load(pretrain_256, map_location='cpu')
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f'Take key {checkpoint_key} in provided checkpoint dict')
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {pretrain_256}')
            print(msg)

        else:
            print(f'{pretrain_256} doesnt exist ; please provide path to existing file')

        print('Freezing pretrained ViT_256 model')
        for param in self.vit_256.parameters():
            param.requires_grad = False
        print('Done')

        self.vit_256.to(self.device_256)

        self.vit_4096 = vit4k_xs(
            img_size=4096,
            patch_size=256,
            input_embed_dim=embed_dim_256,
            output_embed_dim=embed_dim_4096,
            num_classes=0,
        )

        if Path(pretrain_4096).is_file():
            print('Loading pretrained weights for ViT_4096 model...')
            state_dict = torch.load(pretrain_4096, map_location='cpu')
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f'Take key {checkpoint_key} in provided checkpoint dict')
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_4096.state_dict(), state_dict)
            self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {pretrain_4096}')
            print(msg)

        else:
            print(f'{pretrain_4096} doesnt exist ; please provide path to existing file')

        print('Freezing pretrained ViT_4096 model')
        for param in self.vit_4096.parameters():
            param.requires_grad = False
        print('Done')

        self.vit_4096.to(self.device_4096)


    def forward(self, x):

        # x = [1, 3, 4096, 4096]
        x = x.unfold(2, 256, 256).unfold(3, 256, 256)
        x = rearrange(x, 'b c p1 p2 w h -> (b p1 p2) c w h')
        x = x.to(self.device_256, non_blocking=True)

        features_256 = self.vit_256(x).detach().cpu()

        features_256 = features_256.reshape(16, 16, 384)
        features_256 = features_256.transpose(0,1)
        features_256 = features_256.transpose(0,2)
        features_256 = features_256.unsqueeze(0)
        features_256 = features_256.to(self.device_4096, non_blocking=True)

        feature_4096 = self.vit_4096(features_256)

        return feature_4096


class LocalFeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrain_256: str = 'path/to/pretrained/vit_256/weights.pth',
        embed_dim_256: int = 384,
    ):

        super(LocalFeatureExtractor, self).__init__()
        checkpoint_key = 'teacher'

        self.device_256 = torch.device('cuda:0')

        self.vit_256 = vit_small(img_size=256, patch_size=16, embed_dim=embed_dim_256)

        if Path(pretrain_256).is_file():
            print('Loading pretrained weights for ViT_256 model...')
            state_dict = torch.load(pretrain_256, map_location='cpu')
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f'Take key {checkpoint_key} in provided checkpoint dict')
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {pretrain_256}')
            print(msg)

        else:
            print(f'{pretrain_256} doesnt exist ; please provide path to existing file')

        print('Freezing pretrained ViT_256 model')
        for param in self.vit_256.parameters():
            param.requires_grad = False
        print('Done')

        self.vit_256.to(self.device_256)

    def forward(self, x):

        # x = [1, 3, 4096, 4096]
        x = x.unfold(2, 256, 256).unfold(3, 256, 256)           # [1, 3, 16, 4096, 256] -> [1, 3, 16, 16, 256, 256]
        x = rearrange(x, 'b c p1 p2 w h -> (b p1 p2) c w h')    # [256, 3, 256, 256]
        x = x.to(self.device_256, non_blocking=True)

        feature_256 = self.vit_256(x).detach().cpu()            # [256, 384]

        return feature_256