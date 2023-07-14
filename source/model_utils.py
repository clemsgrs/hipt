import math
import torch
import torch.nn as nn
import numpy as np

from typing import Optional
from omegaconf import DictConfig

from source.utils import get_device


def cantor_diagonal(p: int, q: int):
    return (p + q) * (1 + p + q) / 2 + q


class PositionalEncoderFactory:
    def __init__(
        self,
        type: str,
        learned: bool,
        options: Optional[DictConfig] = None,
    ):
        if type == "1d":
            if learned:
                self.pos_encoder = PositionalEmbedding(options.dim, options.max_seq_len)
            else:
                self.pos_encoder = PositionalEncoding(
                    options.dim, options.dropout, options.max_seq_len
                )
        elif type == "2d":
            if learned:
                if options.agg_method == "concat":
                    self.pos_encoder = ConcatPositionalEmbedding2d(
                        options.tile_size,
                        options.dim,
                        options.max_seq_len,
                        options.max_nslide,
                    )
                elif options.agg_method == "self_att":
                    self.pos_encoder = PositionalEmbedding2d(
                        options.tile_size, options.dim, options.max_seq_len
                    )
            else:
                if options.agg_method == "concat":
                    self.pos_encoder = ConcatPositionalEncoding2d(
                        options.dim,
                        options.dropout,
                        options.max_seq_len,
                        options.max_nslide,
                    )
                elif options.agg_method == "self_att":
                    self.pos_encoder = PositionalEncoding2d(
                        options.dim, options.dropout, options.max_seq_len
                    )
        else:
            raise ValueError(f"cfg.model.slide_pos_embed.type ({type}) not supported")

    def get_pos_encoder(self):
        return self.pos_encoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
        """
        x = x.unsqueeze(1)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x).squeeze(1)


class PositionalEncoding2d(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        X = torch.arange(max_len)
        Y = torch.arange(max_len)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, max_len, d_model)
        for x in X:
            for y in Y:
                position = cantor_diagonal(x, y)
                pe[x, y, 0::2] = torch.sin(position * div_term)
                pe[x, y, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x, coords):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
            coords: Tensor, shape [seq_len, 2]
        """
        x = x + self.pe[coords[:, 0], coords[:, 1]]
        return self.dropout(x)


class ConcatPositionalEncoding2d(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1000,
        max_nslide: int = 10,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        X = torch.arange(max_len)
        Y = torch.arange(max_len)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, max_len, d_model)
        for x in X:
            for y in Y:
                position = cantor_diagonal(x, y)
                pe[x, y, 0::2] = torch.sin(position * div_term)
                pe[x, y, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        slide_pos = torch.arange(max_nslide).unsqueeze(1)
        slide_div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        slide_pe = torch.zeros(max_nslide, 1, d_model)
        slide_pe[:, 0, 0::2] = torch.sin(slide_pos * slide_div_term)
        slide_pe[:, 0, 1::2] = torch.cos(slide_pos * slide_div_term)
        self.register_buffer("slide_pe", slide_pe.squeeze(1))

    def forward(self, x, coords):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
            coords: Tensor, shape [seq_len, 3]
        """
        slide_idx, coord = coords[:, 0], coords[:, 1:]
        x = x + self.pe[coord[:, 0], coord[:, 1]] + self.slide_pe[slide_idx]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 3000, gpu_id: int = -1):
        super().__init__()
        device = get_device(gpu_id)
        self.pos_ids = torch.arange(max_len).to(device)
        self.embedder = nn.Embedding(max_len, dim)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
        """
        seq_length = x.shape[0]
        position_ids = self.pos_ids[:seq_length]
        position_embeddings = self.embedder(position_ids)
        x += position_embeddings
        return x


class PositionalEmbedding2d(nn.Module):
    def __init__(self, tile_size: int, dim: int, max_len: int = 512):
        super().__init__()
        self.tile_size = tile_size
        self.embedder_x = nn.Embedding(max_len, dim // 2)
        self.embedder_y = nn.Embedding(max_len, dim // 2)

    def get_grid_values(self, coords: np.ndarray):
        m = coords.min()
        grid_coords = torch.div(coords - m, self.tile_size, rounding_mode="floor")
        return grid_coords

    def forward(self, x, coords):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
            coords: Tensor, shape [seq_len, 3]
        """
        coord_x = self.get_grid_values(coords[:, 1])
        coord_y = self.get_grid_values(coords[:, 2])
        embedding_x = self.embedder_x(coord_x)
        embedding_y = self.embedder_y(coord_y)
        position_embedding = torch.cat([embedding_x, embedding_y], dim=1)
        x += position_embedding
        return x


class ConcatPositionalEmbedding2d(nn.Module):
    def __init__(
        self, tile_size: int, dim: int, max_len: int = 512, max_nslide: int = 10
    ):
        super().__init__()
        self.tile_size = tile_size
        self.embedder_x = nn.Embedding(max_len, dim // 2)
        self.embedder_y = nn.Embedding(max_len, dim // 2)
        self.embedder_slide_pos = nn.Embedding(max_nslide, dim)

    def get_grid_values(self, coords: np.ndarray):
        m = coords.min()
        grid_coords = torch.div(coords - m, self.tile_size, rounding_mode="floor")
        return grid_coords

    def forward(self, x, coords):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
            coords: Tensor, shape [seq_len, 3]
        """
        slide_idx, coord = coords[:, 0], coords[:, 1:]
        coord_x = self.get_grid_values(coord[:, 0])
        coord_y = self.get_grid_values(coord[:, 1])
        embedding_x = self.embedder_x(coord_x)
        embedding_y = self.embedder_y(coord_y)
        position_embedding = torch.cat([embedding_x, embedding_y], dim=1)
        slide_pos_embedding = self.embedder_slide_pos(slide_idx)
        x = x + position_embedding + slide_pos_embedding
        return x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, num_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            num_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x num_classes
        return A, x
