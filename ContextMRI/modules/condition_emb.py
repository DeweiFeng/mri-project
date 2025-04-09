from collections import defaultdict
import json
import math
import random
from typing import Optional, List

import torch
from torch import einsum, nn
import torch.nn.functional as F


def exists(val):
    return val is not None


class SinusoidalEmbedding(nn.Module):
    # Ref: https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        assert (dim % 2) == 0

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = torch.exp(-math.log(theta) * freq_seq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, pos: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None
    ):
        assert exists(pos) or exists(x)
        if pos is None:
            # x: (B, T, D)
            pos = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ContinuousConditionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        max_time_steps: int = 10000,
    ):
        super().__init__()

        if exists(range_min) and exists(range_max) and range_min >= range_max - 1e-6:
            print(
                f"Invalid range_min {range_min} and range_max {range_max}, using range_max = range_min + 1"
            )
            range_max = range_min + 1

        self.range_min = range_min
        self.range_max = range_max
        self.dim = dim
        self.emb = SinusoidalEmbedding(dim, max_time_steps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, )

        # Normalize x to [0, 1]
        if exists(self.range_min) and exists(self.range_max):
            x = x.clamp(min=self.range_min, max=self.range_max)
            x = (x - self.range_min) / (self.range_max - self.range_min)

        return self.mlp(self.emb(pos=x))


class DiscreteConditionEmbedding(nn.Module):
    def __init__(self, dim: int, num_classes: int, str_to_idx=None):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.str_to_idx = str_to_idx
        self.emb = nn.Embedding(num_classes, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, ) or (B, num_classes)

        if x.dim() == 1:
            return self.mlp(self.emb(x))
        elif x.dim() == 2:
            # NOTE: each row of x could contain multiple classes
            x_sum = x.sum(1)
            x[x_sum > 0] = x[x_sum > 0] / x_sum[x_sum > 0].unsqueeze(1)
            emb = F.linear(x, self.emb.weight.t())
            return self.mlp(emb)
        else:
            raise ValueError(
                f"Input tensor x must be of shape (B,) or (B, num_classes), but got {x.shape}"
            )


class ConditionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        metadata_stats: str,
        cfg_prob: float = 0.0,
        cfg_strategy: str = "joint",
    ):
        super().__init__()

        assert cfg_strategy in {
            "joint",
            "independent",
        }, f"cfg_strategy {cfg_strategy} not supported"

        self.dim = dim
        self.cfg_prob = cfg_prob
        self.cfg_strategy = cfg_strategy

        with open(metadata_stats, "r") as f:
            metadata_stats = json.load(f)
        self.metadata_stats = metadata_stats

        self.embeddings = nn.ModuleDict()
        for key, value in metadata_stats.items():
            if isinstance(value, dict):
                # Continuous condition embedding
                range_min = value["min"]
                range_max = value["max"]
                self.embeddings[key] = ContinuousConditionEmbedding(
                    dim=dim, range_min=range_min, range_max=range_max
                )
            else:
                # Discrete condition embedding
                num_classes = len(value)
                if num_classes == 0:
                    continue
                str_to_idx = {v: i for i, v in enumerate(value)}
                self.embeddings[key] = DiscreteConditionEmbedding(
                    dim=dim, num_classes=num_classes, str_to_idx=str_to_idx
                )

    def forward(
        self,
        metadata_list: List[dict],
        dtype=None,
        device=None,
        select_keys: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = torch.device("cpu")

        batch = defaultdict(list)
        # Convert metadata for the condition embedding
        for metadata in metadata_list:
            for key, value in metadata.items():
                if key not in self.embeddings or (
                    select_keys is not None and key not in select_keys
                ):
                    continue

                if isinstance(value, (int, float)):
                    batch[key].append(value)
                elif isinstance(value, str):
                    if value not in self.embeddings[key].str_to_idx:
                        continue
                    batch[key].append(self.embeddings[key].str_to_idx[value])
                elif isinstance(value, list):
                    if len(value) == 0:
                        continue
                    labels_tensor = torch.zeros((self.embeddings[key].num_classes,))
                    for v in value:
                        if v not in self.embeddings[key].str_to_idx:
                            continue
                        labels_tensor[self.embeddings[key].str_to_idx[v]] = 1.0
                    batch[key].append(labels_tensor)
                else:
                    raise ValueError(
                        f"Unsupported value type {type(value)} for key {key}"
                    )

        drop_indices = None
        if self.training and self.cfg_prob > 0 and self.cfg_strategy == "joint":
            drop_indices = torch.randperm(
                len(metadata_list), device=device
            ) < self.cfg_prob * len(metadata_list)

        emb_list = []
        # Compute condition embeddings
        for key, value in batch.items():
            # print(f"key: {key}, value: {value}")
            if len(value) == 0:
                continue

            if isinstance(value[0], float):
                value = torch.tensor(value, dtype=dtype, device=device)
            elif isinstance(value[0], int):
                value = torch.tensor(value, dtype=torch.long, device=device)
            elif isinstance(value[0], torch.Tensor):
                value = torch.stack(value, dim=0).to(dtype=dtype, device=device)

            # print(value)

            # value: (B, ) or (B, num_classes)
            emb = self.embeddings[key](value)

            # print(emb)

            # CFG
            if (
                self.training
                and self.cfg_prob > 0
                and self.cfg_strategy == "independent"
            ):
                drop_indices = torch.randperm(
                    len(metadata_list), device=device
                ) < self.cfg_prob * len(metadata_list)
            if drop_indices is not None:
                emb[drop_indices] = 0.0

            # print(emb)
            # print()

            emb_list.append(emb)

        # print(emb_list)

        if len(emb_list) == 0:
            return torch.zeros(
                (len(metadata_list), self.dim), dtype=dtype, device=device
            )
        else:
            return torch.stack(emb_list, dim=0).sum(0)
