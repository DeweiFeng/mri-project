import argparse
import sys

sys.path.append("..")

import torch

from modules import (
    ContinuousConditionEmbedding,
    DiscreteConditionEmbedding,
    ConditionEmbedding,
)


@torch.no_grad()
def test_condition_emb(metadata_stats: str):
    batch_size = 16
    emb_dim = 4

    # ContinuousConditionEmbedding
    cont_cond_emb = ContinuousConditionEmbedding(
        dim=emb_dim, range_min=0.0, range_max=1.0, max_time_steps=10000
    )
    x = torch.rand(batch_size)

    cont_emb = cont_cond_emb(x)
    assert cont_emb.shape == (
        batch_size,
        emb_dim,
    ), f"Expected shape ({batch_size}, {emb_dim}), but got {cont_emb.shape}"
    print("Continuous condition embedding test passed.")

    # DiscreteConditionEmbedding
    num_classes = 10
    disc_cond_emb = DiscreteConditionEmbedding(dim=emb_dim, num_classes=num_classes)
    # Case 1
    x = torch.randint(0, num_classes, (batch_size,))
    disc_emb = disc_cond_emb(x)
    assert disc_emb.shape == (
        batch_size,
        emb_dim,
    ), f"Expected shape ({batch_size}, {emb_dim}), but got {disc_emb.shape}"
    print("Discrete condition embedding case 1 test passed.")

    # Case 2
    x = torch.rand(batch_size, num_classes)
    x = x > 0.8
    x = x.float()
    disc_emb = disc_cond_emb(x)
    assert disc_emb.shape == (
        batch_size,
        emb_dim,
    ), f"Expected shape ({batch_size}, {emb_dim}), but got {disc_emb.shape}"
    print("Discrete condition embedding case 2 test passed.")

    # ConditionEmbedding
    metadata_list = [
        {
            "anatomy": "brain",
            "slice_number": 1,
            "contrast": "AX FLAIR_FBB",
            "sequence": "TurboSpinEcho",
            "TR": 9000,
            "TE": 81,
            "TI": 2500,
            "flip_angle": 150,
        },
        {
            "anatomy": "brain",
            "slice_number": 12,
            "contrast": "AX FLAIR_FBB",
            "sequence": "TurboSpinEcho",
            "TR": 9000,
            "TE": 81,
            "TI": 2500,
            "flip_angle": 150,
        },
    ]
    cond_emb = ConditionEmbedding(
        dim=emb_dim,
        metadata_stats=metadata_stats,
        cfg_prob=0.5,
        cfg_strategy="independent",
    )
    cond_emb.train()
    print(cond_emb)
    emb = cond_emb(metadata_list)
    if isinstance(emb, torch.Tensor):
        assert emb.shape == (
            2,
            emb_dim,
        ), f"Expected shape (2, {emb_dim}), but got {emb.shape}"
    print("Condition embedding test passed. (cfg_strategy: independent)")

    cond_emb = ConditionEmbedding(
        dim=emb_dim,
        metadata_stats=metadata_stats,
        cfg_prob=0.5,
        cfg_strategy="joint",
    )
    cond_emb.train()
    print(cond_emb)
    emb = cond_emb(metadata_list)
    if isinstance(emb, torch.Tensor):
        assert emb.shape == (
            2,
            emb_dim,
        ), f"Expected shape (2, {emb_dim}), but got {emb.shape}"
    print("Condition embedding test passed. (cfg_strategy: joint)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_stats",
        type=str,
        required=True,
        help="Path to the metadata statistics file.",
    )
    args = parser.parse_args()

    test_condition_emb(**vars(args))


"""
python3 test_condition_emb.py \
    --metadata_stats ../metadata_stats.json
"""
