import argparse
import os

import torch
from diffusers import UNet2DConditionModel
import json

### After training finish, you can convert checkpoint .pth into .safetensors using this .py to make
### easy integration of Stable-Diffusion pipeline.


def split_state_dict(state_dict, prefix: str):
    state_dict_1 = {}
    state_dict_2 = {}
    for key in state_dict.keys():
        if key.startswith(prefix):
            state_dict_1[key.replace(prefix, "")] = state_dict[key]
        else:
            state_dict_2[key] = state_dict[key]
    return state_dict_1, state_dict_2


def main(pth_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {pth_path}...")
    with open("./configs/unet/config_mri.json", "r") as f:
        config = json.load(f)
    # Path to your .pth file
    # pth_path = "../output/checkpoint-step-#/unet_ema_0.999_weights.pth"
    # pth_path = "/home/work/dohun2/MRI-DM/output_clip/checkpoint-step-140000/unet_weights.pth"
    # Load the state dictionary from the .pth file
    state_dict = torch.load(pth_path, map_location="cpu")

    # Split the state_dict into two parts
    condition_emb_state_dict, unet_state_dict = split_state_dict(
        state_dict, "condition_emb."
    )

    # Save the condition embedding state_dict separately
    # Optionally: create a Hugging Face model instance if the model architecture is known
    print("Loading UNet2DConditionModel...")
    model = UNet2DConditionModel.from_config(config)
    model.load_state_dict(unet_state_dict)

    # Save the model in Hugging Face's format (.bin)
    print(f"Saving model to {os.path.join(save_dir, 'unet')}")
    model.save_pretrained(os.path.join(save_dir, "unet"))

    if len(condition_emb_state_dict) > 0:
        # Save the condition embedding state_dict separately
        print(f"Saving condition embedding to {os.path.join(save_dir, 'condition')}")
        os.makedirs(os.path.join(save_dir, "condition"), exist_ok=True)
        torch.save(
            condition_emb_state_dict,
            os.path.join(save_dir, "condition", "condition_emb.pth"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pth_path", type=str, default=None, help="Path to the .pth file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the converted model",
    )
    args = parser.parse_args()

    main(**vars(args))


"""
python3 convert_checkpoint.py \
    --pth_path /data/sls/scratch/hengjui/6S982/exp/test1/checkpoint-step-100/unet_ema_0.999_weights.pth \
    --save_dir /data/sls/scratch/hengjui/6S982/exp/test1/converted
"""
