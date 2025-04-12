import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)
from pipeline_mri import MRIDiffusionPipeline
from utils import row_to_text_string_skm_tea, row_to_text_string, save_image
from mri.utils import real_to_nchw_comp, clear

from modules import ConditionEmbedding

def preprocess_metadata(row_dict):
    if "slice" in row_dict and "slice_number" not in row_dict:
        row_dict["slice_number"] = row_dict.pop("slice")
    return row_dict


def main(args):
    # Set device (note: fix torch.cuda.is_available() call by adding parentheses)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # Load UNet according to MRI type.
    if args.mri_type == "fastmri":
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="fastmri"
        )
    elif args.mri_type == "skm-tea":
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="skm-tea"
        )
    else:
        raise ValueError(f"Not supported mri data type {args.mri_type}")

    # NEW
    if args.enable_condition_emb:
        if hasattr(unet, "condition_emb"):
            print("ConditionEmbedding has already attached to UNet.")
        else:
            condition_emb = ConditionEmbedding(
                dim=unet.config.cross_attention_dim,
                metadata_stats=args.metadata_stats,
                cfg_prob=args.inference_cfg_prob,
                cfg_strategy=args.cfg_strategy,
            )
            # NEW: Load the separately-saved weights if the file is provided.
            if args.condition_emb_weights is not None:
                weight_dict = torch.load(args.condition_emb_weights, map_location=device)
                condition_emb.load_state_dict(weight_dict)
                print("ConditionEmbedding weights loaded from", args.condition_emb_weights)
            else:
                print("No condition embedding weights file provided; using random initialization.")
            unet.add_module("condition_emb", condition_emb)
            print("ConditionEmbedding attached to UNet.")

    unet.eval()
    text_encoder.eval()

    pipeline = MRIDiffusionPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        config_path=args.model_config,
    )
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
    pipeline.scheduler.eta = args.eta

    # NEW: Prepare metadata and prompt for conditional MRI generation.
    # If using auto generation, we fetch a row from CSV, extract both the prompt and raw metadata.
    if args.use_auto:
        row_index = 10
        if args.mri_type == "fastmri":
            df = pd.read_csv("./assets/metadata_brain_val.csv")
            row = df.iloc[row_index]
            prompts = row_to_text_string(row, p=1.0)
            print(prompts)
            raw_metadata = row.to_dict()
            metadata_list = [preprocess_metadata(raw_metadata)]
        elif args.mri_type == "skm-tea":
            df = pd.read_csv("./assets/skm-tea/metadata_val.csv")
            row = df.iloc[row_index]
            prompts = row_to_text_string_skm_tea(row, p=1.0)
            raw_metadata = row.to_dict()
            metadata_list = [preprocess_metadata(raw_metadata)]
    else:
        prompts = args.meta_prompt
        metadata_list = None  # If not auto, you may pass in custom metadata if available.

    print(f"Generated Image from metadata prompts: {prompts} and metadata list {metadata_list} with CFG scale {args.cfg_scale}")

    # NEW: Pass metadata_list to the pipeline call.
    generated_mri = pipeline(
        prompt=[prompts],
        guidance_scale=args.cfg_scale,
        metadata_list=metadata_list  # <--- New parameter passed in
    )["images"]

    # Convert 2-channel image into grayscale
    output = np.abs(real_to_nchw_comp(generated_mri))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    save_image(output, f"{args.output_dir}/sample.png")
    print(f"Successfully generated image from metadata, saved in {args.output_dir}/sample.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Inference")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--model_config", type=str, default="./configs/model_index.json")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./MRI_checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--mri_type", type=str, choices=["fastmri", "skm-tea"], default="fastmri")
    parser.add_argument(
        "--use_auto",
        type=bool,
        default=True,
        help="Recommend to use auto generation of metadata to sync the training text distribution",
    )
    parser.add_argument(
        "--meta_prompt",
        type=str,
        default="",
        help="Use customized metadata prompt for generation. Please match the format of the auto generation",
    )
    # NEW: Add condition embedding arguments to the inference parser.
    parser.add_argument("--enable_condition_emb", action="store_true", default=False, help="Enable condition embedding during inference.")
    parser.add_argument("--condition_emb_weights", type=str, default=None,
                        help="Path to the pre-saved ConditionEmbedding weights")
    parser.add_argument("--metadata_stats", type=str, default=None, help="Path to the metadata statistics JSON file for condition embedding.")
    parser.add_argument("--cfg_strategy", type=str, choices=["joint", "independent"], default="joint", help="Classifier-free guidance strategy for conditioning.")
    parser.add_argument(
    "--inference_cfg_prob",
    type=float,
    default=0.0,
    help="Drop probability for classifier-free guidance conditioning during inference. Set to 0 to use full condition, or >0 to drop parts."
)
    
    args = parser.parse_args()
    main(args)

"""
with cond emb
python inference.py   --cfg_scale 1.0   --num_inference_steps 100   --model_config ./configs/model_index.json   --pretrained_model_name_or_path D:\Research\data\MRI_checkpoint   --output_dir ./output_cond   --mri_type fastmri   --enable_condition_emb   --metadata_stats ./metadata_stats.json   --use_auto True --condition_emb_weights D:\\Research\\data\\test_ckpt_0410\\converted\\condition\\condition_emb.pth
"""

"""
without cond emb
python inference.py   --cfg_scale 1.0   --num_inference_steps 50   --model_config ./configs/model_index.json   --pretrained_model_name_or_path D:\Research\data\MRI_checkpoint   --output_dir ./output_no_cond   --mri_type fastmri   --metadata_stats ./metadata_stats.json   --use_auto True
"""