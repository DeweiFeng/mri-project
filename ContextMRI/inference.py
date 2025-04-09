import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)
from pipeline_mri import MRIDiffusionPipeline
import pandas as pd
from utils import row_to_text_string_skm_tea, row_to_text_string, save_image
from mri.utils import real_to_nchw_comp, clear


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
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

    # Conditional MRI generation
    if args.use_auto:
        row_index = 150
        if args.mri_type == "fastmri":
            df = pd.read_csv("./assets/fastmri/knee/metadata_val.csv")
            row = df.iloc[row_index]
            prompts = row_to_text_string(row, p=1.0)

        elif args.mri_type == "skm-tea":
            df = pd.read_csv("./assets/skm-tea/metadata_val.csv")
            row = df.iloc[row_index]
            prompts = row_to_text_string_skm_tea(row, p=1.0)

    else:
        prompts = args.meta_prompt

    print(f"Generated Image from metadata: {prompts} with CFG scale {args.cfg_scale}")

    generated_mri = pipeline(
        prompt=[prompts],
        guidance_scale=args.cfg_scale,
    )["images"]

    # Convert 2-channel image into grayscale
    output = np.abs(real_to_nchw_comp(generated_mri))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    save_image(output, f"{args.output_dir}/sample.png")
    print(
        f"Successfully generated image from metadata, saved in {args.output_dir}/sample.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Inference")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--model_config", type=str, default="./configs/model_index.json"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="./MRI_checkpoint"
    )
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--mri_type", type=str, choices=["fastmri", "skm-tea"], default="fastmri"
    )
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
    args = parser.parse_args()
    main(args)
