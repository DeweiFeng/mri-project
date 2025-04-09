import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)
from pipeline_mri import MRIDiffusionPipeline
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from mri.utils import clear, get_mask, normalize_np, real_to_nchw_comp
from mri.mri import MulticoilMRI
from sigpy.mri import poisson
from skimage.metrics import peak_signal_noise_ratio
from utils import set_seed, calculate_ssim, calculate_lpips


def main(args):

    set_seed(args.seed)

    args.save_dir = (
        Path(args.save_dir)
        / f"{args.mask_type}"
        / f"acc_{args.acc_factor}"
        / f"cfg{args.cfg_scale}"
        / f"eta{args.eta}"
    )
    args.save_dir.mkdir(exist_ok=True, parents=True)

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

    text_encoder.eval()
    unet.eval()

    pipeline = MRIDiffusionPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        config_path=args.model_config,
    )
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=args.num_timesteps)

    image_size = 512 if args.mri_type == "skm-tea" else 320

    x = (
        torch.tensor(np.load("./assets/fastmri/knee/file1001429/slice/020.npy"))
        .unsqueeze(0)
        .unsqueeze(0)
    )  # [1, 1, H, W]
    mps = torch.tensor(
        np.load("./assets/fastmri/knee/file1001429/mps/020.npy")
    ).unsqueeze(
        0
    )  # [1, N, H, W]
    with open("./assets/fastmri/knee/file1001429/meta/prompt.txt") as f:
        prompt = f.readline().strip()

    print(
        f"Use metadata '''{prompt}''' for reconstruction with cfg scale {args.cfg_scale}"
    )
    prompt = [prompt]

    B = x.shape[0]
    _, N_coil, h, w = mps.shape

    x = x.to(device)
    mps = mps.to(device)

    # Define forward operator
    if args.mask_type == "poisson2d":
        mask = poisson(
            (image_size, image_size), accel=args.acc_factor, dtype=np.float32
        )
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        mask = mask.to(device)
    else:
        mask = get_mask(
            torch.zeros([B, 1, image_size, image_size]),
            image_size,
            B,
            type=args.mask_type,
            acc_factor=args.acc_factor,
            center_fraction=args.center_fraction,
        )
        mask = mask.to(device)

    if args.cfg_scale == 0:
        prompt = [""] * B

    A_funcs = MulticoilMRI(
        image_size, mask, mps
    )  # mask: float32, mps: complex64 -> float32 broadcast to complex64 with equal number
    y = A_funcs.A(x)
    ATy = A_funcs.AT(y)

    recon = pipeline.dds(
        prompt=prompt,
        guidance_scale=args.cfg_scale,
        sample_size=image_size,
        num_inference_steps=args.num_timesteps,
        eta=args.eta,
        A_funcs=A_funcs,
        y=y,
        gamma=args.gamma,
        CG_iter=args.CG_iter,
    )
    recon = real_to_nchw_comp(recon)

    np.save(os.path.join(args.save_dir, f"recon.npy"), clear(recon))
    plt.imsave(
        os.path.join(args.save_dir, f"input.png"), np.abs(clear(ATy)), cmap="gray"
    )
    plt.imsave(
        os.path.join(args.save_dir, f"input_mask.png"), np.abs(clear(mask)), cmap="gray"
    )
    plt.imsave(os.path.join(args.save_dir, f"label.png"), np.abs(clear(x)), cmap="gray")
    plt.imsave(
        os.path.join(args.save_dir, f"recon.png"), np.abs(clear(recon)), cmap="gray"
    )

    # Metric computation
    x = normalize_np(np.abs(clear(x)))
    recon = normalize_np(np.abs(clear(recon)))

    psnr = peak_signal_noise_ratio(x, recon)
    ssim = calculate_ssim(x, recon)
    lpips_score = calculate_lpips(x, recon, device=device)
    print(f"PSNR: {psnr:04f}, SSIM: {ssim:04f}, LPIPS: {lpips_score:04f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI-Reconstruction-Argument")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./MRI_checkpoint",
        help="directory of checkpoint",
    )
    parser.add_argument(
        "--model_config", type=str, default="./configs/model_index.json"
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=100,
        help="Number of timesteps for inference",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./result/recon_complex_multi",
        help="The directory for saving generated images",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducible outputs"
    )
    parser.add_argument("--eta", type=float, default=0.8, help="DDIM eta")

    # MRI-related arguments
    parser.add_argument(
        "--mask_type",
        type=str,
        default="uniform1d",
        help="masking type in the Fourier domain",
    )
    parser.add_argument(
        "--acc_factor", type=int, default=4, help="severity of undersampling"
    )
    parser.add_argument(
        "--center_fraction", type=float, default=0.08, help="severity of undersampling"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=5.0,
        help="regularization weight inversely proportional to proximal step size",
    )
    parser.add_argument(
        "--CG_iter", type=int, default=5, help="Num CG iter per timestep. Default is 5"
    )
    parser.add_argument(
        "--mri_type", type=str, choices=["fastmri", "skm-tea"], default="fastmri"
    )

    args = parser.parse_args()
    main(args)
