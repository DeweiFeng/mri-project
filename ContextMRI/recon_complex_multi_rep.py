import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file as safe_load
import json

from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from dataset_mri import MRIDataset
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)
from pipeline_mri import MRIDiffusionPipeline
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader

from mri.utils import clear, get_mask, normalize_np, real_to_nchw_comp
from mri.mri import MulticoilMRI
from sigpy.mri import poisson
from utils import calculate_ssim, calculate_lpips, batch_update_json, set_seed, count_entries_in_json

from modules import ConditionEmbedding


def main(args):

    set_seed(args.seed)

    args.save_dir = Path(args.save_dir) / f"{args.mask_type}" / f"acc_{args.acc_factor}" / f"cfg{args.cfg_scale}" / f"eta{args.eta}"
    args.save_dir.mkdir(exist_ok=True, parents=True)
    json_path = os.path.join(args.save_dir, "summary.json")
  
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.mri_type == "fastmri":
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="fastmri")
    elif args.mri_type == "skm-tea":
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="skm-tea")
    
    if args.finetune_folder is not None:
        converted_folder = os.path.join(args.finetune_folder, "converted")
        unet_weights_path = os.path.join(converted_folder, "unet", "diffusion_pytorch_model.safetensors")
                
        # Load finetuned UNet weights
        if os.path.exists(unet_weights_path):
            try:
                state_dict = safe_load(unet_weights_path)
            except ImportError:
                state_dict = torch.load(unet_weights_path, map_location=device)
            unet.load_state_dict(state_dict, strict=False)
            print("Finetuned UNet weights loaded from", unet_weights_path)
        else:
            print("UNet weights file not found at", unet_weights_path)
        
        # ConditionEmbedding
        if args.enable_condition_emb:
            if hasattr(unet, "condition_emb"):
                print("ConditionEmbedding already attached to UNet.")
            else:
                condition_emb = ConditionEmbedding(
                    dim=unet.config.cross_attention_dim,
                    metadata_stats=args.metadata_stats,
                    cfg_prob=args.inference_cfg_prob,
                    cfg_strategy=args.cfg_strategy,
                )
                condition_weights_path = os.path.join(converted_folder, "condition", "condition_emb.pth")
                if os.path.exists(condition_weights_path):
                    condition_emb.load_state_dict(torch.load(condition_weights_path, map_location=device))
                    print("ConditionEmbedding weights loaded from", condition_weights_path)
                else:
                    print("ConditionEmbedding weights file not found at", condition_weights_path)
                unet.add_module("condition_emb", condition_emb)
                print("ConditionEmbedding attached to UNet.")
    else:
        print("No finetune folder provided; using base checkpoint weights.")

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

    args.load_dir_meta_brain = args.load_dir_meta_brain if args.load_dir_meta_brain != "null" else None
    args.load_dir_meta_knee = args.load_dir_meta_knee if args.load_dir_meta_knee != "null" else None
    full_dataset = MRIDataset(args.load_dir_meta_knee, args.load_dir_meta_brain, train=False, data_root=args.data_root)
    
    if args.start_idx is not None and args.end_idx is not None:
        # Create a subset of the dataset
        from torch.utils.data import Subset
        indices = list(range(args.start_idx, min(args.end_idx, len(full_dataset))))
        subset_dataset = Subset(full_dataset, indices)
        print(f"Processing subset from index {args.start_idx} to {args.end_idx} (total: {len(indices)} samples)")
        mri_dataset = subset_dataset
    else:
        print(f"Processing complete dataset ({len(full_dataset)} samples)")
        mri_dataset = full_dataset
    
    mri_dataloader = DataLoader(
        mri_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    num_resume = count_entries_in_json(json_path)

    def preprocess_metadata(metadata_dict):
        # Iterate over all key-value pairs in the dictionary.
        new_dict = {}
        for key, value in metadata_dict.items():
            # If the value is a torch.Tensor, convert it to a Python scalar.
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.item()
            else:
                new_dict[key] = value
        return new_dict


    psnr_list = []
    pbar = tqdm(enumerate(mri_dataloader), total=len(mri_dataloader), desc="Inference")
    for i, batch in pbar:
        if i < num_resume:
            continue

        x = batch["image"]
        mps = batch["mps"]
        filename = batch["filename"]
        slice_num = batch["slice_number"]
        prompt = batch["prompt"]
        pathology = batch["pathology"]
        anatomy = batch["anatomy"]
        if "metadata" in batch:
            metadata = batch["metadata"]
            if not isinstance(metadata, list):
                metadata = [metadata]
            metadata = [preprocess_metadata(md) for md in metadata]
        else:
            metadata = None

        B = x.shape[0]
        _, N_coil, h, w = mps.shape
        
        x = x.to(device)
        mps = mps.to(device)
        
        # Define forward operator
        if args.mask_type == "poisson2d":
            mask = poisson((image_size, image_size), accel=args.acc_factor, dtype=np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = mask.to(device)
        else:
            # Define forward operator
            mask = get_mask(
                torch.zeros([B, 1, image_size, image_size]), image_size, B, 
                type=args.mask_type,
                acc_factor=args.acc_factor, 
                center_fraction=args.center_fraction
            )
            mask = mask.to(device)

        if args.cfg_scale == 0:
            prompt = [""] * B

        A_funcs = MulticoilMRI(320, mask, mps)
        y = A_funcs.A(x)
        ATy = A_funcs.AT(y)

        ## 5 iterations
        num_samples = 5

        all_recons = []
        all_metrics = {idx: {"psnr": [], "ssim": [], "lpips": []} for idx in range(B)}
        
        ## Generate multiple samples with different seeds
        for sample_idx in range(num_samples):
            sample_seed = args.seed + sample_idx
            set_seed(sample_seed)
            
            ## Generate reconstruction
            sample_recon = pipeline.dds(
                prompt=prompt, 
                guidance_scale=args.cfg_scale, 
                num_inference_steps=args.num_timesteps, 
                eta=args.eta,
                A_funcs=A_funcs,
                y=y,
                gamma=args.gamma,
                CG_iter=args.CG_iter,
                metadata_list=metadata
            )
            sample_recon = real_to_nchw_comp(sample_recon)
            all_recons.append(sample_recon)
            
            ## Calculate metrics for individual samples (for statistics)
            for i in range(B):
                x_i = x[i:i+1, :, :, :]
                recon_i = sample_recon[i:i+1, :, :, :]
                
                x_i_np = normalize_np(np.abs(clear(x_i)))
                recon_i_np = normalize_np(np.abs(clear(recon_i)))
                
                psnr = peak_signal_noise_ratio(x_i_np, recon_i_np)
                ssim = calculate_ssim(x_i_np, recon_i_np)
                lpips_score = calculate_lpips(x_i_np, recon_i_np, device=device)
                
                all_metrics[i]["psnr"].append(psnr)
                all_metrics[i]["ssim"].append(ssim)
                all_metrics[i]["lpips"].append(lpips_score)
        
        # Stack all reconstructions
        stacked_recons = torch.stack(all_recons, dim=0)
        
        mean_recons = torch.mean(stacked_recons, dim=0)
        std_recons = torch.std(stacked_recons, dim=0)
        
        json_data_list = []
        for i in range(B):
            volume_i = filename[i]
            slice_num_i = slice_num[i].item()
            ATy_i = ATy[i:i+1, :, :, :]
            mask_i = mask[i:i+1, :, :, :]
            pathology_i = pathology[i]
            anatomy_i = anatomy[i]
            x_i = x[i:i+1, :, :, :]
            
            # Use averaged reconstruction
            recon_i = mean_recons[i:i+1, :, :, :]
            uncertainty_i = std_recons[i:i+1, :, :, :]
            
            save_path_input_i = os.path.join(args.save_dir, "input", anatomy_i, volume_i)
            save_path_recon_i = os.path.join(args.save_dir, "recon", anatomy_i, volume_i)
            save_path_label_i = os.path.join(args.save_dir, "label", anatomy_i, volume_i)
            save_path_numpy_i = os.path.join(args.save_dir, "numpy", anatomy_i, volume_i)
            save_path_uncertainty_i = os.path.join(args.save_dir, "uncertainty", anatomy_i, volume_i)
            
            os.makedirs(save_path_input_i, exist_ok=True)
            os.makedirs(save_path_recon_i, exist_ok=True)
            os.makedirs(save_path_label_i, exist_ok=True)
            os.makedirs(save_path_numpy_i, exist_ok=True)
            os.makedirs(save_path_uncertainty_i, exist_ok=True)

            np.save(os.path.join(save_path_numpy_i, f"{slice_num_i:03d}.npy"), clear(recon_i))
            plt.imsave(os.path.join(save_path_input_i, f"{slice_num_i:03d}.png"), np.abs(clear(ATy_i)), cmap='gray')
            plt.imsave(os.path.join(save_path_input_i, f"mask_{slice_num_i:03d}.png"), np.abs(clear(mask_i)), cmap='gray')
            plt.imsave(os.path.join(save_path_label_i, f"{slice_num_i:03d}.png"), np.abs(clear(x_i)), cmap='gray')
            plt.imsave(os.path.join(save_path_recon_i, f"{slice_num_i:03d}.png"), np.abs(clear(recon_i)), cmap='gray')
            
            plt.imsave(os.path.join(save_path_uncertainty_i, f"{slice_num_i:03d}.png"), np.abs(clear(uncertainty_i)), cmap='hot')
            np.save(os.path.join(save_path_uncertainty_i, f"{slice_num_i:03d}.npy"), clear(uncertainty_i))
            
            x_i_np = normalize_np(np.abs(clear(x_i)))
            recon_i_np = normalize_np(np.abs(clear(recon_i)))
            
            psnr = peak_signal_noise_ratio(x_i_np, recon_i_np)
            ssim = calculate_ssim(x_i_np, recon_i_np)
            lpips_score = calculate_lpips(x_i_np, recon_i_np, device=device)
            
            # Calculate statistics across samples
            mean_psnr = sum(all_metrics[i]["psnr"]) / num_samples
            std_psnr = np.std(all_metrics[i]["psnr"])
            mean_ssim = sum(all_metrics[i]["ssim"]) / num_samples
            std_ssim = np.std(all_metrics[i]["ssim"])
            mean_lpips = sum(all_metrics[i]["lpips"]) / num_samples
            std_lpips = np.std(all_metrics[i]["lpips"])
            
            entry = {
                "anatomy": anatomy_i,
                "volume": volume_i,
                "slice_number": slice_num_i,
                "pathology": None if pathology_i == "null" else pathology_i,
                "psnr": psnr, # PSNR of the averaged reconstruction
                "psnr_values": all_metrics[i]["psnr"], # PSNR of individual samples
                "mean_psnr": mean_psnr, # Mean PSNR across individual samples
                "std_psnr": std_psnr,
                "ssim": ssim,
                "ssim_values": all_metrics[i]["ssim"],
                "mean_ssim": mean_ssim,
                "std_ssim": std_ssim,
                "lpips": lpips_score,
                "lpips_values": all_metrics[i]["lpips"],
                "mean_lpips": mean_lpips,
                "std_lpips": std_lpips
            }
            
            json_data_list.append(entry)
            psnr_list.append(psnr)
            
            pbar.set_postfix({"PSNR": psnr, "PSNR_std": std_psnr})
        
        batch_update_json(json_path, json_data_list)
    
    if len(psnr_list) > 0:
        psnr_avg = sum(psnr_list) / len(psnr_list)
        print(f"PSNR: {psnr_avg:04f}\n")
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="MRI-Inference-Argument")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="./MRI_checkpoint", help="directory of checkpoint")
    parser.add_argument('--config_dir', type=str, default="./MRI_checkpoint")
    parser.add_argument('--num_timesteps', type=int, default=100, help="Number of timesteps for inference")
    parser.add_argument('--cfg_scale', type=float, default=1.0, help="CFG scale")
    parser.add_argument('--load_dir_meta_knee', type=str, 
                        default="../fastmri-plus/knee/meta/metadata_val.csv",)
    parser.add_argument('--load_dir_meta_brain', type=str, 
                        default="../fastmri-plus/brain/meta/metadata_val.csv",)
    parser.add_argument('--data_root', type=str, default="../fastmri")
    parser.add_argument("--enable_condition_emb", action="store_true", default=False, help="Enable condition embedding during inference.")
    parser.add_argument("--finetune_folder", type=str, default=None,
                        help="Path to the finetuning folder. This folder should contain a 'converted' subfolder with subfolders 'unet' (containing config.json and diffusion_pytorch_model.safetensors) and 'condition' (containing condition_emb.pth).")
    parser.add_argument("--metadata_stats", type=str, default=None,
                        help="Path to the metadata statistics JSON file for condition embedding.")
    parser.add_argument("--cfg_strategy", type=str, choices=["joint", "independent"],
                        default="joint", help="Classifier-free guidance strategy for conditioning.")
    parser.add_argument("--inference_cfg_prob", type=float, default=0.0,
                        help="Drop probability for classifier-free guidance conditioning during inference. Set to 0 for full condition.")
    parser.add_argument('--save_dir', type=str, 
                        default="./result/recon_complex_multi", help="The directory for saving generated images")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducible outputs")
    parser.add_argument('--eta', type=float, default=0.8, help="DDIM eta")
    parser.add_argument('--mask_type', type=str, 
                        default="uniform1d", help="masking type in the Fourier domain")
    parser.add_argument('--acc_factor', type=int, 
                        default=4, help="severity of undersampling")
    parser.add_argument('--center_fraction', type=float, 
                        default=0.08, help="severity of undersampling")
    parser.add_argument('--gamma', type=float, 
                        default=5.0, help="regularization weight inversely proportional to proximal step size")
    parser.add_argument('--CG_iter', type=int, 
                        default=5, help="Num CG iter per timestep. Default is 5")
    parser.add_argument('--batch_size', default=1, type=int, help="To keep batch size = 1 if mps data is not always same shape")
    parser.add_argument('--mri_type', type=str, choices=["fastmri", "skm-tea"], default="fastmri")
    parser.add_argument('--model_config', type=str, default="./configs/model_index.json")
    
    parser.add_argument('--start_idx', type=int, default=None, 
                        help="Starting index in the dataset (for parallel processing)")
    parser.add_argument('--end_idx', type=int, default=None, 
                        help="Ending index in the dataset (for parallel processing)")
    
    args = parser.parse_args()
    main(args)

"""
Using cond emb
python .\recon_complex_multi.py --pretrained_model_name_or_path D:\Research\data\MRI_checkpoint --load_dir_meta_brain D:\Research\data\metadata_val.csv --load_dir_meta_knee null --data_root D:\Research\data\fastmri\fastmri --enable_condition_emb --finetune_folder D:\Research\data\test_ckpt_0410 --metadata_stats D:\Research\mri-project\ContextMRI\metadata_stats.json
"""

"""
No cond emb
python .\recon_complex_multi.py --pretrained_model_name_or_path D:\Research\data\MRI_checkpoint --load_dir_meta_brain D:\Research\data\metadata_val.csv --load_dir_meta_knee null --data_root D:\Research\data\fastmri\fastmri
"""
