import argparse, json
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn.functional as F
from safetensors.torch import load_file as safe_load

from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from dataset_mri import MRIDataset
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0, AttnAddedKVProcessor2_0

import os

from pipeline_mri import MRIDiffusionPipeline
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader


from mri.utils import clear, get_mask, normalize_np, real_to_nchw_comp
from mri.mri import MulticoilMRI
from sigpy.mri import poisson
from utils import calculate_ssim, calculate_lpips, batch_update_json, set_seed, count_entries_in_json

from modules import ConditionEmbedding

from collections import defaultdict
import torch
import torch.nn as nn

class SaveAttnProcessor(nn.Module):
    def __init__(self, layer_name: str, attention_store: dict):
        super().__init__()
        self.layer_name = layer_name
        self.attention_store = attention_store

    def forward(self, attention_layer, hidden_states, encoder_states = None, attention_mask = None, **kwargs):

        is_spatial = hidden_states.dim() == 4
        if is_spatial:
            batch, channels, height, width = hidden_states.shape
            hidden_states = (hidden_states.view(batch, channels, height * width).transpose(1, 2))

        batch_size, seq_length, _ = hidden_states.shape
        attention_mask = attention_layer.prepare_attention_mask(attention_mask, seq_length, batch_size, out_dim=4)

        encoder_states = hidden_states

        normed_states = attention_layer.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        get_query = attention_layer.to_q(normed_states)
        get_key = attention_layer.to_k(encoder_states)
        get_value = attention_layer.to_v(encoder_states)

        query = attention_layer.head_to_batch_dim(get_query)
        key = attention_layer.head_to_batch_dim(get_key)
        value = attention_layer.head_to_batch_dim(get_value)

        scores = torch.matmul(query, key.transpose(-2, -1))
        
        if getattr(attention_layer, "scale_qk", False):
            scale = getattr(attention_layer, "scale_factor", query.shape[-1] ** -0.5)
            scores = scores * scale
        if attention_mask is not None:
            scores = scores + attention_mask

        attention_probs = torch.softmax(scores, dim=-1, dtype=torch.float32)

        self.attention_store.setdefault(self.layer_name, [])
        
        self.attention_store[self.layer_name].append(attention_probs.detach().cpu())

        context = torch.matmul(attention_probs.to(value.dtype), value)
        context = attention_layer.batch_to_head_dim(context)
        out = attention_layer.to_out[0](context)
        out = attention_layer.to_out[1](out)
        out = out.transpose(1, 2).view(batch, channels, height, width)

        return out

def plot_bar(layer_tensor, save_path, token_ids, tokenizer):
    attn = layer_tensor.mean(dim=(0, 1))
    try:
        # First EOS token
        cut = token_ids.index(tokenizer.eos_token_id) + 1
    except ValueError:
        # No EOS
        cut = len(token_ids)

    scores = attn.max(dim=0).values[:cut]

    tokens = tokenizer.convert_ids_to_tokens(token_ids[:cut])
    x = np.arange(len(scores))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.bar(x, scores.numpy(), width=0.8)
    plt.xticks(x, tokens, rotation=60, ha="right", fontsize=6)
    plt.title(save_path.stem)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def main(args):
    set_seed(args.seed)

    plot_dir = Path(args.save_dir) /"attention_plots"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                        subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path,
                                          subfolder="scheduler")
    
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
        text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=noise_scheduler, config_path=args.model_config)
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=args.num_timesteps)

    image_size = 512 if args.mri_type == "skm-tea" else 320
    
    # -----------------------------------------
    TARGET_LAYER = "mid_block.attentions.0.processor"
    attn_store = defaultdict(list)
    
    patched = dict(pipeline.unet.attn_processors)
    
    for name, proc in patched.items():
        mod = pipeline.unet.get_submodule(name.rsplit(".", 1)[0])
    
        if mod.only_cross_attention and isinstance(proc, AttnProcessor2_0):
            patched[name] = AttnAddedKVProcessor2_0()
    
    patched[TARGET_LAYER] = SaveAttnProcessor(TARGET_LAYER, attn_store)
    
    pipeline.unet.set_attn_processor(patched)
    # -----------------------------------------
    
    args.load_dir_meta_brain = args.load_dir_meta_brain if args.load_dir_meta_brain != "null" else None
    args.load_dir_meta_knee = args.load_dir_meta_knee if args.load_dir_meta_knee != "null" else None
    mri_dataset = MRIDataset(args.load_dir_meta_knee, args.load_dir_meta_brain, train=False, data_root=args.data_root)
    mri_dataloader = DataLoader(
        mri_dataset,          # The dataset to load
        batch_size=args.batch_size,    # Number of samples per batch
        shuffle=False,     # Whether to shuffle data at every epoch
        drop_last=False,  # Drop the last incomplete batch
    )

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
    
    psnr_list=[]

    pbar = tqdm(enumerate(mri_dataloader), total=len(mri_dataloader), desc="Inference")
    
    for i, batch in pbar:            
        attn_store.clear()
        set_seed(args.seed)
        
        x = batch["image"]
        mps = batch["mps"]
        filename = batch["filename"]
        slice_num = batch["slice_number"]
        prompt = batch["prompt"]
        anatomy = batch["anatomy"]
        pathology = batch["pathology"]

        if "metadata" in batch:
            metadata = batch["metadata"]
            if not isinstance(metadata, list):
                metadata = [metadata]
            metadata = [preprocess_metadata(md) for md in metadata]
        else:
            metadata = None

        B = x.shape[0]
        _, N_coil, H, W = mps.shape

        x = x.to(device)
        mps = mps.to(device)

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

        if args.cfg_scale==0:
            prompt=[""]*B
        
        A_funcs = MulticoilMRI(320, mask, mps) # mask: float32, mps: complex64 -> float32 broadcast to complex64 with equal number
        y = A_funcs.A(x)
        ATy = A_funcs.AT(y)
        
        recon = pipeline.dds(
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
        recon = real_to_nchw_comp(recon)

        if args.enable_attention_vis and attn_store:
            layer_tensor = torch.stack(attn_store[TARGET_LAYER])
        
            toks = tokenizer(prompt, padding="max_length", truncation=True,
                       max_length=tokenizer.model_max_length, return_tensors="pt")
            token_ids = toks.input_ids[0].tolist()
        
            png = plot_dir / f"{TARGET_LAYER.replace('.','_')}.png"
            plot_bar(layer_tensor, png, token_ids, tokenizer)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path",default="./MRI_checkpoint")
    p.add_argument("--num_timesteps",type=int,default=10)
    p.add_argument("--cfg_scale",type=float,default=1.0)
    p.add_argument("--load_dir_meta_knee",default="null")
    p.add_argument("--load_dir_meta_brain",default="null")
    p.add_argument("--data_root",default="../fastmri")
    p.add_argument("--enable_condition_emb",action="store_true")
    p.add_argument("--finetune_folder"); p.add_argument("--metadata_stats")
    p.add_argument("--cfg_strategy",choices=["joint","independent"],default="joint")
    p.add_argument("--inference_cfg_prob",type=float,default=0.0)
    p.add_argument("--enable_attention_vis",action="store_true")
    p.add_argument("--save_dir",default="./result/recon_att")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--eta",type=float,default=0.8)
    p.add_argument("--mask_type",default="uniform1d"); p.add_argument("--acc_factor",type=int,default=4)
    p.add_argument("--center_fraction",type=float,default=0.08)
    p.add_argument("--gamma",type=float,default=5.0); p.add_argument("--CG_iter",type=int,default=5)
    p.add_argument("--batch_size",type=int,default=1)
    p.add_argument("--mri_type",choices=["fastmri","skm-tea"],default="fastmri")
    p.add_argument("--model_config",default="./configs/model_index.json")
    main(p.parse_args())