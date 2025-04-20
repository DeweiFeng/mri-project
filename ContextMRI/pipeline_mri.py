import torch
from torch import nn
from typing import Union, List, Optional, Dict, Callable, Any
from tqdm.auto import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
from mri.utils import real_to_nchw_comp, nchw_comp_to_real, clear, CG
import os
from modules import ConditionEmbedding


class MRIDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        text_encoder: nn.Module,
        tokenizer,
        unet: nn.Module,
        scheduler,
        image_processor=None,
        config_path=None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        # set terminal SNR to max
        self.final_alpha_cumprod = torch.Tensor([1.0]).to(self.device)
        if image_processor is not None:
            self.image_processor = image_processor

        self.register_modules(
            unet=self.unet,
            scheduler=self.scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )

        self.model_cpu_offload_seq = "text_encoder->unet"
        if config_path:
            self._load_and_set_config(config_path)

    def _load_and_set_config(self, config_path):
        # Load and parse the configuration from the JSON file
        with open(config_path, "r") as f:
            config = json.load(f)
        self.register_to_config(config=config)

    @property
    def components(self):
        # Return components as a dictionary
        return {
            "unet": self.unet,
            "scheduler": self.scheduler,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
        }

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        max_length=None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("Prompt must be either a string or a list of strings.")

        if max_length is not None:
            tokenizer_max_length = max_length
        else:
            tokenizer_max_length = self.tokenizer.model_max_length

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids, attention_mask=attention_mask
        )[0]

        prompt_embeds = prompt_embeds.to(device=device)

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:

            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif (
                isinstance(negative_prompt, list) and len(negative_prompt) == batch_size
            ):
                uncond_tokens = negative_prompt
            else:
                raise ValueError(
                    "`negative_prompt` should be the same length as `prompt`."
                )

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids, attention_mask=attention_mask
            )[0]
            negative_prompt_embeds = negative_prompt_embeds.to(device=device)

            # duplicate unconditional embeddings for each generation per prompt
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            return torch.cat([negative_prompt_embeds, prompt_embeds])

        else:
            return prompt_embeds

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height),
            int(width),
        )
        if latents is None:
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def denoise_latents(
        self,
        latents,
        timesteps,
        prompt_embeds,
        extra_step_kwargs,
        num_inference_steps,
        guidance_scale,
        negative_prompt_embeds=None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,  # New parameter
    ):
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            latent_model_input = (
                torch.cat([latents] * 2) if guidance_scale > 1 else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            cur_prompt_embeds = prompt_embeds
            # conditioning embeddings
            if metadata_list is not None and hasattr(self.unet, "condition_emb"):
                cond_emb = self.unet.condition_emb(
                    metadata_list, 
                    dtype=latent_model_input.dtype, 
                    device=latent_model_input.device
                )
                cur_prompt_embeds = prompt_embeds + cond_emb.unsqueeze(1)

            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=cur_prompt_embeds
            )[0]

            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
                0
            ]
        return latents

    @torch.no_grad()
    def sample(self, prompt, guidance_scale, num_inference_steps, eta=0.0,metadata_list: Optional[List[Dict[str, Any]]] = None):
        skip = 1000 // num_inference_steps

        # 1. Encode input prompt
        prompt_embeds = self.encode_prompt(
            prompt, self.device, 1, guidance_scale > 1.0, None
        )

        # 2. initialize_latent x_T
        xt = self.prepare_latents(
            len(prompt),
            self.unet.config.in_channels,
            320,
            320,
            prompt_embeds.dtype,
            self.device,
            None,
            None,
        )

        self.final_alpha_cumprod = torch.Tensor([1.0]).to(self.device)

        # 3. DDIM sampling loop
        pbar = tqdm(self.scheduler.timesteps, desc="Sampling")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - skip)

            xt_in = torch.cat([xt] * 2) if guidance_scale > 1 else xt
            
            # KEY CHANGE
            cur_prompt_embeds = prompt_embeds
            if metadata_list is not None and hasattr(self.unet, "condition_emb"):
                cond_emb = self.unet.condition_emb(
                    metadata_list,
                    dtype=xt_in.dtype,
                    device=xt_in.device
                )
                cur_prompt_embeds = prompt_embeds + cond_emb.unsqueeze(1)

            noise_pred = self.unet(
                xt_in, t, encoder_hidden_states=cur_prompt_embeds
            )[0]
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # tweedie
            x0t = (xt - (1 - at).sqrt() * noise_pred) / at.sqrt()
            x0t = torch.clamp(x0t, -4, 4)

            # add noise
            c1 = ((1 - at / at_prev) * (1 - at_prev) / (1 - at)).sqrt() * eta
            c2 = ((1 - at_prev) - c1**2).sqrt()
            if step != len(self.scheduler.timesteps) - 1:
                xt = at_prev.sqrt() * x0t + c1 * torch.randn_like(x0t) + c2 * noise_pred
            else:
                xt = x0t

        return xt

    @torch.no_grad()
    def ddnm(
        self,
        prompt,
        guidance_scale,
        num_inference_steps,
        eta=0.0,
        A_funcs=None,
        y=None,
        use_cfgpp=False,
        save_dir=None,
    ):
        skip = 1000 // num_inference_steps

        # 1. Encode input prompt
        if use_cfgpp or guidance_scale > 1.0:
            cfg_guidance = True
        else:
            cfg_guidance = False
        prompt_embeds = self.encode_prompt(prompt, self.device, 1, cfg_guidance, None)
        # 2. initialize_latent x_T
        xt = self.prepare_latents(
            len(prompt),
            self.unet.config.in_channels,
            320,
            320,
            prompt_embeds.dtype,
            self.device,
            None,
            None,
        )

        self.final_alpha_cumprod = torch.Tensor([1.0]).to(self.device)

        # 3. DDIM sampling loop
        pbar = tqdm(self.scheduler.timesteps, desc="Sampling")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - skip)

            xt_in = torch.cat([xt] * 2) if cfg_guidance else xt
            noise_pred = noise_pred_uncond = self.unet(
                xt_in, t, encoder_hidden_states=prompt_embeds
            )[0]

            if cfg_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # 1. tweedie (with cfg)
            x0t = (xt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # 1-1 (optional). real <-> complex
            if x0t.shape[1] == 2:
                x0t = real_to_nchw_comp(x0t)

            # 2. ddnm
            x0t = x0t - A_funcs.AT(A_funcs.A(x0t) - y)

            if not save_dir is None:
                plt.imsave(
                    str(save_dir / f"{t:03d}.png"), np.abs(clear(x0t)), cmap="gray"
                )

            # 2-1 (optional). real <-> complex
            if x0t.dtype == torch.complex64:
                x0t = nchw_comp_to_real(x0t)

            # 3. add noise
            c1 = ((1 - at / at_prev) * (1 - at_prev) / (1 - at)).sqrt() * eta
            c2 = ((1 - at_prev) - c1**2).sqrt()
            xt = at_prev.sqrt() * x0t + c1 * torch.randn_like(x0t) + c2 * noise_pred

        return x0t

    @torch.no_grad()
    def dds(
        self,
        prompt,
        guidance_scale,
        num_inference_steps,
        sample_size=320,
        eta=0.0,
        A_funcs=None,
        y=None,
        gamma=5.0,
        CG_iter=5,
        save_dir=None,
        metadata_list: Optional[List[Dict[str, Any]]] = None  # <-- New parameter
    ):
        skip = 1000 // num_inference_steps

        # 1. Encode input prompt
        if guidance_scale > 1.0:
            cfg_guidance = True
        else:
            cfg_guidance = False
        prompt_embeds = self.encode_prompt(prompt, self.device, 1, cfg_guidance, None)
        # 2. initialize_latent x_T
        xt = self.prepare_latents(
            len(prompt),
            self.unet.config.in_channels,
            sample_size,
            sample_size,
            prompt_embeds.dtype,
            self.device,
            None,
            None,
        )

        self.final_alpha_cumprod = torch.Tensor([1.0]).to(self.device)

        # 3. DDIM sampling loop
        pbar = tqdm(self.scheduler.timesteps, desc="Sampling")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - skip)

            xt_in = torch.cat([xt] * 2) if cfg_guidance else xt
            
            # NEW: Incorporate conditioning metadata if provided.
            cur_prompt_embeds = prompt_embeds
            if metadata_list is not None and hasattr(self.unet, "condition_emb"):
                cond_emb = self.unet.condition_emb(
                    metadata_list,
                    dtype=xt_in.dtype,
                    device=xt_in.device
                )
                cur_prompt_embeds = prompt_embeds + cond_emb.unsqueeze(1)
                print("prompt_embeds shape: ", prompt_embeds.shape)
                print("cond_emb shape: ", cond_emb.shape)
            
            noise_pred = noise_pred_uncond = self.unet(
                xt_in, t, encoder_hidden_states=cur_prompt_embeds
            )[0]
  
            if cfg_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # 1. tweedie (with cfg)
            x0t = (xt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # 1-1 (optional). real <-> complex
            if x0t.shape[1] == 2:
                x0t = real_to_nchw_comp(x0t)

            # 2. CG
            bcg = x0t + gamma * A_funcs.AT(y)

            def Acg_fn(x):
                return x + gamma * A_funcs.AT(A_funcs.A(x))

            x0t = CG(Acg_fn, bcg, x0t, n_inner=CG_iter)

            if not save_dir is None:
                plt.imsave(
                    os.path.join(save_dir, f"{t:03d}.png"),
                    np.abs(clear(x0t)),
                    cmap="gray",
                )

            # 2-1 (optional). real <-> complex
            if x0t.dtype == torch.complex64:
                x0t = nchw_comp_to_real(x0t)

            # 3. add noise
            c1 = ((1 - at / at_prev) * (1 - at_prev) / (1 - at)).sqrt() * eta
            c2 = ((1 - at_prev) - c1**2).sqrt()
            xt = at_prev.sqrt() * x0t + c1 * torch.randn_like(x0t) + c2 * noise_pred

        return x0t

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        metadata_list: Optional[List[Dict[str, Any]]] = None,  # <-- New parameter!
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        height = height or self.unet.sample_size
        width = width or self.unet.sample_size

        # 1. Encode input prompt
        prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt, guidance_scale > 1.0, negative_prompt
        )

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)

        # 3. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            len(prompt),
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 4. Prepare extra step kwargs
        extra_step_kwargs = {}

        # 5. Denoising loop
        latents = self.denoise_latents(
            latents,
            timesteps,
            prompt_embeds,
            extra_step_kwargs,
            num_inference_steps,
            guidance_scale,
            metadata_list=metadata_list  # <-- New!
        )

        latents = latents.squeeze().detach().cpu().numpy()

        # 6. Decode latents to images
        return latents if not return_dict else {"images": latents}
