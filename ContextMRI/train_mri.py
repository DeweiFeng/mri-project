import torch
import torch.nn as nn
from torch.nn import functional as F

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import json
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from dataset_mri import MRIDataset
from modules import ConditionEmbedding

import diffusers
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
)
from diffusers.utils.torch_utils import is_compiled_module
import wandb

logging.basicConfig(level=logging.INFO)  # Set the level to INFO
logger = get_logger(__name__)

from transformers import T5Tokenizer, T5EncoderModel


def collate_fn(examples):
    pixel_values = [example["image"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    metadata_list = [example["metadata"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "metadata_list": metadata_list,
    }
    return batch


def encode_text(prompt, tokenizer, model, device, max_length=None):
    if max_length is not None:
        tokenizer_max_length = max_length
    else:
        tokenizer_max_length = tokenizer.model_max_length

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer_max_length,
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = None

    with torch.no_grad():
        text_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    return text_embeddings


# NEW encode T5 here:
def encode_text_t5(prompts, tokenizer, model, device, max_length=77):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        hidden = model.encoder(**inputs).last_hidden_state  # [B, L, 768]
    return hidden


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=32):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        dtype = original_linear.weight.dtype  # Get dtype from original layer

        self.lora_A = nn.Linear(in_dim, r, bias=False).to(dtype)
        self.lora_B = nn.Linear(r, out_dim, bias=False).to(dtype)
        self.scaling = self.alpha / self.r

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.scaling * self.lora_B(self.lora_A(x))


def apply_lora_to_unet(unet, target_names, r=4, alpha=32):
    for name, module in unet.named_modules():
        if any(t in name for t in target_names) and isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0]
            attr_name = name.split(".")[-1]

            # Get parent module
            parent = dict(unet.named_modules())[parent_name]
            original = getattr(parent, attr_name)

            # Replace with LoRA-wrapped Linear
            setattr(parent, attr_name, LoRALinear(original, r=r, alpha=alpha))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_list(params):
    return sum(p.numel() for p in params if p.requires_grad)


def main(args):
    # Accelerate config
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # warning
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    if args.pretrained_model_type is None:
        logger.info("Training from scratch")
        unet_config_path = "./configs/unet/config_mri.json"
        with open(unet_config_path, "r") as f:
            unet_config = json.load(f)

        unet = UNet2DConditionModel.from_config(unet_config)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder=args.pretrained_model_type
        )

    if args.use_lora:
        apply_lora_to_unet(
            unet,
            target_names=["to_q", "to_k", "to_v", "to_out.0"],
            r=args.lora_rank,
            alpha=32,
        )

    # NEW t5 here:
    text_proj = None
    if args.text_encoder_type == "clip":
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder.requires_grad_(False)
    else:  #'T5'
        tokenizer = T5Tokenizer.from_pretrained(args.t5_model_name_or_path)
        text_encoder = T5EncoderModel.from_pretrained(args.t5_model_name_or_path)
        # optionally freeze T5 entirely if not fine-tuning
        if not args.finetune_t5:
            for p in text_encoder.parameters():
                p.requires_grad = False

    condition_emb = None
    if args.enable_condition_emb:
        condition_emb = ConditionEmbedding(
            dim=unet.config.cross_attention_dim,
            metadata_stats=args.metadata_stats,
            cfg_prob=args.cfg_prob,
            cfg_strategy=args.cfg_strategy,
        )
        unet.add_module("condition_emb", condition_emb)
        logger.info(
            f"Condition embedding parameters: {count_parameters(condition_emb)}"
        )

    if args.use_lora:
        unet.requires_grad_(False)
        for name, module in unet.named_modules():
            if isinstance(module, (LoRALinear, ConditionEmbedding)):
                for p in module.parameters():
                    p.requires_grad = True

    weight_dtype = torch.float32
    if args.use_lora:
        unet.to(accelerator.device, dtype=weight_dtype)
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logger.info(f"UNet trainable parameters: {count_parameters(unet)}")
    logger.info(f"Text encoder trainable parameters: {count_parameters(text_encoder)}")

    unet.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    ema_unet = AveragedModel(
        unet, multi_avg_fn=get_ema_multi_avg_fn(0.999)
    )  # For 0.999 decay

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_to_save = model.state_dict()  # Save full transformer weights
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                # Pop weights to avoid duplicate saving
                weights.pop()

            if args.use_lora:
                # Save UNet (base + lora weights)
                unet_state = unet.state_dict()
                torch.save(unet_state, os.path.join(output_dir, "unet_with_lora.pth"))

                # Save EMA if needed
                ema_weights = (
                    ema_unet.module.state_dict()
                    if hasattr(ema_unet, "module")
                    else ema_unet.state_dict()
                )
                torch.save(
                    ema_weights, os.path.join(output_dir, "unet_ema_0.999_weights.pth")
                )
            else:
                # Save the full weights of each model
                torch.save(unet_to_save, os.path.join(output_dir, "unet_weights.pth"))

                # Save EMA weights
                ema_weights = (
                    ema_unet.module.state_dict()
                    if hasattr(ema_unet, "module")
                    else ema_unet.state_dict()
                )
                torch.save(
                    ema_weights, os.path.join(output_dir, "unet_ema_0.999_weights.pth")
                )

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected load model: {model.__class__}")

        # Load full weights from the saved state dicts
        unet_state_dict = torch.load(os.path.join(input_dir, "unet_weights.pth"))

        # Set the state dict into the models
        unet_.load_state_dict(unet_state_dict)

        # If mixed precision is being used, ensure the model weights are in float32
        if args.mixed_precision == "fp16":
            models = [unet_]
            # Upcast trainable parameters to fp32
            cast_training_params(models)
        del unet_state_dict
        torch.cuda.empty_cache()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        cast_training_params(models, dtype=torch.float32)

    if args.use_lora:
        for name, module in unet.named_modules():
            if isinstance(module, LoRALinear):
                module.to(accelerator.device, dtype=weight_dtype)
        unet_parameters = [p for n, p in unet.named_parameters() if p.requires_grad]
        if text_proj is not None:
            unet_parameters += list(text_proj.parameters())
        if args.text_encoder_type == "t5" and args.finetune_t5:
            unet_parameters += list(
                filter(lambda p: p.requires_grad, text_encoder.parameters())
            )
        logger.info(
            f"UNet trainable parameters (with LoRA): {count_parameters_list(unet_parameters)}"
        )
        params_to_optimize = [{"params": unet_parameters, "lr": args.learning_rate}]
    else:
        unet_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
        # If using T5, ensure the projector (and optionally T5) get optimized
        if text_proj is not None:
            unet_parameters += list(text_proj.parameters())
        if args.text_encoder_type == "t5" and args.finetune_t5:
            unet_parameters += list(
                filter(lambda p: p.requires_grad, text_encoder.parameters())
            )
        logger.info(
            f"UNet trainable parameters (no LoRA): {count_parameters_list(unet_parameters)}"
        )
        unet_parameters_with_lr = {"params": unet_parameters, "lr": args.learning_rate}
        params_to_optimize = [unet_parameters_with_lr]

    if not args.optimizer.lower() == "adamw":
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        raise ValueError(f"Not supported {args.optimizer.lower()} optimizer")

    train_dataset = MRIDataset(
        metadata_file_knee=args.mri_metadata_dir_knee,
        metadata_file_brain=args.mri_metadata_dir_brain,
        train=True,
        data_root=args.data_root,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=0,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "DeepFloyD-IF-M-MRI"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if accelerator.is_main_process:
        wandb.login()
        wandb.init(project="ContextMRI", name=args.output_dir.split("/")[-1])

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[2]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[2])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [unet]
            with accelerator.accumulate(models_to_accumulate):
                pixel_values = batch["pixel_values"]
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                # NEW T5 here
                text_embeddings = None
                if not args.disable_embedding:
                    if args.text_encoder_type == "clip":
                        text_embeddings = encode_text(
                            prompts,
                            tokenizer,
                            text_encoder,
                            device=accelerator.device,
                            max_length=77,
                        )
                    else:  # t5
                        text_embeddings = encode_text_t5(
                            prompts,
                            tokenizer,
                            text_encoder,
                            device=accelerator.device,
                            max_length=77,
                        )
                    text_embeddings = text_embeddings.contiguous()
                model_input = pixel_values.to(dtype=weight_dtype)

                # conditioning embeddings
                if args.enable_condition_emb:
                    metadata_list = batch["metadata_list"]
                    if isinstance(unet, torch.nn.parallel.DistributedDataParallel):
                        condition_emb_module = unet.module.condition_emb
                    else:
                        condition_emb_module = unet.condition_emb
                    cond_embeddings = condition_emb_module(
                        metadata_list, dtype=weight_dtype, device=accelerator.device
                    )
                    if text_embeddings is None:
                        text_embeddings = cond_embeddings
                    else:
                        text_embeddings = torch.cat(
                            [text_embeddings, cond_embeddings], dim=1
                        )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                )
                noisy_model_input = noise_scheduler.add_noise(
                    model_input, noise, timesteps
                )

                model_pred = unet(
                    noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeddings,
                ).sample

                loss = F.mse_loss(
                    model_pred.float(), noise.float(), reduction="mean"
                )  # epsilon matching

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update EMA
                if accelerator.sync_gradients:
                    ema_unet.update_parameters(unet)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-step-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                wandb.log(
                    {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                )

            if global_step >= args.max_train_steps:
                break


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="MRI foundation training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_type",
        type=str,
        default=None,
        required=False,
        choices=["fastmri", "skm-tea"],
        help="The type of pretrained model to load for continued training.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../fastmri",
        help="The root directory of the training data.",
    )
    parser.add_argument(
        "--mri_metadata_dir_knee",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--mri_metadata_dir_brain",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mri-result",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp32"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    # Proposed condition embedding arguments
    parser.add_argument(
        "--enable_condition_emb",
        action="store_true",
        default=False,
        help=(
            "Whether to use condition embedding. If True, the model will use condition embedding."
        ),
    )
    parser.add_argument(
        "--metadata_stats",
        type=str,
        default=None,
        help=("Path to the metadata statistics file for condition embedding."),
    )
    parser.add_argument(
        "--cfg_prob",
        type=float,
        default=0.5,
        help=("The drop probability of classifier-free guidance."),
    )
    parser.add_argument(
        "--cfg_strategy",
        type=str,
        default="joint",
        choices=["joint", "independent"],
        help=("The strategy for classifier-free guidance."),
    )
    parser.add_argument(
        "--disable_embedding",
        action="store_true",
        default=False,
        help=(
            "Whether to disable CLIP embedding. If True, the model will not use CLIP embedding."
        ),
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Use LoRA to adapt the attention layers for fine-tuning.",
    )

    # Proposed T5 text embedding
    parser.add_argument(
        "--text_encoder_type",
        type=str,
        choices=["clip", "t5"],
        default="clip",
        help="Which text encoder to use for conditioning: 'clip' or 't5'.",
    )
    # only needed if T5 is chosen
    parser.add_argument(
        "--t5_model_name_or_path",
        type=str,
        default="t5-base",
        help="HuggingFace ID or path for T5 encoder.",
    )
    parser.add_argument(
        "--finetune_t5",
        action="store_true",
        help="If using T5, fine-tune its weights (otherwise keep T5 frozen).",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # if args.mri_metadata_dir_brain is None or args.mri_metadata_dir_knee is None:
    #     raise ValueError("Specify both `--metadata-brain` or `--metadata-knee`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
