# Metadata Matters: Fine-Grained Embeddings for Enhanced MRI Reconstruction

This repository contains the implementation for our paper "Metadata Matters: Fine-Grained Embeddings for Enhanced MRI Reconstruction". We propose methods to improve MRI reconstruction quality by leveraging fine-grained metadata conditioning in diffusion models.

# Overview

Medical images are typically accompanied by rich metadata describing acquisition parameters and patient context. While previous work (ContextMRI) introduced text-based conditioning using CLIP embeddings, our approach implements more precise handling of MRI metadata by introducing type-specific embeddings for numerical and categorical values.

Our key innovation is replacing text embeddings with a structured conditioning framework where each metadata attribute (TR, TE, flip angle, etc.) is encoded according to its data type: categorical variables use embedding tables, while numerical values use sinusoidal position embeddings.

# Key Features:

- Fine-grained Metadata Conditioning: Type-aware embeddings for each metadata field (TR, TE, sequence, contrast, etc.)
- T5 Encoder Option: Alternative to CLIP for better encoding of numerical parameters
- Out-of-Distribution Training: Uses cardiac MRI data to enhance the model's reliance on metadata
- Multiple Sampling Strategy: Ensemble approach for more reliable reconstruction
- LoRA Support: Parameter-efficient fine-tuning of pre-trained models

# Data Preparation

The repository supports processing data from standard formats (such as FastMRI) into the required format:

```bash
# Process FastMRI data
python pre_processing_final.py --input_dir /path/to/fastmri/h5_files --output_root ./fastmri --train

# For cardiac MRI data
python pre_process_cmrx.py --input_dir /path/to/cmrxrecon_data --output_dir ./fastmri
```

