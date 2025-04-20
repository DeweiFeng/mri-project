import argparse
from collections import defaultdict
import json
import math
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm


class PredictionModule(nn.Module):
    def __init__(self, emb_size, num_classes):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, emb_size))
        self.attention = nn.MultiheadAttention(emb_size, num_heads=12, batch_first=True)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, emb_size)
        # self.query: (1, 1, emb_size)

        # Expand query to match batch size
        query = self.query.expand(x.size(0), -1, -1)  # (batch_size, 1, emb_size)
        # Apply attention
        x, _ = self.attention(query, x, x)  # (batch_size, 1, emb_size)
        x = x.squeeze(1)
        # Apply layer normalization
        x = self.layer_norm(x)
        # Apply linear layer
        x = self.linear(x)  # (batch_size, num_classes)

        return x


def load_clip_text_encoder(pretrained_dir: str):
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_dir, subfolder="text_encoder"
    )
    return tokenizer, text_encoder


def encode_text(prompt, tokenizer, text_encoder, device, max_length=77):
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
        text_embeddings = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]

    return text_embeddings


def generate_prompt(
    anatomy: str,
    slice_number: int = None,
    contrast: str = None,
    sequence: str = None,
    TR: float = None,
    TE: float = None,
    TI: float = None,
    flip_angle: float = None,
    pathology: str = None,
) -> str:
    prompt = f"{anatomy}"
    if slice_number is not None:
        prompt += f", Slice {slice_number}"
    if contrast is not None:
        prompt += f", {contrast}"
    if pathology is not None:
        prompt += f", Pathology: {random.randint(1, 3)} {pathology}"
    if sequence is not None:
        prompt += f", Sequence: {sequence}"
    if TR is not None:
        prompt += f", TR: {TR:.2f}"
    if TE is not None:
        prompt += f", TE: {TE:.2f}"
    if TI is not None:
        prompt += f", TI: {TI:.2f}"
    if flip_angle is not None:
        prompt += f", Flip angle: {flip_angle:.2f}"

    return prompt


class PromptDataset(Dataset):
    def __init__(self, metadata_stats: str):
        with open(metadata_stats, "r") as f:
            self.metadata = json.load(f)

        self.attributes = [
            key for key, value in self.metadata.items() if len(value) > 1
        ]
        self.categorical_dict = {
            key: {k: i for i, k in enumerate(value)}
            for key, value in self.metadata.items()
            if len(value) > 1 and isinstance(value, list)
        }

        print(f"Attributes: {self.attributes}")
        print(f"Categorical attributes: {self.categorical_dict.keys()}")

    def __len__(self):
        return 1024

    def __getitem__(self, index):
        sampled = {}
        sampled["anatomy"] = self.metadata["anatomy"][
            random.randint(0, len(self.metadata["anatomy"]) - 1)
        ]
        sampled["slice_number"] = random.randint(
            int(self.metadata["slice_number"]["min"]),
            int(self.metadata["slice_number"]["max"]),
        )
        sampled["sequence"] = self.metadata["sequence"][
            random.randint(0, len(self.metadata["sequence"]) - 1)
        ]
        sampled["contrast"] = self.metadata["contrast"][
            random.randint(0, len(self.metadata["contrast"]) - 1)
        ]
        sampled["TR"] = random.uniform(
            float(self.metadata["TR"]["min"]), float(self.metadata["TR"]["max"])
        )
        sampled["TE"] = random.uniform(
            float(self.metadata["TE"]["min"]), float(self.metadata["TE"]["max"])
        )
        sampled["TI"] = random.uniform(
            float(self.metadata["TI"]["min"]), float(self.metadata["TI"]["max"])
        )
        sampled["flip_angle"] = random.uniform(
            float(self.metadata["flip_angle"]["min"]),
            float(self.metadata["flip_angle"]["max"]),
        )
        sampled["pathology"] = self.metadata["pathology"][
            random.randint(0, len(self.metadata["pathology"]) - 1)
        ]

        prompt = generate_prompt(**sampled)
        label = {}
        for key in self.attributes:
            if key in self.categorical_dict:
                label[key] = self.categorical_dict[key][sampled[key]]
            else:
                label[key] = (sampled[key] - float(self.metadata[key]["min"])) / (
                    float(self.metadata[key]["max"]) - float(self.metadata[key]["min"])
                )
                # normalize to [0, 1]
        return prompt, label


def collate_fn(batch):
    prompts = []
    labels = defaultdict(list)
    for prompt, label in batch:
        prompts.append(prompt)
        for key, value in label.items():
            labels[key].append(value)

    for key, value in labels.items():
        labels[key] = torch.tensor(value)

    return prompts, labels


def fix_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text Encoder")
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        required=True,
        help="Path to pretrained model directory",
    )
    parser.add_argument(
        "--metadata_stats",
        type=str,
        required=True,
        help="Path to metadata_stats.json file",
    )

    # Training
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_updates",
        type=int,
        default=1000,
        help="Number of updates for training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fix random seed for reproducibility
    fix_seed(args.seed)

    # Load the tokenizer and text encoder
    tokenizer, text_encoder = load_clip_text_encoder(args.pretrained_dir)
    text_encoder.to(device)
    text_encoder.eval()
    emb_size = text_encoder.config.projection_dim

    # Create the dataset and dataloader
    train_set = PromptDataset(args.metadata_stats)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    test_set = PromptDataset(args.metadata_stats)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Numerical / categorical attributes
    numerical_attributes = [
        key for key in train_set.attributes if key not in train_set.categorical_dict
    ]
    categorical_attributes = [
        key for key in train_set.attributes if key in train_set.categorical_dict
    ]

    # Initialize linear probes
    linear_probes = nn.ModuleDict()
    for key in numerical_attributes:
        linear_probes[key] = PredictionModule(emb_size, 1)
    for key in categorical_attributes:
        linear_probes[key] = PredictionModule(
            emb_size, len(train_set.categorical_dict[key])
        )
    linear_probes.to(device)
    linear_probes.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        linear_probes.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Initialize loss function
    numerical_loss_fn = nn.MSELoss()
    categorical_loss_fn = nn.CrossEntropyLoss()

    # Training loop
    print("Starting training...")
    num_epochs = int(math.ceil(args.num_updates / len(train_loader)))
    pbar = tqdm(total=args.num_updates, desc="Training")
    for epoch in range(num_epochs):
        for prompts, labels in train_loader:
            optimizer.zero_grad()

            # Encode text
            with torch.no_grad():
                text_embeddings = encode_text(
                    prompts,
                    tokenizer,
                    text_encoder,
                    device,
                )

            # Move labels to device
            for key, value in labels.items():
                labels[key] = value.to(device)

            # Forward pass
            outputs = {}
            for key, probe in linear_probes.items():
                outputs[key] = probe(text_embeddings)
            loss = 0
            for key in numerical_attributes:
                loss += numerical_loss_fn(outputs[key].squeeze(), labels[key].float())
            for key in categorical_attributes:
                loss += categorical_loss_fn(outputs[key], labels[key].long().squeeze())

            # Backward pass
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
    pbar.close()
    print("Training completed.")

    # Evaluation loop
    print("Evaluating...")
    fix_seed(args.seed)
    linear_probes.eval()
    numerical_mse = defaultdict(list)
    categorical_acc = defaultdict(list)
    with torch.no_grad():
        for prompts, labels in tqdm(test_loader):
            # Encode text
            text_embeddings = encode_text(
                prompts,
                tokenizer,
                text_encoder,
                device,
            )

            # Move labels to device
            for key, value in labels.items():
                labels[key] = value.to(device)

            # Forward pass
            outputs = {}
            for key, probe in linear_probes.items():
                outputs[key] = probe(text_embeddings)
            for key in numerical_attributes:
                numerical_mse[key].append(
                    numerical_loss_fn(
                        outputs[key].squeeze(), labels[key].float()
                    ).item()
                )
            for key in categorical_attributes:
                categorical_acc[key].append(
                    (outputs[key].argmax(dim=-1) == labels[key].long())
                    .float()
                    .mean()
                    .item()
                )
    print("Evaluation completed.")

    # Calculate average metrics
    numerical_mse_avg = {
        key: sum(values) / len(values) for key, values in numerical_mse.items()
    }
    categorical_acc_avg = {
        key: sum(values) / len(values) for key, values in categorical_acc.items()
    }
    print("Numerical MSE:")
    for key, value in numerical_mse_avg.items():
        print(f"  {key}: {value:.4f}")
    print("Categorical Accuracy:")
    for key, value in categorical_acc_avg.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()

"""
python3 test_hengjui/text_emb_recover.py \
    --pretrained_dir /data/sls/scratch/hengjui/6S982/pretrained/MRI_checkpoint \
    --metadata_stats /data/sls/r/u/hengjui/home/scratch/6S982/mri-project/ContextMRI/test_hengjui/metadata_stats.json \
    --lr 1e-4 \
    --batch_size 256 \
    --num_updates 1000

==== Results ====
Using device: cuda
Attributes: ['anatomy', 'slice_number', 'contrast', 'sequence', 'TR', 'TE', 'TI', 'flip_angle', 'pathology']
Categorical attributes: dict_keys(['anatomy', 'contrast', 'sequence', 'pathology'])
Attributes: ['anatomy', 'slice_number', 'contrast', 'sequence', 'TR', 'TE', 'TI', 'flip_angle', 'pathology']
Categorical attributes: dict_keys(['anatomy', 'contrast', 'sequence', 'pathology'])
Starting training...
Training: 100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [08:47<00:00,  1.89it/s, loss=0.00983]
Training completed.
Evaluating...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.14it/s]
Evaluation completed.
Numerical MSE:
  slice_number: 0.0002
  TR: 0.0041
  TE: 0.0020
  TI: 0.0027
  flip_angle: 0.0010
Categorical Accuracy:
  anatomy: 1.0000
  contrast: 1.0000
  sequence: 1.0000
  pathology: 1.0000
"""
