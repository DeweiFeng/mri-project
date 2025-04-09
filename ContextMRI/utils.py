import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import torch
import json
import lpips
from skimage.metrics import structural_similarity as compare_ssim
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def category_dict(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    category = []
    for cat in data["categories"]:
        category.append(cat["name"])

    return category


# extract the row from the dataframe that matches the filename and slice number
def extract_metadata(df, filename, slice_number):
    # Filter the dataframe for the given filename and slice number
    metadata = df[(df["filename"] == filename) & (df["slice"] == slice_number)]

    # If no matching records found
    if metadata.empty:
        return f"No data found for filename: {filename}, slice: {slice_number}"

    # Return the first matching row as a dictionary or a pandas Series
    return metadata.iloc[0]


def row_to_text_string(row, p=0.5):
    # Extract fields
    anatomy = row["anatomy"]
    slice_number = row["slice"]
    contrast = row["contrast"]
    pathology = row["pathology"]

    # Core string that always appears
    text = f"{anatomy}, Slice {slice_number}, {contrast}"

    if not pd.isna(pathology):
        # Split pathologies and count occurrences
        pathologies = pathology.split(", ")
        pathology_counts = {}
        for path in pathologies:
            pathology_counts[path] = pathology_counts.get(path, 0) + 1

        # Format as "N count pathology"
        numbered_pathologies = [
            f"{count} {path}" for path, count in pathology_counts.items()
        ]
        text += f", Pathology: {', '.join(numbered_pathologies)}"

    # 50% chance to include the sequence and imaging parameters
    if random.random() < p:
        sequence = row["sequence"]
        TR = row["TR"]
        TE = row["TE"]
        TI = row["TI"]
        flip_angle = row["flip_angle"]

        text += (
            f", Sequence: {sequence}, TR: {TR}, TE: {TE}, TI: {TI}, "
            f"Flip angle: {flip_angle}"
        )

    return text


def row_to_dict(row):
    # Extract fields
    anatomy = row["anatomy"]
    slice_number = row["slice"]
    contrast = row["contrast"]
    pathology = row["pathology"]
    sequence = row["sequence"]
    TR = row["TR"]
    TE = row["TE"]
    TI = row["TI"]
    flip_angle = row["flip_angle"]

    # Core string that always appears
    data = {
        "anatomy": anatomy,
        "slice_number": float(slice_number),
        "contrast": contrast,
        "sequence": sequence,
        "TR": float(TR),
        "TE": float(TE),
        "TI": float(TI),
        "flip_angle": float(flip_angle),
        "pathology": [],
    }

    if not pd.isna(pathology):
        # Split pathologies and count occurrences
        pathologies = pathology.split(", ")
        pathology_counts = {}
        for path in pathologies:
            pathology_counts[path] = pathology_counts.get(path, 0) + 1

        # Add pathologies to the data dictionary
        data["pathology"] = [(path, count) for path, count in pathology_counts.items()]

    return data


# Convert CSV row into text string with appropriate format
def row_to_text_string_skm_tea(row, p=0.5):
    # Extract fields
    slice_number = row["slice"]
    age = row["PatientAge"]
    sex = row["PatientSex"]
    pathology = row["pathology"]

    text = f"Qdess, Knee, Slice {slice_number}, Age: {age}, Sex: {sex}"

    if not pd.isna(pathology):
        # Split pathologies and count occurrences
        pathologies = pathology.split(", ")
        pathology_counts = {}
        for path in pathologies:
            pathology_counts[path] = pathology_counts.get(path, 0) + 1

        # Format as "N count pathology"
        numbered_pathologies = [
            f"{count} {path}" for path, count in pathology_counts.items()
        ]
        text += f", Pathology: {', '.join(numbered_pathologies)}"

    if random.random() < p:
        tr = row["RepetitionTime"]
        te = row["EchoTime1"]
        fa = row["FlipAngle"]
        text += f", TR: {tr}, TE: {te}, Flip Angle: {fa}"

    return text


def calculate_ssim(image1, image2):

    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for SSIM calculation.")
    ssim, _ = compare_ssim(image1, image2, full=True, data_range=2.0)
    return ssim


def calculate_lpips(image1_np, image2_np, device):

    model = lpips.LPIPS(net="vgg").to(device)  # Use VGG backbone

    # Ensure the input is float32
    image1_np = image1_np.astype(np.float32)
    image2_np = image2_np.astype(np.float32)

    # Convert NumPy arrays to PyTorch tensors and add batch and channel dimensions
    image1_tensor = (
        torch.tensor(image1_np).unsqueeze(0).unsqueeze(0).to(device)
    )  # Shape: (1, 1, 320, 320)
    image2_tensor = (
        torch.tensor(image2_np).unsqueeze(0).unsqueeze(0).to(device)
    )  # Shape: (1, 1, 320, 320)

    # Normalize to [-1, 1] as required by LPIPS
    image1_tensor = (image1_tensor * 2) - 1
    image2_tensor = (image2_tensor * 2) - 1

    # Calculate LPIPS score
    lpips_score = model(image1_tensor, image2_tensor)

    return lpips_score.item()


def batch_update_json(filepath, new_data_list):

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, "r") as f:
            data = json.load(f)  # Load JSON
    else:
        data = []

    data.extend(new_data_list)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


# Convert user_input to text_string with appropriate format
def user_input_to_text_string(
    anatomy,
    slice_number,
    contrast,
    pathology=None,
    sequence=None,
    TR=None,
    TE=None,
    TI=None,
    flip_angle=None,
):

    text = f"{anatomy}, Slice {slice_number}, {contrast}"

    if pathology is not None:
        text += f", Pathology: {pathology}"

    if (
        sequence is not None
        and TR is not None
        and TE is not None
        and TI is not None
        and flip_angle is not None
    ):
        text += (
            f", {sequence}, TR: {TR}, TE: {TE}, TI: {TI}, " f"Flip angle: {flip_angle}"
        )

    return text


def save_image(image, path):
    plt.imsave(path, image, cmap="gray")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_entries_in_json(file_path):

    if not os.path.exists(file_path):
        return 0

    with open(file_path, "r") as file:
        data = json.load(file)

    # Count entries based on the data structure
    if isinstance(data, dict):
        return len(data)
    elif isinstance(data, list):
        return len(data)
    else:
        raise ValueError(
            "Unsupported JSON structure. It must be a list or a dictionary."
        )
