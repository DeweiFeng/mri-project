import argparse
import json

import pandas as pd
from tqdm import tqdm

from dataset_mri import MRIDataset


def main(
    save_path,
    metadata_file_knee=None,
    metadata_file_brain=None,
    data_root="../fastmri",
):
    dataset = MRIDataset(
        metadata_file_knee=metadata_file_knee,
        metadata_file_brain=metadata_file_brain,
        train=True,
        data_root=data_root,
    )
    metadata = dataset.metadata
    metadata_stats = {
        "anatomy": metadata["anatomy"].unique().tolist(),
        "slice_number": {
            "min": float(metadata["slice"].min()),
            "max": float(metadata["slice"].max()),
        },
        "contrast": metadata["contrast"].unique().tolist(),
        "sequence": metadata["sequence"].unique().tolist(),
        "TR": {
            "min": float(metadata["TR"].min()),
            "max": float(metadata["TR"].max()),
        },
        "TE": {
            "min": float(metadata["TE"].min()),
            "max": float(metadata["TE"].max()),
        },
        "TI": {
            "min": float(metadata["TI"].min()),
            "max": float(metadata["TI"].max()),
        },
        "flip_angle": {
            "min": float(metadata["flip_angle"].min()),
            "max": float(metadata["flip_angle"].max()),
        },
    }

    rows = dataset.valid_rows
    pathology_list = set()
    for row in tqdm(rows):
        pathology = row["pathology"]
        if not pd.isna(pathology):
            pathologies = pathology.split(", ")
            pathology_counts = {}
            for path in pathologies:
                pathology_counts[path] = pathology_counts.get(path, 0) + 1
            pathology_list.update(pathology_counts.keys())
    pathology_list = list(pathology_list)
    metadata_stats["pathology"] = pathology_list

    with open(save_path, "w") as f:
        json.dump(metadata_stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the metadata statistics.",
    )
    parser.add_argument(
        "--metadata_file_knee",
        type=str,
        default=None,
        help="Path to the metadata file for knee data.",
    )
    parser.add_argument(
        "--metadata_file_brain",
        type=str,
        default=None,
        help="Path to the metadata file for brain data.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../fastmri",
        help="Root directory for the data.",
    )
    args = parser.parse_args()

    main(**vars(args))

"""
python3 get_metadata.py \
    --save_path metadata_stats.json \
    --metadata_file_brain /data/sls/scratch/hengjui/6S982/data/fastmri_sample/metadata_brain.csv \
    --data_root /data/sls/scratch/hengjui/6S982/data/fastmri_sample/fastmri
"""
