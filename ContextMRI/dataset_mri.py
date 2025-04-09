import os
import pandas as pd
import numpy as np
import random
import torch
from utils import row_to_text_string, row_to_text_string_skm_tea, row_to_dict


class SKMDataset:
    def __init__(self, metadata_file, train=True):
        self.metadata = pd.read_csv(metadata_file)
        self.train = train

        valid_rows = []
        for idx, row in self.metadata.iterrows():
            anatomy = row["anatomy"]
            filename = row["filename"]
            echo_num = row["echo"]
            slice_number = row["slice"]
            if self.train:
                file_path = os.path.join(
                    f"../skm-tea/processed_data/train/slice",
                    filename,
                    f"echo{echo_num}_{slice_number:03d}.npy",
                )
            else:
                file_path = os.path.join(
                    f"../skm-tea/processed_data/val/slice",
                    filename,
                    f"echo{echo_num}_{slice_number:03d}.npy",
                )

            if os.path.exists(file_path):
                valid_rows.append(row)

        self.metadata = pd.DataFrame(valid_rows).reset_index(drop=True)

    def get_item_from_index(self, idx):

        row = self.metadata.iloc[idx]
        anatomy = row["anatomy"]
        filename = row["filename"]
        echo_num = row["echo"]
        slice_number = row["slice"]
        pathology = row["pathology"]

        if self.train:
            text = row_to_text_string_skm_tea(row)
        else:
            text = row_to_text_string_skm_tea(row, p=1.0)

        if self.train:
            file_path = os.path.join(
                f"../skm-tea/processed_data/train/slice",
                filename,
                f"echo{echo_num}_{slice_number:03d}.npy",
            )
        else:
            file_path_img = os.path.join(
                f"../skm-tea/processed_data/val/slice",
                filename,
                f"echo{echo_num}_{slice_number:03d}.npy",
            )
            file_path_mps = os.path.join(
                f"../skm-tea/processed_data/val/mps",
                filename,
                f"{slice_number:03d}.npy",
            )

        if self.train:
            if os.path.exists(file_path):
                image_data = np.load(file_path)
                real = image_data.real.astype(np.float32)
                imag = image_data.imag.astype(np.float32)
                real = torch.from_numpy(real).float().unsqueeze(0)
                imag = torch.from_numpy(imag).float().unsqueeze(0)
                img = torch.cat([real, imag], dim=0)
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        else:
            if os.path.exists(file_path_img) and os.path.exists(file_path_mps):
                image_data = np.load(file_path_img)
                mps_data = np.load(file_path_mps)
                image_data = torch.tensor(image_data).unsqueeze(0)
                mps_data = torch.tensor(mps_data)
            else:
                raise FileNotFoundError(f"File not found: {file_path_img}")

            if pd.isna(pathology):
                pathology = "null"

        if self.train:
            return img, text
        else:
            return (
                image_data,
                mps_data,
                text,
                filename,
                slice_number,
                pathology,
                anatomy,
            )

    def __getitem__(self, idx):

        if self.train:
            img, text = self.get_item_from_index(idx)
        else:
            img, mps, text, filename, slice_number, pathology, anatomy = (
                self.get_item_from_index(idx)
            )

        if self.train:
            if img.shape != (2, 512, 512):
                raise ValueError(f"Not supported Image size: {img.shape}")
        else:
            if img.shape != (1, 512, 512):
                raise ValueError(f"Not supported Image size: {img.shape}")

        if self.train:
            # dropout condition for unconditional generation
            p = random.random()
            if p < 0.1:
                text = ""

            return {"image": img, "prompt": text}
        else:
            return {
                "image": img,
                "mps": mps,
                "prompt": text,
                "filename": filename,
                "slice_number": slice_number,
                "pathology": pathology,
                "anatomy": anatomy,
            }

    def __len__(self):
        return len(self.metadata)


class MRIDataset:
    def __init__(
        self,
        metadata_file_knee=None,
        metadata_file_brain=None,
        train=True,
        data_root="../fastmri",
    ):
        assert metadata_file_knee is not None or metadata_file_brain is not None
        # self.metadata_knee = pd.read_csv(metadata_file_knee)
        # self.metadata_brain = pd.read_csv(metadata_file_brain)
        self.train = train
        self.data_root = data_root

        # self.metadata_knee_prev = self.metadata_knee.copy()

        def read_metadata(metadata_file):
            valid_rows = []
            metadata_file = pd.read_csv(metadata_file)
            for idx, row in metadata_file.iterrows():
                anatomy = row["anatomy"]
                filename = row["filename"]
                slice_number = row["slice"]
                if self.train:
                    file_path = os.path.join(
                        data_root,
                        f"{anatomy}_mvue_320_train",
                        "slice",
                        filename,
                        f"{slice_number:03d}.npy",
                    )
                else:
                    file_path = os.path.join(
                        data_root,
                        f"{anatomy}_mvue_320_val",
                        "slice",
                        filename,
                        f"{slice_number:03d}.npy",
                    )

                if os.path.exists(file_path):
                    valid_rows.append(row)

            return valid_rows

        self.metadata_knee = None
        self.metadata_brain = None
        self.valid_rows = []
        if metadata_file_knee is not None:
            valid_rows_knee = read_metadata(metadata_file_knee)
            self.valid_rows.extend(valid_rows_knee)
            self.metadata_knee = pd.DataFrame(valid_rows_knee).reset_index(drop=True)
        if metadata_file_brain is not None:
            valid_rows_brain = read_metadata(metadata_file_brain)
            self.valid_rows.extend(valid_rows_brain)
            self.metadata_brain = pd.DataFrame(valid_rows_brain).reset_index(drop=True)

        # Combine two metadata into one
        if self.metadata_knee is not None and self.metadata_brain is not None:
            self.metadata = pd.concat(
                [self.metadata_knee, self.metadata_brain], ignore_index=True
            )
        elif self.metadata_knee is not None:
            self.metadata = self.metadata_knee
        elif self.metadata_brain is not None:
            self.metadata = self.metadata_brain
        else:
            raise ValueError("No valid metadata found.")

    def get_item_from_index(self, idx):

        row = self.metadata.iloc[idx]
        anatomy = row["anatomy"]
        filename = row["filename"]
        slice_number = row["slice"]
        pathology = row["pathology"]

        if self.train:
            text = row_to_text_string(row)
        else:
            text = row_to_text_string(row, p=1.0)

        metadata_dict = row_to_dict(row)

        # Change the path in your custom MRI data repository
        if self.train:
            file_path = os.path.join(
                self.data_root,
                f"{anatomy}_mvue_320_train",
                "slice",
                filename,
                f"{slice_number:03d}.npy",
            )
        else:
            file_path_img = os.path.join(
                self.data_root,
                f"{anatomy}_mvue_320_val",
                "slice",
                filename,
                f"{slice_number:03d}.npy",
            )
            file_path_mps = os.path.join(
                self.data_root,
                f"{anatomy}_mvue_320_val",
                "mps",
                filename,
                f"{slice_number:03d}.npy",
            )

        # The file path should be contained numpy complex64 values
        if self.train:
            if os.path.exists(file_path):
                image_data = np.load(file_path)
                real = image_data.real.astype(np.float32)
                imag = image_data.imag.astype(np.float32)
                real = torch.from_numpy(real).float().unsqueeze(0)
                imag = torch.from_numpy(imag).float().unsqueeze(0)
                img = torch.cat([real, imag], dim=0)
            else:
                raise FileNotFoundError(f"File not found: {file_path}")

        else:
            if os.path.exists(file_path_img) and os.path.exists(file_path_mps):
                image_data = np.load(file_path_img)
                mps_data = np.load(file_path_mps)
                image_data = torch.tensor(image_data).unsqueeze(0)
                mps_data = torch.tensor(mps_data)

        if pd.isna(pathology):
            pathology = "null"

        if self.train:
            return img, text, metadata_dict
        else:
            return (
                image_data,
                mps_data,
                text,
                metadata_dict,
                filename,
                slice_number,
                pathology,
                anatomy,
            )

    def __getitem__(self, idx):

        if self.train:
            img, text, metadata_dict = self.get_item_from_index(idx)
        else:
            (
                img,
                mps,
                text,
                metadata_dict,
                filename,
                slice_number,
                pathology,
                anatomy,
            ) = self.get_item_from_index(idx)

        if self.train:
            if img.shape != (2, 320, 320):
                raise ValueError(f"Not supported Image size: {img.shape}")
        else:
            if img.shape != (1, 320, 320):
                raise ValueError(f"Not supported Image size: {img.shape}")

        if self.train:
            # p = random.random()
            # if p < 0.1:
            #     text = ""
            # NOTE: we'll implement CFG in the training loop
            return {"image": img, "prompt": text, "metadata": metadata_dict}

        else:
            return {
                "image": img,
                "mps": mps,
                "prompt": text,
                "metadata": metadata_dict,
                "filename": filename,
                "slice_number": slice_number,
                "pathology": pathology,
                "anatomy": anatomy,
            }

    def __len__(self):
        return len(self.metadata)
