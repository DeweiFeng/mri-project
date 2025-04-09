import sys

sys.path.append("..")

from dataset_mri import MRIDataset


def test_dataset():
    dataset = MRIDataset(
        metadata_file_knee=None,
        metadata_file_brain="/data/sls/scratch/hengjui/6S982/data/fastmri_sample/metadata_brain.csv",
        train=True,
        data_root="/data/sls/scratch/hengjui/6S982/data/fastmri_sample/fastmri",
    )
    print(f"Dataset length: {len(dataset)}")
    data = dataset[0]
    print("Sample:")
    print(f"  Image shape:  {data['image'].shape}")
    print(f"  Prompt:       {data['prompt']}")
    print(f"  Metadata:     {data['metadata']}")


if __name__ == "__main__":
    test_dataset()
