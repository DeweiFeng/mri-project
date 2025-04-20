#!/usr/bin/env python3
"""
Create ContextMRI database from HDF5 files.

For each HDF5 file the script:
  1. Loads raw k-space data and transposes it to (slices, coils, H, W).
  2. Iterates over all slices and, for each:
      - Computes coil images via a centered IFFT
      - Estimates ESPIRiT sensitivity maps
      - Computes the MVUE
      - Center-crops the combined image and maps to 320x320
      - Normalizes the combined image by its 99th percentile magnitude
  3. Saves each slice’s image and maps as separate .npy files
"""

import os
import glob
import argparse
import h5py
import numpy as np
import sigpy as sp
import matplotlib.pyplot as plt
import sigpy.mri


def ifft2c(data):
    """
    Apply a centered 2D inverse FFT along the last two axes.
    data: (..., H, W)
    """
    data = np.fft.ifftshift(data, axes=(-2, -1))
    im_space = np.fft.ifft2(data, axes=(-2, -1), norm="ortho")
    im_space = np.fft.fftshift(im_space, axes=(-2, -1))
    return im_space

def center_crop(arr, crop_size=320):
    """
    Center-crop
    """
    if arr.ndim == 2:
        H, W = arr.shape
        sh, sw = (H - crop_size) // 2, (W - crop_size) // 2
        return arr[sh:sh+crop_size, sw:sw+crop_size]
    elif arr.ndim == 3:
        H, W = arr.shape[-2:]
        sh, sw = (H - crop_size) // 2, (W - crop_size) // 2
        return arr[..., sh:sh+crop_size, sw:sw+crop_size]
    else:
        raise ValueError("Unsupported array dimensions for cropping")

def process_slice(slice_kspace, calib_width=24, crop_size=320):
    """
    Process one slice of k-space data (shape: (num_coils, H, W))
    Returns:
      - combined_image_norm: Combined MVUE reconstruction (complex-valued, shape: (crop_size, crop_size))
      - coil_maps_norm: ESPIRiT sensitivity map (complex-valued, shape: (num_coils, crop_size, crop_size))
    """
    # Compute coil images via a centered IFFT.
    coil_imgs = ifft2c(slice_kspace)
    
    # ESPIRiT sensitivity maps
    espirit_app = sp.mri.app.EspiritCalib(
        slice_kspace,
        calib_width=calib_width,
        thresh=0.02,
        kernel_width=6,
        crop=0.95,
        device=sp.Device(-1)
    )
    coil_maps = espirit_app.run()
    print("coil maps shape: ", coil_maps.shape)
    
    # Coil Combination (MVUE):
    maps_conj = np.conjugate(coil_maps)
    numerator = np.sum(maps_conj * coil_imgs, axis=0)
    denominator = np.sum(np.abs(coil_maps)**2, axis=0)
    combined_image = np.zeros_like(numerator)
    valid = denominator > 0
    combined_image[valid] = numerator[valid] / denominator[valid]
    
    # Center crop both the combined image and sensitivity maps
    combined_image = center_crop(combined_image, crop_size)
    coil_maps = center_crop(coil_maps, crop_size)
    
    # Normalize the combined image by the 99th percentile of its magnitude
    mag = np.abs(combined_image)
    scale = np.percentile(mag, 99)
    if scale == 0:
        scale = 1.0
    combined_image_norm = combined_image / scale
    coil_maps_norm = coil_maps / scale

    return combined_image_norm.astype(np.complex64), coil_maps_norm.astype(np.complex64)

def process_file(file_path, output_root, train=True, calib_width=24, crop_size=320, show_plots=False):
    """
    Process one HDF5 file
    """
    patient_id = os.path.splitext(os.path.basename(file_path))[0]
    
    anatomy = "brain" if "brain" in patient_id.lower() else "knee"
    split_folder = f"{anatomy}_mvue_320_train" if train else f"{anatomy}_mvue_320_val"
    
    out_slice_dir = os.path.join(output_root, split_folder, "slice", patient_id)
    out_mps_dir   = os.path.join(output_root, split_folder, "mps",    patient_id)
    out_plots_dir = os.path.join(output_root, split_folder, "plots",  patient_id)
    
    os.makedirs(out_slice_dir, exist_ok=True)
    os.makedirs(out_mps_dir,   exist_ok=True)
    if show_plots:
        os.makedirs(out_plots_dir, exist_ok=True)
    
    with h5py.File(file_path, "r") as hf:
        volume_kspace = hf['kspace'][()]
    print("SHAPE: ", volume_kspace.shape)
    
    num_slices = volume_kspace.shape[0]
    print(f"Processing patient: {patient_id} | Total slices: {num_slices}")
    
    for slice_idx in range(num_slices):
        slice_kspace = volume_kspace[slice_idx]  # shape: (num_coils, H, W)

        # If plotting, get raw coil-images once per slice
        if show_plots:
            coil_imgs = ifft2c(slice_kspace)  # (C, H, W)
        
        combined_img, coil_maps = process_slice(slice_kspace, calib_width, crop_size)
        
        # Save arrays
        slice_filename = os.path.join(out_slice_dir, f"{slice_idx:03d}.npy")
        mps_filename   = os.path.join(out_mps_dir,   f"{slice_idx:03d}.npy")
        np.save(slice_filename, combined_img)
        np.save(mps_filename,   coil_maps)
        
        # Save diagnostic plots for every slice & every coil
        if show_plots:
            for coil_idx in range(coil_maps.shape[0]):
                # Crop raw coil‑n image
                raw = center_crop(coil_imgs[coil_idx], crop_size)
                
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(np.abs(raw),         cmap='gray')
                axs[0].set_title(f"Raw coil{coil_idx} |Iₙ|")
                
                axs[1].imshow(np.abs(combined_img), cmap='gray')
                axs[1].set_title("Combined |I|")
                
                axs[2].imshow(np.abs(coil_maps[coil_idx]), cmap='gray')
                axs[2].set_title(f"ESPIRiT mag Sₙ")
                
                axs[3].imshow(np.angle(coil_maps[coil_idx]), cmap='hsv')
                axs[3].set_title(f"ESPIRiT phase ∠Sₙ")
                
                for ax in axs:
                    ax.axis('off')
                
                plt.suptitle(f"{patient_id} | slice {slice_idx:03d} | coil {coil_idx:02d}")
                plot_path = os.path.join(out_plots_dir, f"{slice_idx:03d}_coil{coil_idx:02d}.png")
                fig.savefig(plot_path, bbox_inches='tight')
                plt.close(fig)
    
    print(f"Finished processing patient: {patient_id}\n  Saved slice data in: {out_slice_dir}\n  Saved mps data in: {out_mps_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Create ContextMRI database from fastMRI HDF5 files (one patient per file)."
    )
    parser.add_argument("--input_dir", type=str, default="./h5_files", 
                        help="Directory containing fastMRI HDF5 files")
    parser.add_argument("--output_root", type=str, default="../fastmri", 
                        help="Root directory for the output database")
    parser.add_argument("--train", action="store_true", 
                        help="Process training data (otherwise validation)")
    parser.add_argument("--calib_width", type=int, default=24, 
                        help="Calibration width for ESPIRiT (default: 24 from docs)")
    parser.add_argument("--crop_size", type=int, default=320, 
                        help="Final cropped image size")
    parser.add_argument("--show_plots", action="store_true", 
                        help="Show plots for the first slice of each patient (for debugging)")
    args = parser.parse_args()
    
    h5_files = glob.glob(os.path.join(args.input_dir, "*.h5"))
    
    for file_path in h5_files:
        process_file(
            file_path=file_path,
            output_root=args.output_root,
            train=args.train,
            calib_width=args.calib_width,
            crop_size=args.crop_size,
            show_plots=args.show_plots
        )

if __name__ == "__main__":
    main()