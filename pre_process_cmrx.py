import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sigpy as sp
import argparse
from sigpy.mri.app import EspiritCalib, SenseRecon

def load_ds(path, key):
    with h5py.File(path, 'r') as f:
        def rec(g):
            for name, obj in g.items():
                if isinstance(obj, h5py.Dataset) and key.lower() in name.lower():
                    arr = obj[()]
                    if hasattr(arr.dtype, 'names') and arr.dtype.names:
                        r,i = arr.dtype.names[:2]
                        arr = arr[r] + 1j*arr[i]
                    return arr
                if isinstance(obj, h5py.Group):
                    out = rec(obj)
                    if out is not None:
                        return out
        return rec(f)

def process_modality(patient_dir, patient_id, modality, slice_idx=0,
                     pad_sz=320, out_root="cardimri/cardi_mvue_320_val"):
    ks_path = os.path.join(patient_dir, f"{modality}_ks.mat")
    calib_path = os.path.join(patient_dir, f"{modality}_calib.mat")

    out_slice_dir = os.path.join(out_root, "slice", f"{patient_id}_{modality}")
    out_mps_dir = os.path.join(out_root, "mps", f"{patient_id}_{modality}")
    os.makedirs(out_slice_dir, exist_ok=True)
    os.makedirs(out_mps_dir, exist_ok=True)
    
    ksp_all = load_ds(ks_path,    "Recon_ks")
    acs_all = load_ds(calib_path, "Calib")
    nframe, nslice, ncoil, ky_full, kx_full = ksp_all.shape
    a, b, c, ky_acs, d = acs_all.shape
    
    crop_sz = min(ky_full, kx_full)
    print(crop_sz)
    
    ky_lo = (ky_full - ky_acs)//2
    ky_hi = ky_lo + ky_acs
    for fr in range(nframe):
        ksp_us  = ksp_all [fr, slice_idx]
        acs_blk = acs_all[fr, slice_idx]
        ksp_c = ksp_us.copy()
        ksp_c[:, ky_lo:ky_hi, :] = acs_blk
        csm   = EspiritCalib(ksp_c, calib_width=ky_acs,
                             thresh=0.02, kernel_width=6, crop=0.9,
                             device=sp.Device(-1)).run()
        recon = SenseRecon(ksp_us, csm, max_iter=30).run()
        H, W = recon.shape
        sh, sw = (H-crop_sz)//2, (W-crop_sz)//2
        cropped = recon[sh:sh+crop_sz, sw:sw+crop_sz]
        pad_h = (pad_sz - crop_sz)//2
        pad_w = (pad_sz - crop_sz)//2
        padded = np.pad(
            cropped,
            ((pad_h, pad_sz-crop_sz-pad_h),
             (pad_w, pad_sz-crop_sz-pad_w)),
            mode='constant',
            constant_values=0
        )
        
        mag = np.abs(padded)
        p99 = np.percentile(mag, 99)
        if p99 <= 0: p99 = 1.0
        padded = padded / p99
        
        slice_fname = os.path.join(out_slice_dir, f"{fr:03d}.npy")
        np.save(slice_fname, padded.astype(np.complex64))
        
        H_csm, W_csm = csm.shape[-2:]
        sh_csm, sw_csm = (H_csm-crop_sz)//2, (W_csm-crop_sz)//2
        
        if csm.ndim == 3:
            csm_cropped = csm[:, sh_csm:sh_csm+crop_sz, sw_csm:sw_csm+crop_sz]
            csm_padded = np.pad(
                csm_cropped,
                ((0, 0),
                 (pad_h, pad_sz-crop_sz-pad_h),
                 (pad_w, pad_sz-crop_sz-pad_w)),
                mode='constant',
                constant_values=0
            )
        else:
            csm_cropped = csm[sh_csm:sh_csm+crop_sz, sw_csm:sw_csm+crop_sz]
            csm_padded = np.pad(
                csm_cropped,
                ((pad_h, pad_sz-crop_sz-pad_h),
                 (pad_w, pad_sz-crop_sz-pad_w)),
                mode='constant',
                constant_values=0
            )
        
        csm_padded = csm_padded / p99
        
        mps_fname = os.path.join(out_mps_dir, f"{fr:03d}.npy")
        np.save(mps_fname, csm_padded.astype(np.complex64))
        
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True,
                   help="Root folder containing P001/, P002/, ...")
    p.add_argument("--output_dir", required=True,
                   help="Where to write cardi_mvue_320_train/")
    p.add_argument("--start_idx", type=int, default=0,
                   help="First patient folder index (inclusive)")
    p.add_argument("--end_idx", type=int, default=None,
                   help="Last patient index (exclusive; default=end)")
    p.add_argument("--slice", type=int, default=0,
                   help="Fixed slice index to reconstruct")
    p.add_argument("--modalities", nargs="+",
                   default=["cine_lax", "cine_sax"],
                   help="Which modalities to run")
    p.add_argument("--pad_sz", type=int, default=320,
                   help="Zero‐pad output size")
    args = p.parse_args()

    pats = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    end = args.end_idx or len(pats)
    pats = pats[args.start_idx : end]

    for patient_id in pats:
        patient_dir = os.path.join(args.input_dir, patient_id)
        for mod in args.modalities:
            process_modality(
                patient_dir, patient_id, mod,
                slice_idx=args.slice,
                pad_sz=args.pad_sz,
                out_root=args.output_dir
            )

if __name__ == "__main__":
    main()