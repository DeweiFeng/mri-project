import json
import numpy as np

def compute_metrics_stats(json_file_path):
    """
    Loads a JSON file containing PSNR, SSIM, and LPIPS values per slice,
    calculates the mean and standard deviation for each metric, and prints the results.
    """
    # Load data from JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract metrics
    psnrs = [entry['psnr'] for entry in data]
    ssims = [entry['ssim'] for entry in data]
    lpips = [entry['lpips'] for entry in data]

    # Compute statistics
    stats = {
        'PSNR': {
            'mean': np.mean(psnrs),
            'std': np.std(psnrs)
        },
        'SSIM': {
            'mean': np.mean(ssims),
            'std': np.std(ssims)
        },
        'LPIPS': {
            'mean': np.mean(lpips),
            'std': np.std(lpips)
        }
    }
    
    # Print results
    print("Metric Statistics:")
    for metric, values in stats.items():
        print(f"{metric}: mean = {values['mean']:.4f}, std = {values['std']:.4f}")

if __name__ == "__main__":
    # Update the filename if needed
    compute_metrics_stats('D:\\Research\\mri-project\\ContextMRI\\result_new\\recon_complex_multi\\uniform1d\\acc_4\\cfg1.0\\eta0.8\\summary.json')
