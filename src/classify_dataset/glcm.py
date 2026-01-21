import torch
import math
import numpy as np

from skimage.feature import graycomatrix, graycoprops
from dataset.DataManager import DataManager


def glcm(image, distances=(1,), angles=(0, math.pi/4, math.pi/2, 3*math.pi/4), levels=256, symmetric=True, normed=True):
    """
    Compute GLCM stats for a single image.

    Args:
        image: torch.Tensor or numpy array of shape (C,H,W) or (H,W
        distances: List or tuple of pixel pair distance offsets.
        angles: List or tuple of pixel pair angles in radians.
        levels: Number of gray levels for quantization (default 256).
        symmetric: If True, the GLCM will be symmetric.
        normed: If True, the GLCM will be normalized.

    Returns:
        Dict with GLCM properties homogeneity, energy, correlation.
    """
    # Convert torch to numpy
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.array(image)

    # Ensure arr is 2D (H,W). If (C,H,W): average channels or squeeze if single channel.
    if arr.ndim == 3:
        # Expect (C, H, W)
        if arr.shape[0] > 1:
            arr2 = arr.mean(axis=0)
        else:
            arr2 = arr[0]
    elif arr.ndim == 2:
        arr2 = arr
    else:
        raise ValueError("`image` must be 2D or 3D (C,H,W). Got ndim=%d" % arr.ndim)

    # Replace non-finite values and clamp to [0,1]
    arr2 = np.nan_to_num(arr2, nan=0.0, posinf=0.0, neginf=0.0)
    arr2 = np.clip(arr2, 0.0, 1.0)

    # Map to integer levels [0, levels-1]
    if levels <= 0:
        raise ValueError("levels must be > 0")
    if levels <= 256:
        dtype = np.uint8
    elif levels <= 65536:
        dtype = np.uint16
    else:
        # graycomatrix accepts integer arrays; large levels may be impractical
        dtype = np.uint32

    quant = (arr2 * (levels - 1)).round().astype(dtype)

    # Ensure quant is 2D numpy array (H, W)
    if quant.ndim != 2:
        raise ValueError("Quantized image must be 2D before calling graycomatrix")

    # Compute GLCM and averaged properties
    glcm_mat = graycomatrix(quant, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)

    homogeneity = float(np.mean(graycoprops(glcm_mat, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm_mat, "energy")))
    correlation = float(np.mean(graycoprops(glcm_mat, "correlation")))

    return {"homogeneity": homogeneity, "energy": energy, "correlation": correlation}

if __name__ == "__main__":
    dm = DataManager(cfg=None, batch_size=100, seed=42, pixel_size=120, dataset="fashion")
    train, _, _ = dm.get_loaders(1, 0, 0)
    for imgs, labels in train:
        for i in range(imgs.size(0)):
            img = imgs[i]
            print(img.size())
            glcm_stats = glcm(img)
            print(f"Image {i}: GLCM Homogeneity={glcm_stats['homogeneity']:.4f}, Energy={glcm_stats['energy']:.4f}, Correlation={glcm_stats['correlation']:.4f}")
        break
    # save_results = True
    # datasets = ["mnist", "fashion", "cifar10", "stl10", "cxr8", "brain_tumor", "eurosat_rgb"]
    # pixel_sizes = [28, 120, 440, 720, 1024]
    # for d in datasets:
    #     for p in pixel_sizes:
    #         # Pixel sizes must be consistent across datasets in order to have comparable GLCM statistics
    #         dm = DataManager(batch_size=100, seed=42, pixel_size=p, dataset=d)
    #         train_loader, _, _ = dm.get_loaders(0.1, 0.1, 0.8)
    #
    #         results = compute_glcm_statistics_from_dataloader(train_loader)
    #
    #         if save_results:
    #             file_name = "glcm_results.csv"
    #             file_exists = os.path.isfile(file_name)
    #             with open(file_name, "a") as f:
    #                 if not file_exists:
    #                     f.write("dataset,pixel_size,mean_contrast,std_contrast,mean_dissimilarity,std_dissimilarity,mean_homogeneity,std_homogeneity,mean_energy,std_energy,mean_correlation,std_correlation,mean_asm,std_asm\n")
    #                 res = results["dataset_statistics"]
    #                 f.write(f"{d},{p},{res['contrast']['mean']},{res['contrast']['std']},{res['dissimilarity']['mean']},{res['dissimilarity']['std']},{res['homogeneity']['mean']},{res['homogeneity']['std']},{res['energy']['mean']},{res['energy']['std']},{res['correlation']['mean']},{res['correlation']['std']},{res['ASM']['mean']},{res['ASM']['std']}\n")
    #
    #         print(f"\n{'=' * 60}")
    #         print(f"GLCM summary for dataset: {d}")
    #         print(f"{'=' * 60}")
    #
    #         for feature, stats in results["dataset_statistics"].items():
    #             print(
    #                 f"{feature:15s} "
    #                 f"mean = {stats['mean']:.4f} ± {stats['std']:.4f}"
    #             )