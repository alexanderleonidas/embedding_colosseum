import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import tqdm

from skimage.feature import graycomatrix, graycoprops
from dataset.DataManager import DataManager


def compute_glcm_statistics_from_dataloader(
    dataloader: DataLoader,
    distances=(1,),
    angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    levels=256,
    symmetric=True,
    normed=True
) -> dict:
    """
    Compute GLCM statistics for all images in a DataLoader.

    Args:
        dataloader: PyTorch DataLoader yielding (images, labels)
        distances: Pixel pair distance offsets
        angles: Pixel pair angles (radians)
        levels: Number of gray levels
        symmetric: Whether GLCM is symmetric
        normed: Whether to normalize GLCM

    Returns:
        Dictionary with per-image stats, dataset stats, and per-class stats
    """

    # Storage
    features = defaultdict(list)
    class_features = defaultdict(lambda: defaultdict(list))

    print(f"Computing GLCM statistics for {len(dataloader.dataset)} images...")

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Processing images"):
            images, labels = batch

            for i in range(images.shape[0]):
                image = images[i]

                # Convert to grayscale
                if image.shape[0] == 3:
                    gray = (
                        0.299 * image[0] +
                        0.587 * image[1] +
                        0.114 * image[2]
                    )
                elif image.shape[0] == 1:
                    gray = image[0]
                else:
                    gray = image.squeeze()

                # Normalize to [0, levels-1]
                gray = gray.cpu().numpy()
                gray = gray - gray.min()
                if gray.max() > 0:
                    gray = gray / gray.max()
                gray = (gray * (levels - 1)).astype(np.uint8)

                # Compute GLCM
                glcm = graycomatrix(
                    gray,
                    distances=distances,
                    angles=angles,
                    levels=levels,
                    symmetric=symmetric,
                    normed=normed
                )

                # Compute statistics (averaged over distances & angles)
                haralick_stats = {
                    "contrast": np.mean(graycoprops(glcm, "contrast")),
                    "dissimilarity": np.mean(graycoprops(glcm, "dissimilarity")),
                    "homogeneity": np.mean(graycoprops(glcm, "homogeneity")),
                    "energy": np.mean(graycoprops(glcm, "energy")),
                    "correlation": np.mean(graycoprops(glcm, "correlation")),
                    "ASM": np.mean(graycoprops(glcm, "ASM"))
                }

                for k, v in haralick_stats.items():
                    features[k].append(v)

                label = labels[i].item()
                for k, v in haralick_stats.items():
                    class_features[label][k].append(v)

    # Convert to numpy arrays
    for k in features:
        features[k] = np.array(features[k])

    # Dataset-level statistics
    dataset_stats = {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v))
        }
        for k, v in features.items()
    }

    # Per-class statistics
    per_class_stats = {}
    for cls, feats in class_features.items():
        per_class_stats[cls] = {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "count": len(v)
            }
            for k, v in feats.items()
        }

    return {
        "per_image_features": features,
        "dataset_statistics": dataset_stats,
        "per_class_statistics": per_class_stats,
        "glcm_parameters": {
            "distances": distances,
            "angles": angles,
            "levels": levels,
            "symmetric": symmetric,
            "normed": normed
        }
    }


if __name__ == "__main__":
    exp_dict = {
        "mnist": 28,
        "fashion": 28,
        "cifar10": 32,
        "stl10": 96,
        "cxr8": 1024,
        "brain_tumor": 640,
        "eurosat_rgb": 64
    }

    for dataset_name, pixel_size in exp_dict.items():
        dm = DataManager(
            batch_size=64,
            seed=42,
            pixel_size=pixel_size,
            dataset=dataset_name
        )
        # print(dm.root)
        train_loader, _, _ = dm.get_loaders(0.1, 0.1, 0.8)

        results = compute_glcm_statistics_from_dataloader(train_loader)

        print(f"\n{'=' * 60}")
        print(f"GLCM summary for dataset: {dataset_name}")
        print(f"{'=' * 60}")

        for feature, stats in results["dataset_statistics"].items():
            print(
                f"{feature:15s} "
                f"mean = {stats['mean']:.4f} ± {stats['std']:.4f}"
            )