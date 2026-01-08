import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import tqdm
from dataset.DataManager import DataManager


def compute_edge_density_from_dataloader(
    dataloader: DataLoader,
    threshold: float = 0.1,
    normalize: bool = True
) -> dict:
    """
    Compute edge density for all images in a PyTorch DataLoader using Sobel filters.

    Args:
        dataloader: PyTorch DataLoader yielding (images, labels) or images
        threshold: Gradient magnitude threshold for edge detection
        normalize: Whether to return density in [0, 1] (default True)

    Returns:
        Dictionary containing:
            - 'edge_densities': List of edge density values per image
            - 'mean_edge_density'
            - 'std_edge_density'
            - 'min_edge_density'
            - 'max_edge_density'
            - 'per_class_stats' (if labels available)
    """
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    edge_densities = []
    all_labels = []
    class_edge_densities = defaultdict(list)

    print(f"Computing edge density for {len(dataloader.dataset)} images...")
    print(f"Using device: {device}")

    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=torch.float32,
        device=device
    ).view(1, 1, 3, 3)

    with torch.no_grad():
        for batch_data in tqdm.tqdm(dataloader, desc="Processing images"):
            if isinstance(batch_data, (list, tuple)):
                images, labels = batch_data[0], batch_data[1]
            else:
                images = batch_data
                labels = None

            images = images.to(device)

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
                    gray = image if len(image.shape) == 2 else image.squeeze()

                # Normalize to [0,1]
                if gray.max() > 1.0:
                    gray = gray / 255.0

                # Add batch + channel dimensions
                gray = gray.unsqueeze(0).unsqueeze(0)

                # Sobel gradients
                gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
                gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)

                grad_mag = torch.sqrt(gx ** 2 + gy ** 2)

                # Edge mask
                edges = grad_mag > threshold
                edge_pixels = edges.sum().item()
                total_pixels = edges.numel()

                density = edge_pixels / total_pixels if normalize else edge_pixels
                edge_densities.append(density)

                if labels is not None:
                    label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                    all_labels.append(label)
                    class_edge_densities[label].append(density)

    edge_densities = np.array(edge_densities)
    all_labels = np.array(all_labels) if all_labels else None

    results = {
        "edge_densities": edge_densities,
        "mean_edge_density": float(np.mean(edge_densities)),
        "std_edge_density": float(np.std(edge_densities)),
        "min_edge_density": float(np.min(edge_densities)),
        "max_edge_density": float(np.max(edge_densities)),
        "threshold": threshold,
        "normalized": normalize,
        "total_images": len(edge_densities)
    }

    if class_edge_densities:
        per_class_stats = {}
        for class_label, densities in class_edge_densities.items():
            d = np.array(densities)
            per_class_stats[class_label] = {
                "mean": float(np.mean(d)),
                "std": float(np.std(d)),
                "min": float(np.min(d)),
                "max": float(np.max(d)),
                "count": len(d)
            }
        results["per_class_stats"] = per_class_stats

    return results


if __name__ == "__main__":
    # exp_dict = {"mnist": 28, "fashion": 28, "cifar10": 32, "stl10": 96, "cxr8": 1024, "brain_tumor": 640, "eurosat_rgb": 64}
    exp_dict = {"cifar10": 32}
    for dataset_name, pixel_size in exp_dict.items():
        dm = DataManager(batch_size=100,seed=42,pixel_size=pixel_size,dataset=dataset_name)
        train_loader, _, _ = dm.get_loaders(1, 0, 0)

        results = compute_edge_density_from_dataloader(train_loader,threshold=0.1,normalize=True)

        print(f"\n{'=' * 50}")
        print("Edge density summary for dataset:", dataset_name)
        print(f"{'=' * 50}")
        print(f"Total images processed: {results['total_images']}")
        print(f"Mean edge density: {results['mean_edge_density']:.4f} ± {results['std_edge_density']:.4f}")
        print(f"Edge density range: [{results['min_edge_density']:.4f}, {results['max_edge_density']:.4f}]")

        if "per_class_stats" in results:
            print("\nPer-class statistics:")
            for class_label, stats in results["per_class_stats"].items():
                print(
                    f"  Class {class_label}: {stats['count']} images, "
                    f"mean = {stats['mean']:.4f} ± {stats['std']:.4f}"
                )

            max_class = max(
                results["per_class_stats"].items(),
                key=lambda x: x[1]["mean"]
            )
            print(
                f"Class with highest mean edge density: "
                f"{max_class[0]} ({max_class[1]['mean']:.4f})"
            )