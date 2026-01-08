import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import tqdm
from dataset.DataManager import DataManager


def compute_shannon_entropy_from_dataloader(dataloader: DataLoader, bins: int = 256, normalize: bool = True) -> dict:
    """
    Compute Shannon Entropy for all images in a PyTorch DataLoader.

    Args:
        dataloader: PyTorch DataLoader yielding (images, labels) or images
        bins: Number of bins for histogram (default 256 for 8-bit images)
        normalize: Whether to normalize entropy to [0, 1] (divide by log2(bins))

    Returns:
        Dictionary containing:
            - 'entropies': List of entropy values for each image
            - 'mean_entropy': Mean entropy across dataset
            - 'std_entropy': Standard deviation of entropy
            - 'min_entropy': Minimum entropy
            - 'max_entropy': Maximum entropy
            - 'per_class_stats': Dictionary with entropy stats per class (if labels available)
    """
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Initialize storage
    all_entropies = []
    all_labels = []

    # For per-class statistics
    class_entropies = defaultdict(list)

    print(f"Computing Shannon Entropy for {len(dataloader.dataset)} images...")
    print(f"Using device: {device}")

    # Process batches
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm.tqdm(dataloader, desc="Processing images")):
            # Handle different DataLoader return formats
            if isinstance(batch_data, (list, tuple)):
                images, labels = batch_data[0], batch_data[1]
            else:
                images = batch_data
                labels = None

            images = images.to(device)

            # Process each image in the batch
            for i in range(images.shape[0]):
                image = images[i]

                # Handle different image formats
                if len(image.shape) == 4:  # Batch of images with color: [B, C, H, W]
                    image = image[i]

                # Convert to grayscale if RGB (weighted average)
                if image.shape[0] == 3:  # RGB: [C, H, W]
                    # Convert to grayscale using luminance weights
                    # Using PyTorch operations for GPU compatibility
                    gray_image = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2])
                    intensity_tensor = gray_image
                elif image.shape[0] == 1:  # Grayscale: [1, H, W]
                    intensity_tensor = image[0]
                else:  # Single channel or already 2D
                    if len(image.shape) == 2:
                        intensity_tensor = image
                    else:
                        # Flatten to 2D if needed
                        intensity_tensor = image.view(image.shape[-2], image.shape[-1])

                # Normalize to [0, 1] if needed
                if intensity_tensor.max() > 1.0:
                    intensity_tensor = intensity_tensor / 255.0

                # Compute histogram
                # Move to CPU for histogram computation
                intensity_np = intensity_tensor.cpu().numpy().flatten()

                # Compute histogram and probabilities
                hist, bin_edges = np.histogram(intensity_np, bins=bins, range=(0.0, 1.0))
                probabilities = hist / np.sum(hist)

                # Remove zero probabilities to avoid log(0)
                probabilities = probabilities[probabilities > 0]

                # Compute Shannon Entropy: H = -Σ p_i * log2(p_i)
                entropy = -np.sum(probabilities * np.log2(probabilities))

                # Normalize to [0, 1] if requested
                if normalize:
                    entropy = entropy / np.log2(bins)

                all_entropies.append(entropy)

                # Store label if available
                if labels is not None:
                    label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                    all_labels.append(label)
                    class_entropies[label].append(entropy)

    # Convert to numpy arrays
    all_entropies = np.array(all_entropies)
    all_labels = np.array(all_labels) if all_labels else None

    # Compute overall statistics
    results = {
        'entropies': all_entropies,
        'mean_entropy': float(np.mean(all_entropies)),
        'std_entropy': float(np.std(all_entropies)),
        'min_entropy': float(np.min(all_entropies)),
        'max_entropy': float(np.max(all_entropies)),
        'bins': bins,
        'normalized': normalize,
        'total_images': len(all_entropies)
    }

    # Compute per-class statistics if labels are available
    if class_entropies:
        per_class_stats = {}
        for class_label, entropies in class_entropies.items():
            entropies_arr = np.array(entropies)
            per_class_stats[class_label] = {
                'mean': float(np.mean(entropies_arr)),
                'std': float(np.std(entropies_arr)),
                'min': float(np.min(entropies_arr)),
                'max': float(np.max(entropies_arr)),
                'count': len(entropies_arr)
            }
        results['per_class_stats'] = per_class_stats

    return results

if __name__ == '__main__':
    exp_dict = {"mnist": 28, "fashion": 28, "cifar10": 32, "stl10": 96, "cxr8": 1024, "brain_tumor": 640, "eurosat_rgb": 64}
    for dataset_name, pixel_size in exp_dict.items():
        dm = DataManager(batch_size=100, seed=42, pixel_size=pixel_size, dataset=dataset_name)
        train_loader, _, _ = dm.get_loaders(1,0,0)

        # Compute entropy (CPU version)
        results = compute_shannon_entropy_from_dataloader(train_loader,bins=256,normalize=True)

        # Print summary
        print(f"\n{'=' * 50}")
        print("Shannon entropy summary for dataset:", dataset_name)
        print(f"{'=' * 50}")
        print(f"Total images processed: {results['total_images']}")
        print(f"Mean entropy: {results['mean_entropy']:.4f} ± {results['std_entropy']:.4f}")
        print(f"Entropy range: [{results['min_entropy']:.4f}, {results['max_entropy']:.4f}]")

        if 'per_class_stats' in results:
            print(f"\nPer-class statistics:")
            for class_label, stats in results['per_class_stats'].items():
                print(f"  Class {class_label}: {stats['count']} images, "
                      f"mean = {stats['mean']:.4f} ± {stats['std']:.4f}")

        # Per-class analysis
        if 'per_class_stats' in results:
            max_mean_class = max(results['per_class_stats'].items(),key=lambda x: x[1]['mean'])
            print(f"Class with highest mean entropy: {max_mean_class[0]} "
                  f"({max_mean_class[1]['mean']:.4f})")
