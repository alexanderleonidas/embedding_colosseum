import torch
import numpy as np
from dataset.DataManager import DataManager

def shannon_entropy(image, bins=256, normalise=True) -> float:
    """
    Compute the Shannon entropy of a single image tensor.
    Args:
        image: PyTorch tensor of shape (C, H, W) or (H, W)
        bins: Number of histogram bins to use (default 256)
        normalise: If True, normalise entropy to [0, 1] by dividing
    Returns:
        A float in bits representing the Shannon entropy of the image.
    """
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")

    image = image.detach().cpu().float().flatten()
    if image.numel() == 0:
        return 0.0

    # remove non-finite values
    finite_mask = torch.isfinite(image)
    if not finite_mask.all():
        image = image[finite_mask]
        if image.numel() == 0:
            return 0.0

    # ensure values are within histogram range
    image = torch.clamp(image, 0.0, 1.0)

    hist = torch.histc(image, bins=bins, min=0.0, max=1.0)
    total = float(hist.sum().item())
    if total == 0.0:
        return 0.0

    prob = hist / total
    prob = prob[prob > 0.0]
    entropy = -float((prob * torch.log2(prob)).sum().item())

    if normalise:
        if bins <= 1:
            return 0.0
        entropy = entropy / float(np.log2(bins))

    return entropy


# Example usage
if __name__ == '__main__':
    for dataset_name in ["mnist", "fashion", "cifar10", "stl10", "cxr8", "brain_tumor", "eurosat_rgb"]:
        dm = DataManager(cfg=None, batch_size=100, seed=42, pixel_size=120, dataset=dataset_name)
        train, _, _ = dm.get_loaders(1.0, 0, 0)
        results = []
        for batch_idx, data in enumerate(train):
            images, labels = data
            for i in range(images.size(0)):
                img = images[i]
                if img.shape[0] > 1:
                    img = torch.mean(img, dim=0, keepdim=True)
                result = shannon_entropy(img)
                results.append(result)
                # global_idx = batch_idx * images.shape[0] + i
                # print(f"Image {global_idx}: Shannon Entropy={result:.4f}")

        results = np.array(results)
        print(f"Processed {len(results)} images for {dataset_name}.")
        print(f"Mean Shannon Entropy: {results.mean():.4f} ± {results.std():.4f}")
        print(f"Shannon Entropy range: [{results.min():.4f}, {results.max():.4f}]")