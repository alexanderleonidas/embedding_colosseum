import numpy as np
import torch
from dataset.DataManager import DataManager

def edge_density(image, threshold=0.1, normalise=True):

    # Convert to grayscale if image has multiple channels
    if image.size(0) > 1:
        image = torch.mean(image, dim=0, keepdim=True)

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    # Sobel gradients
    gx = torch.nn.functional.conv2d(image, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(image, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2)

    # Edge mask
    edges = grad_mag > threshold
    edge_pixels = int(edges.sum().item())
    total_pixels = int(edges.numel())

    return edge_pixels / total_pixels if normalise else edge_pixels

# Example usage
if __name__ == "__main__":
    for dataset_name in ["mnist", "fashion", "cifar10", "stl10", "cxr8", "brain_tumor", "eurosat_rgb"]:
        dm = DataManager(cfg=None, batch_size=100, seed=42, pixel_size=120, dataset=dataset_name)
        train, _, _ = dm.get_loaders(1, 0, 0)
        results = []
        for imgs, labels in train:
            for i in range(imgs.size(0)):
                img = imgs[i]
                density = edge_density(img, threshold=0.1, normalise=True)
                results.append(density)
        results = np.array(results)
        print(f"Dataset {dataset_name}: Mean Edge Density={results.mean():.4f}")