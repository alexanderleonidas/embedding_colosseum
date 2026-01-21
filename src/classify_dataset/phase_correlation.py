import numpy as np
import torch
import tqdm
from scipy import ndimage
from scipy.fft import fft2, ifft2

from dataset.DataManager import DataManager


def phase_correlation(image, ref_image):
    """
    Compute phase correlation between two images.
    Args:
        image: 2D numpy array
        ref_image: 2D numpy array
    Returns:
        Peak value of the phase correlation surface
    """
    # Compute Fast Fourier Transform
    f1 = fft2(image)
    f2 = fft2(ref_image)

    # Compute cross-power spectrum
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)

    # Compute inverse FFT to get correlation surface
    corr = np.abs(ifft2(cross_power))

    # return the maximum peak value
    return corr.max()


def compute_symmetry_score(image):
    """
    Compute symmetry scores for a PyTorch image tensor.

    Args:
        image: PyTorch tensor of shape (C, H, W) or (H, W)
    Returns:
        Average symmetry scores for reflection and rotation
    """

    # Convert to numpy
    image = image.cpu().numpy()

    pc_lr = phase_correlation(image, np.fliplr(image))
    pc_ud = phase_correlation(image, np.flipud(image))
    pc_rot = phase_correlation(image, ndimage.rotate(image, 90, reshape=False, order=1))

    return (pc_lr + pc_ud + pc_rot) / 3


# Example usage
if __name__ == "__main__":
    for dataset_name in [
        "mnist",
        "fashion",
        "cifar10",
        "stl10",
        "cxr8",
        "brain_tumor",
        "eurosat_ms",
    ]:
        dm = DataManager(
            cfg=None, batch_size=100, seed=42, pixel_size=256, dataset=dataset_name
        )
        train, _, _ = dm.get_loaders(1, 0, 0)
        results = []

        for imgs, labels in tqdm.tqdm(train):
            for i in range(imgs.size(0)):
                img = imgs[i]
                # if img.shape[0] > 1:
                #     # Convert to grayscale if image has multiple channels
                #     img = torch.mean(img, dim=0, keepdim=True)
                pc_sym = compute_symmetry_score(img)
                results.append(pc_sym)
                # print(f"Image {i}: Phase Correlation Symmetry={pc_sym:.4f}")
        results = np.array(results)
        print(f"dataset: {dataset_name}")
        print(
            f"Phase Correlation Symmetry - Mean: {results.mean():.4f}, Std: {results.std():.4f}"
        )
        print(f"Max: {results.max():.4f}, Min: {results.min():.4f}")
