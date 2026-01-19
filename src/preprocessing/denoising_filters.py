import cv2
import numpy as np
import torch


class MedianBlur:
    """
    A PyTorch transform to apply a Median blur.
    Excellent for removing salt-and-pepper noise while preserving edges.
    """

    def __init__(self, kernel_size=3):
        # Kernel size must be an odd integer
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # Convert tensor [C, H, W] to numpy array [H, W, C] and scale to [0, 255]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Apply median blur
        blurred_np = cv2.medianBlur(img_np, self.kernel_size)

        # Handle case where image might have been squeezed to [H, W]
        if blurred_np.ndim == 2:
            blurred_np = np.expand_dims(blurred_np, axis=-1)

        # Convert back to tensor [C, H, W] and scale to [0, 1]
        return torch.from_numpy(blurred_np).permute(2, 0, 1).float() / 255.0


class BilateralFilter:
    """
    A PyTorch transform to apply a Bilateral filter.
    Excellent for smoothing while preserving sharp edges.
    """

    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = int(d)
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # Convert tensor [C, H, W] to numpy array [H, W, C] and scale to [0, 255]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Apply bilateral filter
        # Note: OpenCV's bilateralFilter works on 3-channel (color) or 1-channel (grayscale) images.
        blurred_np = cv2.bilateralFilter(
            img_np, self.d, self.sigma_color, self.sigma_space
        )

        # Handle case where image might have been squeezed to [H, W]
        if blurred_np.ndim == 2:
            blurred_np = np.expand_dims(blurred_np, axis=-1)

        # Convert back to tensor [C, H, W] and scale to [0, 1]
        return torch.from_numpy(blurred_np).permute(2, 0, 1).float() / 255.0
