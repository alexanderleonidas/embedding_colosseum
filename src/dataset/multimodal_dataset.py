from typing import List, Optional, Union, Any, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiModalImageDataset(Dataset):
    """
    Dataset that loads images of varying channels.
    Can accept either:
      - a list of image file paths (List[str])
      - an existing torchvision-style Dataset (returns image or (image, label))
    Channel padding is handled by a custom collate function if `preserve_channels` is True.
    """

    def __init__(self,
                 image_paths: Union[List[str], Dataset],
                 max_channels: int = 16,
                 image_size: int = 128,
                 labels: Optional[List] = None,
                 dataset_name: str = "unknown",
                 preserve_channels: bool = True):
        """
        Args:
            image_paths: List of image file paths or a Dataset instance (e.g., torchvision MNIST)
            max_channels: Maximum number of channels (for reference)
            image_size: Target image size
            labels: Optional labels for classification (overrides dataset labels if provided)
            dataset_name: Name for tracking dataset origin
            preserve_channels: If True, keep original channels (padding done in collate)
        """
        # Accept both lists of paths and Dataset objects
        self.is_dataset = isinstance(image_paths, Dataset)
        if self.is_dataset:
            self.dataset = image_paths
            self.image_paths = None
        else:
            self.dataset = None
            self.image_paths = image_paths

        self.max_channels = max_channels
        self.image_size = image_size
        self.labels = labels
        self.dataset_name = dataset_name
        self.preserve_channels = preserve_channels

        # Normalization to [-1, 1] for Tanh output (applied to PIL images via transform)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Maps [0,1] to [-1,1]
        ])

    def __len__(self):
        if self.is_dataset:
            return len(self.dataset)
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Retrieve raw item from either stored paths or wrapped dataset
        ds_label = None
        if self.is_dataset:
            item = self.dataset[idx]
            if isinstance(item, tuple) or isinstance(item, list):
                img = item[0]
                if len(item) > 1:
                    ds_label = item[1]
            else:
                img = item
        else:
            img_path = self.image_paths[idx]
            img = self._load_image(img_path)

        # If `img` is a path-like string (some datasets may return paths), load it
        if isinstance(img, str):
            img = self._load_image(img)

        # Convert to tensor and normalize
        img_tensor: torch.Tensor
        if isinstance(img, np.ndarray):
            # Convert HxW or HxWxC numpy -> CHW torch tensor
            if img.ndim == 2:
                img_tensor = torch.from_numpy(img).float().unsqueeze(0)
            elif img.ndim == 3:
                # If channels are last, move to first
                if img.shape[2] <= img.shape[0]:  # heuristic: probably HWC
                    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
                else:
                    img_tensor = torch.from_numpy(img).float()
            else:
                raise ValueError("Unsupported numpy image shape: {}".format(img.shape))

            # If likely in 0-255 range, scale to [0,1]
            if img_tensor.max() > 1.5:
                img_tensor = img_tensor / 255.0

            # Normalize to [-1, 1]
            img_tensor = (img_tensor - 0.5) / 0.5

        elif isinstance(img, torch.Tensor):
            img_tensor = img.float()
            # Ensure CHW
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.ndim == 3 and img_tensor.shape[0] not in (1, 3) and img_tensor.shape[2] in (1, 3):
                # likely HWC -> convert to CHW
                img_tensor = img_tensor.permute(2, 0, 1)

            # If values look like 0-255, scale
            if img_tensor.max() > 1.5:
                img_tensor = img_tensor / 255.0

            # Normalize to [-1,1]
            img_tensor = (img_tensor - 0.5) / 0.5

        else:
            # PIL Image (common for torchvision datasets without transforms)
            img_tensor = self.transform(img)

        # Resize if needed (ensure CxHxW)
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)

        if img_tensor.shape[-2:] != (self.image_size, self.image_size):
            # Ensure batch dimension for interpolate: (N, C, H, W)
            tmp = img_tensor.unsqueeze(0)
            # If single-channel but using bilinear, interpolate accepts it
            img_tensor = F.interpolate(tmp, size=(self.image_size, self.image_size), mode='bilinear').squeeze(0)

        # ONLY pad if preserve_channels is False
        # Otherwise, collate_fn will handle padding
        if not self.preserve_channels:
            current_channels = img_tensor.shape[0]
            if current_channels < self.max_channels:
                padding = torch.zeros(
                    self.max_channels - current_channels,
                    img_tensor.shape[1],
                    img_tensor.shape[2]
                )
                img_tensor = torch.cat([img_tensor, padding], dim=0)
            elif current_channels > self.max_channels:
                img_tensor = img_tensor[:self.max_channels]

        # Determine returned label
        if self.labels is not None:
            return img_tensor, self.labels[idx]
        if ds_label is not None:
            return img_tensor, ds_label
        return img_tensor

    def _load_image(self, path: str):
        """Load image handling various formats"""
        try:
            # Try PIL first (handles most common formats)
            img = Image.open(path)
            if img.mode == 'L':  # Grayscale
                return img
            elif img.mode == 'RGB':
                return img
            else:
                return img.convert('RGB')
        except Exception:
            # Try OpenCV for multispectral/special formats
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
            raise ValueError(f"Could not load image: {path}")

