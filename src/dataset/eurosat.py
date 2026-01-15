import os
from typing import List, Union

import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EUROSAT(Dataset):
    """
    Robust loader for EuroSAT images.
    - Reads .tif files with tifffile to avoid Pillow/libtiff decoding limits.
    - Keeps multispectral arrays as HxWxC numpy arrays (so transforms.ToTensor preserves channels).
    - Converts 3-channel data to PIL RGB for compatibility with existing transforms.
    """

    def __init__(self, img_paths: List[str], labels: List[int], transform=None):
        self.img_path = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _read_tif(self, path: str) -> Union[np.ndarray, Image.Image]:
        arr = tifffile.imread(path)  # possible shapes: (bands,H,W) or (H,W,bands) or (H,W)
        # Normalize shapes: if (bands, H, W) -> (H, W, bands)
        if arr.ndim == 3 and arr.shape[0] <= 32 and arr.shape[0] > arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = self._normalize_to_uint8(arr)
            return Image.fromarray(arr)  # grayscale PIL
        elif arr.ndim == 3:
            ch = arr.shape[2]
            arr = self._normalize_to_uint8(arr)
            if ch == 3:
                return Image.fromarray(arr)  # RGB PIL
            else:
                # preserve multispectral HxWxC numpy array
                return arr
        else:
            raise RuntimeError(f"Unsupported TIFF shape {arr.shape} for {path}")

    def __getitem__(self, idx):
        path = self.img_path[idx]
        label = int(self.labels[idx])

        # Prefer tifffile for .tif to avoid libtiff/Pillow warnings
        img_obj: Union[np.ndarray, Image.Image]
        if path.lower().endswith(".tif") or path.lower().endswith(".tiff"):
            img_obj = self._read_tif(path)
        else:
            # Non-tif: use Pillow as before
            pil_img = Image.open(path)
            pil_img.load()
            img_obj = pil_img

        # Apply transform (torchvision transforms accept PIL Image or HxWxC numpy array)
        if self.transform is not None:
            out = self.transform(img_obj)
        else:
            # Fallback: ensure tensor
            out = transforms.ToTensor()(img_obj)

        return out, label


def extract_eurosat_dataset(root, rgb=True):
    data_folder = os.path.join(root, "EuroSAT")
    if rgb:
        data_folder = os.path.join(data_folder, "EuroSAT_RGB")
    else:
        data_folder = os.path.join(data_folder, "EuroSAT_geo")
    classes = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]
    class_to_label = {name: i for i, name in enumerate(classes)}

    img_paths = []
    labels = []
    if os.path.isdir(data_folder):
        for class_name in classes:
            class_path = os.path.join(data_folder, class_name)
            if os.path.isdir(class_path):
                img_files = [
                    f
                    for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff"))
                ]
                for img_name in img_files:
                    img_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_to_label[class_name])

    return img_paths, labels


if __name__ == "__main__":
    img_paths, labels = extract_eurosat_dataset(root="./data", rgb=False)
    print(img_paths[:5])
    print(labels[:5])
    dataset = EUROSAT(img_paths, labels, transform=transforms.ToTensor())
    img, label = dataset[0]
    print(img.shape, label)