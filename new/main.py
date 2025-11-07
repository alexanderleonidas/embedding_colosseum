import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class DataManager:
    """
    DataManager Data Manager for MNIST and EMNIST datasets.
    """
    def __init__(self, batch_size: int, seed: int, dataset='mnist', transform=None, pixel_size=None):
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        if transform == 'augmented':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif transform == 'noised':
            noise_level = 0.2
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.clamp(x + noise_level * torch.randn_like(x), 0.0, 1.0)),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        self.dataset_name = dataset
        self.full_train, self.test_ds = self._get_dataset(dataset)
        self.full_train.data = [self.resize_image(image, pixel_size) for image in self.full_train.data]
        self.test_ds.data = [self.resize_image(image, pixel_size) for image in self.test_ds.data]

    def _get_dataset(self, dataset):
        if dataset == 'mnist':
            full_train = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
            test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        elif dataset == 'fashion':
            full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=self.transform)
            test_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        else:
            raise ValueError("Unsupported dataset. Choose 'mnist', or 'fashion'.")

        return full_train, test_ds

    def resize_image(self, image, size):
        """
        Resize `image` (PIL Image, numpy array, or torch.Tensor) to `size` (int or (h, w)).
        Returns a flattened torch.FloatTensor normalized to [0, 1].
        """
        # Handle tensor input
        if isinstance(image, torch.Tensor):
            arr = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            arr = image
        elif isinstance(image, Image.Image):
            img = image
            arr = None
        else:
            raise TypeError("image must be a PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        # Convert the numpy array to PIL Image if needed
        if arr is not None:
            # If single-channel with extra dim, squeeze
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            # If floats in [0,1], convert to 0-255
            if not np.issubdtype(arr.dtype, np.uint8):
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            img = Image.fromarray(arr)

        # Normalize size: accept int or (h, w). PIL expects (width, height).
        if size is None:
            size_tuple = img.size  # already (width, height)
        elif isinstance(size, int):
            size_tuple = (size, size)
        else:
            h, w = size
            size_tuple = (w, h)

        # Resize with bilinear
        img_resized = img.resize(size_tuple, resample=Image.BILINEAR)

        arr_res = np.array(img_resized)
        # If RGB, convert to grayscale by averaging (MNIST is grayscale so this is just defensive)
        if arr_res.ndim == 3:
            arr_res = arr_res.mean(axis=2)

        # Normalize to float32 [0,1] and return as flattened torch tensor
        arr_res = arr_res.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr_res).float()
        return tensor.flatten()

    def get_loaders(self):
        """
        Prepare DataLoaders for training and testing datasets

        :return: Tuple of (train_loader, test_loader)
        """
        train_loader = DataLoader(self.full_train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader


Dataloader = DataManager(batch_size=100, seed=42, dataset='mnist', pixel_size=18)
train_loader, test_loader = Dataloader.get_loaders()
print("hello")