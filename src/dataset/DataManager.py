import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.dataset.brain_tumor import BRAINTUMOR, extract_brain_tumor_dataset
from src.dataset.cxr8 import CXR8, extract_chest_xray_dataset
from src.dataset.eurosat import EUROSAT, extract_eurosat_dataset


class DataManager:
    """
    Data Manager for loading a specified dataset.
    """

    def __init__(
        self,
        batch_size: int,
        seed: int,
        pixel_size: int,
        dataset="mnist",
        transform="normalise",
    ):
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        if transform == "greyscale":
            transform = transforms.Compose(
                [transforms.Grayscale(), transforms.Resize((pixel_size, pixel_size))]
            )
        elif transform == "normalise":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((pixel_size, pixel_size))]
            )
        else:
            raise ValueError(
                "Unsupported transform. Choose 'normalise' or 'greyscale'."
            )
        self.transform = transform
        self.root = self._get_root()
        self._data = self._get_dataset(dataset)

    def _get_root(self):
        root = os.getcwd()
        root = root.split("src/")[0]
        root = os.path.join(root, "src/dataset/data")
        return root

    def _get_dataset(self, dataset):
        if dataset == "mnist":
            full_train = datasets.MNIST(
                root=self.root, train=True, download=True, transform=self.transform
            )
            test_ds = datasets.MNIST(
                root=self.root, train=False, download=True, transform=self.transform
            )
            all_data = torch.utils.data.ConcatDataset([full_train, test_ds])

        elif dataset == "mnist_binary":
            full_train = datasets.MNIST(
                root=self.root, train=True, download=True, transform=self.transform
            )
            test_ds = datasets.MNIST(
                root=self.root, train=False, download=True, transform=self.transform
            )

            # Select digits for binary classification
            DIGIT_A = 0  # <-- choose here
            DIGIT_B = 1

            train_mask = (full_train.targets == DIGIT_A) | (
                full_train.targets == DIGIT_B
            )
            test_mask = (test_ds.targets == DIGIT_A) | (test_ds.targets == DIGIT_B)

            # Filtered dataset for selected digits
            full_train.data = full_train.data[train_mask]
            full_train.targets = full_train.targets[train_mask]
            test_ds.data = test_ds.data[test_mask]
            test_ds.targets = test_ds.targets[test_mask]

            # and labels mapped - DIGIT_A → -1, DIGIT_B → +1
            full_train.targets = torch.where(full_train.targets == DIGIT_A, -1.0, 1.0)
            test_ds.targets = torch.where(test_ds.targets == DIGIT_A, -1.0, 1.0)

            all_data = full_train

        elif dataset == "fashion":
            full_train = datasets.FashionMNIST(
                root=self.root, train=True, download=True, transform=self.transform
            )
            test_ds = datasets.FashionMNIST(
                root=self.root, train=False, download=True, transform=self.transform
            )
            all_data = torch.utils.data.ConcatDataset([full_train, test_ds])

        elif dataset == "cifar10":
            full_train = datasets.CIFAR10(
                root=self.root, train=True, download=True, transform=self.transform
            )
            test_ds = datasets.CIFAR10(
                root=self.root, train=False, download=True, transform=self.transform
            )
            all_data = torch.utils.data.ConcatDataset([full_train, test_ds])

        elif dataset == "stl10":
            full_train = datasets.STL10(
                root=self.root, split="train", download=True, transform=self.transform
            )
            test_ds = datasets.STL10(
                root=self.root, split="test", download=True, transform=self.transform
            )
            all_data = torch.utils.data.ConcatDataset([full_train, test_ds])

        elif dataset == "cxr8":
            img_paths, labels = extract_chest_xray_dataset()
            all_data = CXR8(img_paths, labels, transform=self.transform)
        elif dataset == "cxr8_binary":
            img_paths, labels = extract_chest_xray_dataset(binary=True)
            all_data = CXR8(img_paths, labels, transform=self.transform)
        elif dataset == "brain_tumor":
            img_paths, labels = extract_brain_tumor_dataset()
            all_data = BRAINTUMOR(img_paths, labels, transform=self.transform)
        elif dataset == "eurosat_ms":
            img_paths, labels = extract_eurosat_dataset(rgb=False)
            all_data = EUROSAT(img_paths, labels, transform=self.transform)
        elif dataset == "eurosat_rgb":
            img_paths, labels = extract_eurosat_dataset(rgb=True)
            all_data = EUROSAT(img_paths, labels, transform=self.transform)
        else:
            raise ValueError(
                "Unsupported dataset. Choose 'mnist{_binary}', 'fashion', 'cifar10', 'stl10', 'cxr8{_binary}', 'brain_tumor' or 'eurosat_{rgb,ms}'."
            )

        return all_data

    def get_loaders(self, train_split: float, val_split: float, test_split: float):
        """
        Prepare DataLoaders for training and testing datasets.

        :param train_split: Proportion of training data.
        :param val_split: Proportion of validation data.
        :param test_split: Proportion of test data.
        :return: Tuple of (train_loader, val_loader, test_loader)
        """
        if (
            val_split <= 0
            or val_split >= 1
            or train_split <= 0
            or train_split >= 1
            or test_split <= 0
            or test_split >= 1
        ):
            raise ValueError("Invalid split values. Values must be between 0 and 1.")
        if abs((train_split + val_split + test_split) - 1.0) > 1e-8:
            raise ValueError("Sum of split values must be equal to 1.")

        test_size = int(len(self._data) * test_split)
        train_size = int(len(self._data) * train_split)
        val_size = len(self._data) - train_size - test_size
        train_ds, val_ds, test_ds = random_split(
            self._data, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    dm = DataManager(batch_size=100, seed=42, dataset="eurosat_rgb", pixel_size=64)
    print(dm.root)
    # train, val, test = dm.get_loaders(0.8, 0.1, 0.1)
    # print(
    #     f"Train loader length: {len(train)}, Val loader length: {len(val)}, Test loader length: {len(test)}"
    # )
    # img, label = train.dataset[0]
    # print(img.shape)
    # print(label)
    # img = transforms.ToPILImage()(img)
    # img.show()
