import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class DataManager:
    """
    Data Manager for loading a specified dataset.
    """

    def __init__( self, batch_size: int, seed: int, dataset="mnist", pixel_size=None):
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        if pixel_size is None:
            transform = transforms.ToTensor()
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((pixel_size, pixel_size))])
        self.transform = transform

        self.dataset_name = dataset
        self.full_train, self.test_ds = self._get_dataset(dataset)

    def _get_dataset(self, dataset):
        if dataset == "mnist":
            full_train = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
            test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)
        elif dataset == "fashion":
            full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=self.transform)
            test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=self.transform)
        elif dataset == "cifar10":
            full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=self.transform)
            test_ds = datasets.CIFAR10(root="./data",train=False, download=True, transform=self.transform)
        elif dataset == "stl10":
            full_train = datasets.STL10(root="./data", split="train", download=True, transform=self.transform)
            test_ds = datasets.STL10(root="./data", split="test", download=True, transform=self.transform)
        elif dataset == "cxr8":
            raise NotImplemented("Dataset not supported yet.")
        elif dataset == "brain_tumor":
            raise NotImplemented("Dataset not supported yet.")
        elif dataset == "roco":
            raise NotImplemented("Dataset not supported yet.")
        else:
            raise ValueError("Unsupported dataset. Choose 'mnist', 'fashion', 'cifar10', 'stl10', 'cxr8', 'brain_tumor' or 'roco'.")

        return full_train, test_ds

    def get_loaders(self, val_split=None, train_size=None):
        """
        Prepare DataLoaders for training and testing datasets. If val_split is specified, it splits the training data.

        :return: Tuple of (train_loader, val_loader, test_loader) if val_split is specified, otherwise (train_loader, None, test_loader).
        """
        if train_size is not None:
            if isinstance(train_size, float) and 0 < train_size <= 1.0:
                train_size = int(len(self.full_train) * train_size)
            else:
                raise ValueError("train_size must be a float between 0 and 1 or a positive integer.")
            self.full_train, _ = random_split(self.full_train, [train_size, len(self.full_train) - train_size], generator=self.generator)

        if val_split is not None and val_split > 0:
            val_size = int(len(self.full_train) * val_split)
            train_size = len(self.full_train) - val_size
            train_ds, val_ds = random_split(self.full_train, [train_size, val_size])

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
            return train_loader, val_loader, test_loader
        else:
            train_loader = DataLoader(self.full_train, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
            return train_loader, None, test_loader


if __name__ == "__main__":
    Dataloader = DataManager(batch_size=100, seed=42, dataset="mnist", pixel_size=100)

    # print example image and its size
    tr, ts = Dataloader.get_loaders()
    img, _ = tr.dataset[0]
    figure = plt.figure(figsize=(20, 4))
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    print(img.shape)
