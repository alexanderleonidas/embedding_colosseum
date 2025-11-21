import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from brain_tumor import extract_brain_tumor_dataset, BRAINTUMOR


class DataManager:
    """
    Data Manager for loading a specified dataset.
    """

    def __init__( self, batch_size: int, seed: int, pixel_size: int, dataset="mnist"):
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((pixel_size, pixel_size))])
        self._data = self._get_dataset(dataset)

    def _get_dataset(self, dataset):
        if dataset == "mnist":
            full_train = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
            test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)
            all_data = torch.cat((full_train.data, test_ds.data), dim=0)
        elif dataset == "fashion":
            full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=self.transform)
            test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=self.transform)
            all_data = torch.cat((full_train.data, test_ds.data), dim=0)
        elif dataset == "cifar10":
            full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=self.transform)
            test_ds = datasets.CIFAR10(root="./data",train=False, download=True, transform=self.transform)
            all_data = torch.cat((full_train.data, test_ds.data), dim=0)
        elif dataset == "stl10":
            full_train = datasets.STL10(root="./data", split="train", download=True, transform=self.transform)
            test_ds = datasets.STL10(root="./data", split="test", download=True, transform=self.transform)
            all_data = torch.cat((full_train.data, test_ds.data), dim=0)
        elif dataset == "cxr8":
            raise NotImplemented("Dataset not supported yet.")
        elif dataset == "brain_tumor":
            img_paths, labels = extract_brain_tumor_dataset()
            all_data = BRAINTUMOR(img_paths, labels, transform=self.transform)
        elif dataset == "eurosat":
            raise NotImplemented("Dataset not supported yet.")
        else:
            raise ValueError("Unsupported dataset. Choose 'mnist', 'fashion', 'cifar10', 'stl10', 'cxr8', 'brain_tumor_{ct, mri}' or 'eurosat'.")

        return all_data

    def get_loaders(self, train_split: float, val_split: float, test_split: float):
        """
        Prepare DataLoaders for training and testing datasets.

        :param train_split: Proportion of training data.
        :type train_split: float
        :param val_split: Proportion of validation data.
        :type val_split: float
        :param test_split: Proportion of test data.
        :type test_split: float
        :return: Tuple of (train_loader, val_loader, test_loader)
        """
        if val_split <= 0 or val_split >= 1 or train_split <= 0 or train_split >= 1 or test_split <= 0 or test_split >= 1:
            raise ValueError("Invalid split values. Values must be between 0 and 1.")
        if val_split + train_split + test_split != 1:
            raise ValueError("Sum of split values must be equal to 1.")

        test_size = int(len(self._data) * test_split)
        train_size = int(len(self._data) * train_split)
        val_size = len(self._data) - train_size - test_size
        train_ds, val_ds, test_ds = random_split(self._data, [train_size, val_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader



# Example usage
if __name__ == "__main__":
    Dataloader = DataManager(batch_size=100, seed=42, dataset="brain_tumor", pixel_size=512)
    train, val, test = Dataloader.get_loaders(0.8,0.1,0.1)
    print(f"Train loader length: {len(train)}, Val loader length: {len(val)}, Test loader length: {len(test)}")
    img, label = train.dataset[0]
    print(img.shape)
    img = transforms.ToPILImage()(img)
    img.show()