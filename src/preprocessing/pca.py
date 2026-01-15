import math

import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.dataset.DataManager import DataManager

# batch_size = 256
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.view(-1))  # flatten 28×28 → 784
# ])
#
# train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# print(train_loader)
# print(test_loader)
# print(train_dataset[0][0].shape)


def normalize_to_angles(X):  # normalizes from [0,1] to [0, π]
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    return X * math.pi


def transform_to_pca_loader(
    train_loader, val_loader, test_loader, batch_size=32, n_components=20
):
    # Stack all training images into a single tensor
    train_features = []
    train_labels = []
    for x, y in train_loader:
        x = x.view(x.shape[0], -1)  # flatten the image
        train_features.append(x)
        train_labels.append(y)
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    train_features_np = train_features.cpu().numpy()
    pca.fit(train_features_np)

    # Transform all datasets in one vectorized pass
    train_pca = torch.tensor(pca.transform(train_features_np), dtype=torch.float32)

    # Stack and transform validation data
    val_features = torch.cat([x.view(x.shape[0], -1) for x, y in val_loader], dim=0)
    val_labels = torch.cat([y for x, y in val_loader], dim=0)
    val_pca = torch.tensor(
        pca.transform(val_features.cpu().numpy()), dtype=torch.float32
    )

    # Stack and transform test data
    test_features = torch.cat([x.view(x.shape[0], -1) for x, y in test_loader], dim=0)
    test_labels = torch.cat([y for x, y in test_loader], dim=0)
    test_pca = torch.tensor(
        pca.transform(test_features.cpu().numpy()), dtype=torch.float32
    )

    # Normalize all at once
    for pca_tensor in [train_pca, val_pca, test_pca]:
        pca_min = pca_tensor.min()
        pca_max = pca_tensor.max()
        pca_tensor.sub_(pca_min).div_(pca_max - pca_min + 1e-8).mul_(math.pi)

    train_pca_loader = DataLoader(
        TensorDataset(train_pca, train_labels),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_pca_loader = DataLoader(
        TensorDataset(val_pca, val_labels),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_pca_loader = DataLoader(
        TensorDataset(test_pca, test_labels),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_pca_loader, val_pca_loader, test_pca_loader


if __name__ == "__main__":
    dm = DataManager(batch_size=256, seed=42, dataset="mnist", pixel_size=28)
    train_loader, val_loader, test_loader = dm.get_loaders(0.8, 0.1, 0.1)
    pca_train, pca_val, pca_test = transform_to_pca_loader(
        train_loader, val_loader, test_loader
    )
    print(pca_train[0][0].shape)
    print(pca_val[0][0].shape)
    print(pca_test[0][0].shape)
