import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from sklearn.decomposition import PCA
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

def transform_to_pca_loader(train_loader, val_loader, test_loader, n_components=20):
    batch_size = int(len(train_loader)/len(train_loader.dataset))
    # Stack all training images into a single tensor
    train_features = []
    train_labels = []
    for x, y in train_loader:
        train_features.append(x) # flatten the image
        train_labels.append(y)
    train_features = torch.cat(train_features, dim=0)  # shape (60000, 784)
    train_labels = torch.cat(train_labels, dim=0)

    pca = PCA(n_components=n_components)
    pca.fit(train_features.numpy())
    train_pca = pca.transform(train_features.numpy())    # np array (60000, n_components)
    train_pca = torch.tensor(train_pca, dtype=torch.float32)

    # Stack validation data
    val_features = []
    val_labels = []
    for x, y in val_loader:
        val_features.append(x)
        val_labels.append(y)

    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    test_features = []
    test_labels = []
    for x, y in test_loader:
        test_features.append(x)
        test_labels.append(y)

    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    val_pca = pca.transform(val_features.numpy())
    val_pca = torch.tensor(val_pca, dtype=torch.float32)
    test_pca = pca.transform(test_features.numpy())
    test_pca = torch.tensor(test_pca, dtype=torch.float32)

    train_pca_dataset = TensorDataset(train_pca, train_labels)
    val_pca_dataset = TensorDataset(val_pca, val_labels)
    test_pca_dataset = TensorDataset(test_pca, test_labels)

    train_pca_loader = DataLoader(train_pca_dataset, batch_size=batch_size, shuffle=True)
    val_pca_loader = DataLoader(val_pca_dataset, batch_size=batch_size, shuffle=False)
    test_pca_loader = DataLoader(test_pca_dataset, batch_size=batch_size, shuffle=False)

    print(train_pca_loader)
    print(test_pca_loader)
    return train_pca_loader, val_pca_loader, test_pca_loader


if __name__ == "__main__":
    dm = DataManager(batch_size=256, seed=42, dataset="mnist", pixel_size=28)
    train_loader, val_loader, test_loader = dm.get_loaders(0.8, 0.1, 0.1)
    pca_train, pca_val, pca_test = transform_to_pca_loader(train_loader, val_loader, test_loader)
    print(pca_train[0][0].shape)
    print(pca_val[0][0].shape)
    print(pca_test[0][0].shape)

