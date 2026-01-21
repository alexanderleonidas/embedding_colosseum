import os
import random
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from src.dataset.brain_tumor import BRAINTUMOR, extract_brain_tumor_dataset
from src.dataset.cxr8 import CXR8, extract_chest_xray_dataset
from src.dataset.eurosat import EUROSAT, extract_eurosat_dataset
from src.preprocessing.denoising_filters import BilateralFilter, MedianBlur
from src.preprocessing.homomorphic_filter import HomomorphicFilter

try:
    import ssl

    import certifi

    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except Exception as _:
    # If certifi isn't available or something fails, keep default context and warn
    import warnings

    warnings.warn(
        "Could not set certifi SSL context; HTTPS downloads may fail. Install certifi with: pip install certifi"
    )


class DataManager:
    """
    Data Manager for loading a specified dataset.
    """

    def __init__(
        self,
        cfg: dict,
        batch_size: int,
        seed: int,
        pixel_size: int,
        dataset="mnist",
        transform=None,
        make_binary=False,
    ):
        self.cfg = cfg
        self.make_binary = make_binary
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        tf_list = [transforms.ToTensor()]
        if cfg is not None:
            # Resizing and padding not needed for pca based methods, taking original size
            if not cfg.embedding.pca:
                if cfg.dataset.orig_width - cfg.training.image_width < 0:
                    # Pad as the target size is bigger than original
                    pad = cfg.training.image_width - cfg.dataset.orig_width
                    pad //= 2
                    tf_list += [transforms.Pad(pad)]
                else:
                    tf_list += [
                        transforms.Resize(
                            (cfg.training.image_width, cfg.training.image_width)
                        )
                    ]

            if not cfg.embedding.supports_color:
                tf_list.append(transforms.Grayscale(num_output_channels=1))
        elif cfg is None and pixel_size is not None:
            tf_list += [transforms.Resize((pixel_size, pixel_size))]

        if transform is None or transform == "None":
            pass
        elif transform == "GaussianBlur":
            tf_list += [
                transforms.GaussianBlur(kernel_size=3, sigma=1.0),
            ]
        elif transform == "ContrastScaling":
            tf_list += [
                lambda x: self._contrast_scale_image_preproc(
                    x
                ),  # contrast scaling transformation as lamba expression in Compose
            ]

        elif transform == "MedianBlur":
            tf_list += [
                MedianBlur(kernel_size=3),
            ]
        elif transform == "BilateralFilter":
            tf_list += [
                BilateralFilter(d=9, sigma_color=75, sigma_space=75),
            ]
        elif transform == "HomomorphicFilter":
            tf_list += [
                transforms.Grayscale(num_output_channels=1),
                HomomorphicFilter(a=0.5, b=1.5, cutoff=32),
            ]
        else:
            raise ValueError("Unsupported transform (image preprocessing) method.\n")

        self.transform = transforms.Compose(tf_list)
        self.root = self._get_root()
        self._data = self._get_dataset(dataset)

    def _contrast_scale_image_preproc(self, x, factor=1.5):
        mean = x.mean()
        return torch.clamp(mean + factor * (x - mean), 0.0, 1.0)

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
            all_data = ConcatDataset([full_train, test_ds])

        elif dataset == "fashion":
            full_train = datasets.FashionMNIST(
                root=self.root, train=True, download=True, transform=self.transform
            )
            test_ds = datasets.FashionMNIST(
                root=self.root, train=False, download=True, transform=self.transform
            )
            all_data = ConcatDataset([full_train, test_ds])

        elif dataset == "cifar10":
            full_train = datasets.CIFAR10(
                root=self.root, train=True, download=True, transform=self.transform
            )
            test_ds = datasets.CIFAR10(
                root=self.root, train=False, download=True, transform=self.transform
            )
            all_data = ConcatDataset([full_train, test_ds])

        elif dataset == "stl10":
            full_train = datasets.STL10(
                root=self.root, split="train", download=True, transform=self.transform
            )
            test_ds = datasets.STL10(
                root=self.root, split="test", download=True, transform=self.transform
            )
            all_data = ConcatDataset([full_train, test_ds])

        elif dataset == "cxr8":
            img_paths, labels = extract_chest_xray_dataset(self.root, all_folders=True)
            all_data = CXR8(img_paths, labels, transform=self.transform)
        elif dataset == "brain_tumor":
            img_paths, labels = extract_brain_tumor_dataset(self.root)
            all_data = BRAINTUMOR(img_paths, labels, transform=self.transform)
        elif dataset == "eurosat_ms":
            img_paths, labels = extract_eurosat_dataset(self.root, rgb=False)
            all_data = EUROSAT(img_paths, labels, transform=self.transform)
        elif dataset == "eurosat_rgb":
            img_paths, labels = extract_eurosat_dataset(self.root, rgb=True)
            all_data = EUROSAT(img_paths, labels, transform=self.transform)
        else:
            raise ValueError(
                "Unsupported dataset. Choose 'mnist{_binary}', 'fashion', 'cifar10', 'stl10', 'cxr8{_binary}', 'brain_tumor' or 'eurosat_{rgb,ms}'."
            )

        if self.make_binary and dataset != "brain_tumor":
            class_a = 0  # CHOOSE HERE classes for binary classification
            class_b = 1
            all_data = self.make_binary_dataset(all_data, class_a, class_b)

        return all_data

    class _BinaryDataset(Dataset):
        def __init__(self, dataset, class_a, class_b, neg=0, pos=1):
            self.dataset = dataset
            self.class_a = class_a
            self.class_b = class_b
            self.neg = neg
            self.pos = pos

            # Try to obtain labels without calling __getitem__ for every sample
            labels = None
            if hasattr(dataset, "targets"):
                labels = dataset.targets
            elif hasattr(dataset, "labels"):
                labels = dataset.labels
            elif hasattr(dataset, "y"):
                labels = dataset.y

            if labels is not None:
                # Normalise to a Python list for fast iteration
                try:
                    labels = list(labels)
                except Exception:
                    labels = [int(l) for l in labels]

                # Get indices for both classes
                indices_a = [i for i, y in enumerate(labels) if y == class_a]
                indices_b = [i for i, y in enumerate(labels) if y == class_b]
            else:
                # Fallback: single pass that calls __getitem__ (slower)
                indices_a = []
                indices_b = []
                for i in range(len(dataset)):
                    x, y = dataset[i]
                    if y == class_a:
                        indices_a.append(i)
                    elif y == class_b:
                        indices_b.append(i)

            # Balance the dataset by taking the minimum number of samples
            min_samples = min(len(indices_a), len(indices_b))

            # Randomly sample from both classes
            random.shuffle(indices_a)
            random.shuffle(indices_b)

            self.indices = indices_a[:min_samples] + indices_b[:min_samples]

            # Shuffle the combined indices
            random.shuffle(self.indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            x, y = self.dataset[self.indices[idx]]
            y = self.neg if y == self.class_a else self.pos
            return x, y

    def make_binary_dataset(
            self, dataset, class_a, class_b, negative_label=0, positive_label=1
    ):
        """
        Convert ANY dataset (including ConcatDataset) into a balanced binary dataset.
        Uses dataset attributes like `targets` or `labels` when available to avoid
        expensive per-item __getitem__ calls.
        """
        if isinstance(dataset, ConcatDataset):
            return ConcatDataset(
                [
                    self.make_binary_dataset(
                        ds, class_a, class_b, negative_label, positive_label
                    )
                    for ds in dataset.datasets
                ]
            )

        return self._BinaryDataset(
            dataset, class_a, class_b, negative_label, positive_label
        )

    def get_dataset(self, num_points: int = None):
        """
        Get the dataset, optionally returning a random subset of `num_points`
        sampled without replacement (reproducible using the manager's generator).
        """
        if num_points is None:
            return self._data

        total = len(self._data)
        if num_points <= 0:
            raise ValueError("num_points must be a positive integer")
        if num_points > total:
            raise ValueError(f"num_points ({num_points}) greater than dataset size ({total})")

        indices = torch.randperm(total, generator=self.generator)[:num_points].tolist()
        return Subset(self._data, indices)

    def get_loaders(self, train_split: float, val_split: float, test_split: float):
        """
        Prepare DataLoaders for training and testing datasets.

        Args:
            train_split: Proportion of training data.
            val_split: Proportion of validation data.
            test_split: Proportion of test data.
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if (
            val_split < 0
            or val_split >= 1
            or train_split < 0
            or train_split > 1
            or test_split < 0
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
    #
    dataset_names = ["mnist", "fashion", "cifar10", "stl10", "cxr8", "brain_tumor", "eurosat_rgb"]
    all_datasets = {}
    for d in dataset_names:
        dm = DataManager(cfg=None, batch_size=100, seed=42, pixel_size=128, dataset=d)
        print(len(dm.get_dataset()))

    # sizes = set()
    # modes = set()
    # channels = set()
    # for d in train_loader:
    #     for img in d:
    #         with Image.open(d) as im:
    #             sizes.add(im.size)
    #             modes.add(im.mode)
    #             channels.add(len(im.getbands()))

            # if len(im.getbands()) == 4:
            #     print(f"RGB image found: {p}")
            #     img = Image.open(p).convert("RGB")
            #     img.show()
            #     break

    # print("Sizes:", sizes)
    # print("Modes:", modes)
    # print("Channels (counts):", channels)




    # dm = DataManager(
    #     cfg=None,
    #     batch_size=100,
    #     seed=42,
    #     dataset="cxr8",
    #     pixel_size=640,
    #     make_binary=True,
    # )
    # print(dm.root)
    # train, val, test = dm.get_loaders(0.8, 0.1, 0.1)
    # print(train.batch_size)
    # print(
    #     f"Train loader length: {len(train)}, Val loader length: {len(val)}, Test loader length: {len(test)}"
    # )
    # # img, label = train.dataset[0]
    # # print(img.size)
    # # print(label)
    # for img, label in train:
    #     print(img.size())
    #     print(label)
    #     break

