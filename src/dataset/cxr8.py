import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# The default size of the pictures in this dataset is 1024x1024 pixels


class CXR8(Dataset):
    def __init__(self, img_paths, labels, transform):
        self.img_path = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def extract_chest_xray_dataset():
    root = "./data/CXR8/images"
    csv_path = "./data/CXR8/Data_Entry_2017_v2020.csv"
    class_to_label = {
        "No Finding": 0,
        "Infiltration": 1,
        "Effusion": 2,
        "Atelectasis": 3,
        "Nodule": 4,
        "Mass": 5,
        "Pneumothorax": 6,
        "Consolidation": 7,
        "Pleural_Thickening": 8,
        "Cardiomegaly": 9,
        "Emphysema": 10,
        "Edema": 11,
        "Fibrosis": 12,
        "Pneumonia": 13,
        "Hernia": 14,
    }

    img_paths = []
    labels = []
    if not os.path.isdir(root):
        print(f"\nImages folder not found: {root}")
        return img_paths, labels

    # read the necessary columns only and build a lookup dict to avoid O(n^2) dataframe scans
    data_entry = pd.read_csv(csv_path, usecols=["Image Index", "Finding Labels"])
    mapping = dict(zip(data_entry["Image Index"], data_entry["Finding Labels"]))

    # faster directory listing with scandir
    with os.scandir(root) as it:
        files = [entry.name for entry in it if entry.is_file()]
    files.sort()
    img_paths = [os.path.join(root, f) for f in files]

    # lookup first label via dict (O(1) per file)
    first_labels = [mapping.get(f, "").split("|")[0] for f in files]


    labels = [class_to_label.get(lbl, -1) for lbl in first_labels]

    # print(f"\nTotal Images: {len(img_paths)}")
    # if binary:
    #     print(f"Classes: ['No Finding' (0), 'Finding' (1)]")
    #     print(f"No Finding: {labels.count(0)} images ({labels.count(0) / len(labels) * 100:.1f}%)")
    #     print(f"Finding: {labels.count(1)} images ({labels.count(1) / len(labels) * 100:.1f}%)")
    # else:
    #     for i in range(len(classes)):
    #         print(f"{classes[i]}: {labels.count(i)} images ({labels.count(i) / len(labels) * 100:.1f}%)")
    return img_paths, labels


if __name__ == "__main__":
    img_paths, labels = extract_chest_xray_dataset()
    print(img_paths[:5])
    print(labels[:5])
