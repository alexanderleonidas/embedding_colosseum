import os
import logging

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# The default size of the pictures in this dataset is 1024x1024 pixels


class CXR8(Dataset):
    def __init__(self, img_paths, labels, transform):
        self.img_path = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert("L")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def extract_chest_xray_dataset(root, all_folders=False):
    data_folder1 = os.path.join(root, 'CXR8', 'images')
    data_folder2 = os.path.join(root, 'CXR8', 'images')
    csv_path = os.path.join(root,'CXR8','Data_Entry_2017_v2020.csv')
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
    if not os.path.isdir(data_folder1):
        print(f"\nImages folder not found: {data_folder1}")
        return img_paths, labels
    if not os.path.isdir(data_folder2) and all_folders:
        print(f"\nImages folder not found: {data_folder2}")
        return img_paths, labels

    # read the necessary columns only and build a lookup dict to avoid O(n^2) dataframe scans
    data_entry = pd.read_csv(csv_path, usecols=["Image Index", "Finding Labels"])
    mapping = dict(zip(data_entry["Image Index"], data_entry["Finding Labels"]))

    # faster directory listing with scandir
    with os.scandir(data_folder1) as it:
        files = [entry.name for entry in it if entry.is_file()]
    if all_folders:
        with os.scandir(data_folder2) as it:
                files += [entry.name for entry in it if entry.is_file()]
    files.sort()
    img_paths = [os.path.join(data_folder1, f) for f in files]
    if all_folders:
        img_paths += [os.path.join(data_folder2, f) for f in files]

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
    all_img_paths, labels = extract_chest_xray_dataset(root="./data", all_folders=True)
    print("Total images: ", len(all_img_paths))
    print(all_img_paths[:5])
    print(labels[:5])
    # imgs = [Image.open(img) for img in img_paths]
    # sizes = [img.size for img in imgs]
    # sizes = set(sizes)
    # print(sizes)
    sizes = set()
    modes = set()
    channels = set()
    for p in all_img_paths:
        with Image.open(p) as im:
            sizes.add(im.size)
            modes.add(im.mode)
            channels.add(len(im.getbands()))

            # if len(im.getbands()) == 4:
            #     print(f"RGB image found: {p}")
            #     img = Image.open(p).convert("RGB")
            #     img.show()
            #     break

    print("Sizes:", sizes)
    print("Modes:", modes)
    print("Channels (counts):", channels)

