import os
from PIL import Image
from torch.utils.data import Dataset

# The default size of the pictures in this dataset is 64x64 pixels

class EUROSAT(Dataset):
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

def extract_eurosat_dataset(rgb=True):
    root = "./data/EuroSAT"
    if rgb:
        root = os.path.join(root, "EuroSAT_RGB")
    else:
        root = os.path.join(root, "EuroSAT_geo")
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    class_to_label= {'AnnualCrop':0, 'Forest':1, 'HerbaceousVegetation':2, 'Highway':3, 'Industrial':4, 'Pasture':5, 'PermanentCrop':6, 'Residential':7, 'River':8, 'SeaLake':9}

    img_paths = []
    labels = []
    if os.path.isdir(root):
        for class_name in classes:
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
                for img_name in img_files:
                    img_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_to_label[class_name])

    print(f"\nTotal Images: {len(img_paths)}")
    for i in range(len(classes)):
        print(f"{classes[i]}: {labels.count(i)} images ({labels.count(i) / len(labels) * 100:.1f}%)")

    return img_paths, labels

if __name__ == "__main__":
    img_paths, labels = extract_eurosat_dataset()
    print(img_paths[:5])
    print(labels[:5])
