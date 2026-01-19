import os

from PIL import Image
from torch.utils.data import Dataset

# The default size of the pictures in this dataset is 640x640 pixels


class BRAINTUMOR(Dataset):
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


def extract_brain_tumor_dataset(root):
    data_folder = os.path.join(root, "BRAINTUMOR")
    ct_directory = os.path.join(data_folder, "Brain Tumor CT scan Images")
    mri_directory = os.path.join(data_folder, "Brain Tumor MRI images")
    classes = ["Healthy", "Tumor"]

    img_paths = []
    labels = []
    class_to_label = {"Healthy": 1, "Tumor": -1}
    if os.path.isdir(ct_directory) and os.path.isdir(mri_directory):
        for class_name in classes:
            ct_class_path = os.path.join(ct_directory, class_name)
            mri_class_path = os.path.join(mri_directory, class_name)
            if os.path.isdir(ct_class_path) and os.path.isdir(mri_class_path):
                ct_img_files = [
                    f
                    for f in os.listdir(ct_class_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                mri_img_files = [
                    f
                    for f in os.listdir(mri_class_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                for ct_img_name in ct_img_files:
                    img_paths.append(os.path.join(ct_class_path, ct_img_name))
                    labels.append(class_to_label[class_name])
                for mri_img_name in mri_img_files:
                    img_paths.append(os.path.join(mri_class_path, mri_img_name))
                    labels.append(class_to_label[class_name])
    return img_paths, labels


# Example usage
if __name__ == "__main__":
    img_paths, labels = extract_brain_tumor_dataset(root="./data")
    print(len(img_paths))
    print(img_paths[:5])
    print(labels[:5])
    img = Image.open(img_paths[100])
    print(img.size)
    # img = transforms.ToPILImage()(img)
    # img.show()
