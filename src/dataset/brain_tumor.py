import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# The default size of the pictures in this dataset is 640x640 pixels


class BRAINTUMOR(Dataset):
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

def extract_brain_tumor_dataset():
    root = "./data/BRAINTUMOR"
    ct_directory = os.path.join(root, "Brain Tumor CT scan Images")
    mri_directory = os.path.join(root, "Brain Tumor MRI Images")
    classes = ['Healthy', 'Tumor']

    img_paths = []
    labels = []
    class_to_label = {'Healthy': 0, 'Tumor': 1}
    if os.path.isdir(ct_directory) and os.path.isdir(mri_directory):
        for class_name in classes:
            ct_class_path = os.path.join(ct_directory, class_name)
            mri_class_path = os.path.join(mri_directory, class_name)
            if os.path.isdir(ct_class_path) and os.path.isdir(mri_class_path):
                ct_img_files = [f for f in os.listdir(ct_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                mri_img_files = [f for f in os.listdir(mri_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for ct_img_name in ct_img_files:
                    img_paths.append(os.path.join(ct_class_path, ct_img_name))
                    labels.append(class_to_label[class_name])
                for mri_img_name in mri_img_files:
                    img_paths.append(os.path.join(mri_class_path, mri_img_name))
                    labels.append(class_to_label[class_name])

    # print(f"\nTotal Images: {len(img_paths)}")
    # print(f"Tumor (yes): {labels.count(1)} images ({labels.count(1) / len(labels) * 100:.1f}%)")
    # print(f"No Tumor (no): {labels.count(0)} images ({labels.count(0) / len(labels) * 100:.1f}%)")
    return img_paths, labels

if __name__ == "__main__":
    img_paths, labels = extract_brain_tumor_dataset()
    print(img_paths[:5])
    print(labels[:5])
    img = Image.open(img_paths[100])
    # img = transforms.ToPILImage()(img)
    print(img.size)
    img.show()
