import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = self._get_img_labels()
        self.label_encoder = self._encode_labels()

    def _get_img_labels(self):
        img_labels = []
        for label in os.listdir(self.img_dir):
            class_dir = os.path.join(self.img_dir, label)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        img_labels.append((os.path.join(class_dir, filename), label))
        return img_labels

    def _encode_labels(self):
        labels = [label for _, label in self.img_labels]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        return label_encoder
    
    def get_label_mapping(self):
        return dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.label_encoder.transform([label])[0]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def _show_encoded_labels(self):
        unique_labels = list(set([label for _, label in self.img_labels])) # Get unique labels
        unique_labels.sort() # Sort labels alphabetically for consistent output
        encoded_labels = {} # Create a dictionary to store the results
        for label in unique_labels:
            encoded_value = self.label_encoder.transform([label])[0]
            encoded_labels[label] = encoded_value
        # Print the results
        print("Label Encoding Results:")
        print("----------------------")
        for label, encoded_value in encoded_labels.items():
            print(f"Original Label: {label:<20} Encoded Label: {encoded_value}")
        return encoded_labels