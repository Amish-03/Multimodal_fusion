
import os
from torch.utils.data import Dataset
from PIL import Image

class UnimodalDataset(Dataset):
    """
    Dataset for a single biometric modality (either Hand or Iris).
    """
    def __init__(self, dataset_root, modality='hand', transform=None):
        """
        Args:
            dataset_root: Root directory of the MULB dataset.
            modality: 'hand' or 'iris'.
            transform: Optional transforms to apply.
        """
        self.transform = transform
        
        if modality == 'hand':
            self.data_root = os.path.join(dataset_root, "hand dataset")
        elif modality == 'iris':
            self.data_root = os.path.join(dataset_root, "iris dataset")
        else:
            raise ValueError("Modality must be 'hand' or 'iris'")
        
        self.classes = sorted([d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for subject in self.classes:
            subject_dir = os.path.join(self.data_root, subject)
            images = sorted([os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            for img_path in images:
                self.samples.append((img_path, self.class_to_idx[subject]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            import torch
            return torch.zeros(3, 224, 224), label

        if self.transform:
            img = self.transform(img)
            
        return img, label
