
"""
Optimized Unimodal Dataset for pre-resized images with parallel GPU loading.
Uses pre-processed 224x224 images to skip resize during training.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class UnimodalDatasetOptimized(Dataset):
    """
    Optimized dataset for pre-resized images (224x224).
    Loads images without resize to maximize throughput.
    """
    def __init__(self, dataset_root, modality='hand', transform=None):
        """
        Args:
            dataset_root: Root directory of the pre-resized dataset.
            modality: 'hand' or 'iris'.
            transform: Optional transforms (should NOT include Resize).
        """
        self.transform = transform
        
        if modality == 'hand':
            self.data_root = os.path.join(dataset_root, "hand dataset")
        elif modality == 'iris':
            self.data_root = os.path.join(dataset_root, "iris dataset")
        else:
            raise ValueError("Modality must be 'hand' or 'iris'")
        
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Dataset not found at {self.data_root}. Run preprocess_gpu.py first!")
        
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
            # Images are already 224x224, load directly
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return torch.zeros(3, 224, 224), label

        if self.transform:
            img = self.transform(img)
            
        return img, label
