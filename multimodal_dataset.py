
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MULBDataset(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        
        self.hand_root = os.path.join(dataset_root, "hand dataset")
        self.iris_root = os.path.join(dataset_root, "iris dataset")
        
        self.classes = sorted([d for d in os.listdir(self.hand_root) if os.path.isdir(os.path.join(self.hand_root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for subject in self.classes:
            hand_dir = os.path.join(self.hand_root, subject)
            iris_dir = os.path.join(self.iris_root, subject)
            
            if not os.path.exists(iris_dir):
                continue
                
            hand_images = sorted([os.path.join(hand_dir, f) for f in os.listdir(hand_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            iris_images = sorted([os.path.join(iris_dir, f) for f in os.listdir(iris_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Pair 1-to-1 up to the length of the shorter list
            min_len = min(len(hand_images), len(iris_images))
            
            for i in range(min_len):
                self.samples.append((hand_images[i], iris_images[i], self.class_to_idx[subject]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hand_path, iris_path, label = self.samples[idx]
        
        try:
            hand_img = Image.open(hand_path).convert('RGB')
            iris_img = Image.open(iris_path).convert('RGB')
        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}")
            # Return a dummy or handle gracefully? For now, let's just retry or return the next one (simple hack)
            # A better way is to filter valid images beforehand, but let's assume they are mostly valid.
            # We'll just return zeros if it fails to avoid crashing, but log it.
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), label

        if self.transform:
            hand_img = self.transform(hand_img)
            iris_img = self.transform(iris_img)
            
        return hand_img, iris_img, label
