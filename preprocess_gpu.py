
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Config
ORIGINAL_ROOT = r"C:\Users\win10\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset"
TARGET_ROOT = r"C:\Users\win10\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset_224"
BATCH_SIZE = 64
IMG_SIZE = 224

class ImageFileDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
        
        # Walk and find all images
        print(f"Scanning {root_dir}...")
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(dirpath, f)
                    rel_path = os.path.relpath(full_path, root_dir)
                    self.files.append((full_path, rel_path))
        print(f"Found {len(self.files)} images.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        full_path, rel_path = self.files[idx]
        try:
            # Load as RGB
            img = Image.open(full_path).convert('RGB')
            # Resize and ToTensor on CPU to ensure consistent size for batching
            img_tensor = self.transform(img)
            return img_tensor, rel_path
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), "ERROR"

def main():
    if not os.path.exists(TARGET_ROOT):
        os.makedirs(TARGET_ROOT)

    dataset = ImageFileDataset(ORIGINAL_ROOT)
    # increased workers for file IO
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8) 
    
    print("Starting processing (Resizing on CPU, Saving)...")
    
    for batch_imgs, batch_rel_paths in tqdm(dataloader):
        # batch_imgs is already [B, 3, 224, 224]
        
        # Iterate through batch and save
        for i in range(len(batch_imgs)):
            rel_path = batch_rel_paths[i]
            if rel_path == "ERROR":
                continue
                
            save_path = os.path.join(TARGET_ROOT, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Convert back to PIL
            save_img = transforms.ToPILImage()(batch_imgs[i])
            save_img.save(save_path, quality=95)

    print("Preprocessing Complete.")
    print(f"New dataset location: {TARGET_ROOT}")

if __name__ == "__main__":
    main()
