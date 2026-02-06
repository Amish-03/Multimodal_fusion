
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from multimodal_dataset import MULBDataset
from late_fusion_model import LateFusionModel

def evaluate_and_plot():
    # Config
    BATCH_SIZE = 64
    DATASET_ROOT = r"C:\Users\win10\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset_224"
    MODEL_PATH = "late_fusion_mulb_optimized.pth"
    OUTPUT_IMAGE = "confusion_matrix_late.png"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & Dataloader
    print("Loading dataset...")
    # No resize needed, just tensor & norm
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MULBDataset(DATASET_ROOT, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    num_classes = len(dataset.classes)
    
    print(f"Loaded {len(dataset)} samples across {num_classes} classes.")

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = LateFusionModel(num_classes=num_classes).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"Error: Model file {MODEL_PATH} not found!")
        return

    model.eval()

    # 3. Inference
    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for hand_imgs, iris_imgs, labels in tqdm(dataloader):
            hand_imgs = hand_imgs.to(device)
            iris_imgs = iris_imgs.to(device)
            
            outputs = model(hand_imgs, iris_imgs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc:.4f}")

    # 5. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    # Plotting
    plt.figure(figsize=(20, 16))
    
    # Since there are 188 classes, a full annotated heatmap might be too crowded text-wise.
    # We will plot it without numbers in cells if classes > 30, but keep the color map.
    annotate = num_classes <= 30
    
    sns.heatmap(cm, annot=annotate, fmt='d', cmap='Blues', 
                xticklabels=False, yticklabels=False) # Hide ticks for 188 classes to avoid clutter
    
    plt.title(f'Confusion Matrix (Accuracy: {acc:.2%})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Confusion matrix saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    evaluate_and_plot()
