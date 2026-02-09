
"""
Generate confusion matrices for unimodal (Hand-only and Iris-only) models.
Creates individual PNG files for each modality.
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

from unimodal_dataset_optimized import UnimodalDatasetOptimized
from unimodal_model import UnimodalModel


def evaluate_and_plot(modality, model_path, output_image, dataset_root):
    """Generate confusion matrix for a single modality."""
    
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Generating Confusion Matrix: {modality.upper()}")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    # Dataset & Dataloader (no resize needed for pre-processed images)
    print(f"Loading {modality} dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = UnimodalDatasetOptimized(dataset_root, modality=modality, transform=transform)
    num_classes = len(full_dataset.classes)
    
    # Use same split as training (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size], 
                                    generator=torch.Generator().manual_seed(42))
    
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Test samples: {len(test_dataset)}, Classes: {num_classes}")

    # Load Model
    print(f"Loading model from {model_path}...")
    model = UnimodalModel(num_classes=num_classes).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print(f"Error: Model file {model_path} not found!")
        return None

    model.eval()

    # Inference
    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    # Plotting
    plt.figure(figsize=(20, 16))
    
    # For 188 classes, don't annotate cells
    annotate = num_classes <= 30
    
    # Create colorful visualization
    sns.heatmap(cm, annot=annotate, fmt='d', cmap='Blues', 
                xticklabels=False, yticklabels=False)
    
    plt.title(f'{modality.upper()} Model Confusion Matrix\n(Test Accuracy: {acc:.2%})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add accuracy annotation
    plt.figtext(0.5, 0.02, f'Total Classes: {num_classes} | Test Samples: {len(test_dataset)}', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_image}")
    return acc


def main():
    parser = argparse.ArgumentParser(description='Generate Unimodal Confusion Matrices')
    parser.add_argument('--modality', type=str, choices=['hand', 'iris', 'both'], 
                        default='both', help='Which modality to evaluate')
    args = parser.parse_args()
    
    # Paths
    DATASET_ROOT = r"C:\Users\win10\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset_224"
    
    results = {}
    modalities = ['hand', 'iris'] if args.modality == 'both' else [args.modality]
    
    for mod in modalities:
        model_path = f"{mod}_only_optimized.pth"
        output_image = f"confusion_matrix_{mod}.png"
        
        acc = evaluate_and_plot(mod, model_path, output_image, DATASET_ROOT)
        if acc is not None:
            results[mod] = acc
    
    # Summary
    print("\n" + "="*60)
    print("  CONFUSION MATRIX GENERATION SUMMARY")
    print("="*60)
    for mod, acc in results.items():
        print(f"  {mod.upper()}: {acc*100:.2f}% - confusion_matrix_{mod}.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
