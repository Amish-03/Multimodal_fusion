
"""
Optimized unimodal training with pre-resized images and parallel GPU loading.
Trains both modalities sequentially with maximum GPU efficiency.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import argparse

from unimodal_dataset_optimized import UnimodalDatasetOptimized
from unimodal_model import UnimodalModel


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, modality=""):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        print(f"[{modality.upper()}] Epoch {epoch+1}/{num_epochs}", flush=True)
        
        for i, (images, labels) in enumerate(train_loader):
            # Non-blocking async transfer to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 10 == 0:
                print(f"  Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}", flush=True)

        end_time = time.time()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"[{modality.upper()}] Epoch [{epoch+1}/{num_epochs}] Done. Time: {end_time - start_time:.2f}s. Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%", flush=True)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return acc, precision, recall, f1


def train_single_modality(modality, dataset_root, batch_size, num_epochs, device):
    """Train a single modality and return results."""
    print(f"\n{'='*60}")
    print(f"  TRAINING UNIMODAL MODEL: {modality.upper()}")
    print(f"{'='*60}\n")
    
    # Optimized transforms - NO RESIZE needed (pre-processed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloaders with parallel workers
    print(f"Loading pre-resized {modality} dataset...", flush=True)
    full_dataset = UnimodalDatasetOptimized(dataset_root, modality=modality, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Total samples: {len(full_dataset)}, Classes: {num_classes}")
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # High num_workers for parallel loading, pin_memory for async GPU transfer
    num_workers = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4  # Prefetch 4 batches per worker
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model Setup
    model = UnimodalModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print(f"Starting {modality.upper()} training with parallel loading...", flush=True)
    train_start = time.time()
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs, modality=modality)
    train_time = time.time() - train_start
    
    # Evaluation
    print(f"\nEvaluating {modality.upper()} model...", flush=True)
    acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
    
    # Save Model
    model_path = f"{modality}_only_optimized.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return {
        'modality': modality,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time
    }


def main():
    parser = argparse.ArgumentParser(description='Train Optimized Unimodal Models with Pre-resized Images')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--modality', type=str, choices=['hand', 'iris', 'both'], default='both', 
                        help='Modality to train: hand, iris, or both')
    args = parser.parse_args()

    # Pre-resized dataset path
    DATASET_ROOT = r"C:\Users\win10\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset_224"
    
    print("\n" + "="*60)
    print("  OPTIMIZED UNIMODAL TRAINING")
    print("  Using Pre-resized Images + Parallel GPU Loading")
    print("="*60)

    # Check if pre-resized dataset exists
    if not os.path.exists(DATASET_ROOT):
        print(f"\n[ERROR] Pre-resized dataset not found at:\n  {DATASET_ROOT}")
        print("\nPlease run 'python preprocess_gpu.py' first to pre-resize images.")
        return

    # Device setup with optimizations
    if torch.cuda.is_available():
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}", flush=True)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Auto-tune for input size
    else:
        print("\nCUDA NOT Available. Using CPU.", flush=True)
        device = torch.device("cpu")

    results = []
    modalities = ['hand', 'iris'] if args.modality == 'both' else [args.modality]
    
    total_start = time.time()
    
    for modality in modalities:
        result = train_single_modality(
            modality=modality,
            dataset_root=DATASET_ROOT,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=device
        )
        results.append(result)
    
    total_time = time.time() - total_start

    # Print Summary
    print("\n" + "="*60)
    print("  UNIMODAL TRAINING SUMMARY")
    print("="*60)
    for r in results:
        print(f"\n  {r['modality'].upper()} MODEL:")
        print(f"    Accuracy:  {r['accuracy']*100:.2f}%")
        print(f"    Precision: {r['precision']:.4f}")
        print(f"    Recall:    {r['recall']:.4f}")
        print(f"    F1 Score:  {r['f1']:.4f}")
        print(f"    Train Time: {r['train_time']:.2f}s")
    print(f"\n  Total Time: {total_time:.2f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
