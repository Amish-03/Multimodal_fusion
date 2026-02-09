
"""
Train and evaluate unimodal models (Hand-only or Iris-only).
Compare results against fusion models.
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

from unimodal_dataset import UnimodalDataset
from unimodal_model import UnimodalModel


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
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
        print(f"Epoch [{epoch+1}/{num_epochs}] Complete. Time: {end_time - start_time:.2f}s. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%", flush=True)


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


def main():
    parser = argparse.ArgumentParser(description='Train Unimodal Biometric Model')
    parser.add_argument('--modality', type=str, choices=['hand', 'iris'], required=True, help='Modality to train on: hand or iris')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    # Parameters
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = 0.001
    MODALITY = args.modality
    DATASET_ROOT = r"C:\Users\win10\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset"
    
    print(f"\n{'='*60}")
    print(f"  Training UNIMODAL Model: {MODALITY.upper()} ONLY")
    print(f"{'='*60}\n")

    # Device setup
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("CUDA NOT Available. Using CPU.", flush=True)
        device = torch.device("cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloaders
    print(f"Preparing {MODALITY} dataset...", flush=True)
    full_dataset = UnimodalDataset(DATASET_ROOT, modality=MODALITY, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Total samples: {len(full_dataset)}, Total classes: {num_classes}")
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model Setup
    model = UnimodalModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print(f"Starting {MODALITY.upper()} training...", flush=True)
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS)
    
    # Evaluation
    print(f"\nEvaluating {MODALITY.upper()} model...", flush=True)
    acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print(f"  {MODALITY.upper()} MODEL RESULTS")
    print("="*60)
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("="*60 + "\n")
    
    # Save Model
    model_path = f"{MODALITY}_only_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
