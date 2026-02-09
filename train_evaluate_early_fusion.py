
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import time

from multimodal_dataset import MULBDataset
from early_fusion_model import EarlyFusionModel

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
        start_time = time.time()
        
        for i, (hand_imgs, iris_imgs, labels) in enumerate(train_loader):
            hand_imgs = hand_imgs.to(device, non_blocking=True)
            iris_imgs = iris_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass (Early Fusion Model handles concatenation internally)
            outputs = model(hand_imgs, iris_imgs)
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
        for hand_imgs, iris_imgs, labels in test_loader:
            hand_imgs = hand_imgs.to(device, non_blocking=True)
            iris_imgs = iris_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(hand_imgs, iris_imgs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    print("\nEvaluation Results:", flush=True)
    print(f"Accuracy:  {acc:.4f}", flush=True)
    print(f"Precision: {precision:.4f}", flush=True)
    print(f"Recall:    {recall:.4f}", flush=True)
    print(f"F1 Score:  {f1:.4f}", flush=True)
    
    return acc, precision, recall, f1

def main():
    # Parameters for EARLY FUSION Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DATASET_ROOT = r"C:\Users\win10\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset_224"
    
    print(f"Checking GPU availability...", flush=True)
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}", flush=True)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("CUDA NOT Available. Using CPU.", flush=True)
        device = torch.device("cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloaders
    print("Preparing dataset...", flush=True)
    if not os.path.exists(DATASET_ROOT):
         print(f"Error: Dataset not found at {DATASET_ROOT}")
         return

    full_dataset = MULBDataset(DATASET_ROOT, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Total samples: {len(full_dataset)}, Total classes: {num_classes}")
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Model Setup (Early Fusion)
    print("Initializing Early Fusion Model...")
    model = EarlyFusionModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print("Starting Early Fusion training...", flush=True)
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS)
    
    # Evaluation
    print("Starting evaluation...", flush=True)
    evaluate_model(model, test_loader, device)
    
    # Save Model
    torch.save(model.state_dict(), "early_fusion_mulb_optimized.pth")
    print("Model saved to early_fusion_mulb_optimized.pth")

if __name__ == "__main__":
    main()
