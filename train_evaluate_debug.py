
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from multimodal_dataset import MULBDataset
from late_fusion_model import LateFusionModel

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
        for i, (hand_imgs, iris_imgs, labels) in enumerate(train_loader):
            hand_imgs = hand_imgs.to(device)
            iris_imgs = iris_imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(hand_imgs, iris_imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 5 == 0:
                print(f"  Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}", flush=True)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Complete. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%", flush=True)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for hand_imgs, iris_imgs, labels in test_loader:
            hand_imgs = hand_imgs.to(device)
            iris_imgs = iris_imgs.to(device)
            labels = labels.to(device)
            
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
    # Parameters for DEBUG
    BATCH_SIZE = 8
    NUM_EPOCHS = 2
    LEARNING_RATE = 0.001
    DATASET_ROOT = r"C:\Users\amish\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloaders
    print("Preparing dataset...", flush=True)
    full_dataset = MULBDataset(DATASET_ROOT, transform=transform)
    num_classes = len(full_dataset.classes)
    
    # DEBUG: Use only a small subset (e.g., 50 samples)
    if len(full_dataset) > 50:
         indices = list(range(50))
         full_dataset = Subset(full_dataset, indices)
         print("DEBUG MODE: Dataset truncated to 50 samples.", flush=True)

    print(f"Total samples: {len(full_dataset)}, Total classes (potential): {num_classes}", flush=True)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Setup
    model = LateFusionModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print("Starting training...", flush=True)
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS)
    
    # Evaluation
    print("Starting evaluation...", flush=True)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
