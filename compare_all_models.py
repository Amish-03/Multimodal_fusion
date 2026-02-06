
"""
Compare All Fusion Strategies: Unimodal (Hand, Iris) vs Multimodal (Early, Late Fusion).
This script trains and evaluates all four models and produces a comparison table.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

from unimodal_dataset import UnimodalDataset
from unimodal_model import UnimodalModel
from multimodal_dataset import MULBDataset
from early_fusion_model import EarlyFusionModel
from late_fusion_model import LateFusionModel


def train_unimodal(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for images, labels in train_loader:
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

        epoch_time = time.time() - start_time
        print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%, Time: {epoch_time:.1f}s", flush=True)


def train_multimodal(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for hand_imgs, iris_imgs, labels in train_loader:
            hand_imgs = hand_imgs.to(device, non_blocking=True)
            iris_imgs = iris_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(hand_imgs, iris_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_time = time.time() - start_time
        print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%, Time: {epoch_time:.1f}s", flush=True)


def evaluate_unimodal(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc


def evaluate_multimodal(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for hand_imgs, iris_imgs, labels in test_loader:
            hand_imgs = hand_imgs.to(device, non_blocking=True)
            iris_imgs = iris_imgs.to(device, non_blocking=True)
            outputs = model(hand_imgs, iris_imgs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc


def main():
    # Parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DATASET_ROOT = r"C:\Users\win10\.cache\kagglehub\datasets\olankadhim\multimodal-biometric-dataset-mulb\versions\1\MULB dataset"
    
    # Device setup
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}\n", flush=True)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU.\n", flush=True)
        device = torch.device("cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {}

    # ========== 1. HAND ONLY ==========
    print("="*60)
    print("  TRAINING: HAND ONLY")
    print("="*60)
    hand_dataset = UnimodalDataset(DATASET_ROOT, modality='hand', transform=transform)
    num_classes = len(hand_dataset.classes)
    train_size = int(0.8 * len(hand_dataset))
    test_size = len(hand_dataset) - train_size
    train_ds, test_ds = random_split(hand_dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = UnimodalModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_unimodal(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    results['Hand Only'] = evaluate_unimodal(model, test_loader, device)
    torch.save(model.state_dict(), "hand_only_model.pth")
    print(f"  >> Hand Only Accuracy: {results['Hand Only']*100:.2f}%\n")

    # ========== 2. IRIS ONLY ==========
    print("="*60)
    print("  TRAINING: IRIS ONLY")
    print("="*60)
    iris_dataset = UnimodalDataset(DATASET_ROOT, modality='iris', transform=transform)
    train_size = int(0.8 * len(iris_dataset))
    test_size = len(iris_dataset) - train_size
    train_ds, test_ds = random_split(iris_dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = UnimodalModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_unimodal(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    results['Iris Only'] = evaluate_unimodal(model, test_loader, device)
    torch.save(model.state_dict(), "iris_only_model.pth")
    print(f"  >> Iris Only Accuracy: {results['Iris Only']*100:.2f}%\n")

    # ========== 3. EARLY FUSION ==========
    print("="*60)
    print("  TRAINING: EARLY FUSION (Hand + Iris)")
    print("="*60)
    mm_dataset = MULBDataset(DATASET_ROOT, transform=transform)
    train_size = int(0.8 * len(mm_dataset))
    test_size = len(mm_dataset) - train_size
    train_ds, test_ds = random_split(mm_dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = EarlyFusionModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_multimodal(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    results['Early Fusion'] = evaluate_multimodal(model, test_loader, device)
    torch.save(model.state_dict(), "early_fusion_comparison.pth")
    print(f"  >> Early Fusion Accuracy: {results['Early Fusion']*100:.2f}%\n")

    # ========== 4. LATE FUSION ==========
    print("="*60)
    print("  TRAINING: LATE FUSION (Hand + Iris)")
    print("="*60)
    model = LateFusionModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_multimodal(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    results['Late Fusion'] = evaluate_multimodal(model, test_loader, device)
    torch.save(model.state_dict(), "late_fusion_comparison.pth")
    print(f"  >> Late Fusion Accuracy: {results['Late Fusion']*100:.2f}%\n")

    # ========== FINAL COMPARISON ==========
    print("\n" + "="*60)
    print("  FINAL COMPARISON: Unimodal vs Multimodal Fusion")
    print("="*60)
    print(f"  {'Model':<20} | {'Accuracy':>10}")
    print("-"*35)
    for model_name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {model_name:<20} | {acc*100:>9.2f}%")
    print("="*60)
    
    # Fusion Benefit Analysis
    best_unimodal = max(results['Hand Only'], results['Iris Only'])
    best_fusion = max(results['Early Fusion'], results['Late Fusion'])
    fusion_gain = (best_fusion - best_unimodal) * 100
    
    print(f"\n  🔬 FUSION BENEFIT: +{fusion_gain:.2f}% improvement over best single modality")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
