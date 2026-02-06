# Multimodal Biometric Fusion System (Hand + Iris)

This repository contains the implementation of a **Multimodal Biometric Recognition System** that fuses **Hand** and **Iris** modalities using a **Late Fusion** Deep Learning approach. The system is built using PyTorch and utilizes pre-trained ResNet18 models for feature extraction.

## 📌 Project Overview

Biometric systems based on a single modality can be affected by noise, non-universality, and spoofing attacks. This project employs a **Multimodal** approach to improve robustness and accuracy by combining two distinct biometric traits:
1.  **Hand Geometry/Shape**
2.  **Iris Patterns**

The core of the system is a **Late Fusion Neural Network** that extracts features from both images independently and fuses them at the classification stage.

## 🏗️ Model Architecture: Late Fusion

The model uses a "Late Fusion" strategy, where separate neural networks process each modality, and their high-level features are concatenated before the final classification.

### Architecture Details:
*   **Backbone 1 (Hand Branch)**: ResNet18 (Pre-trained on ImageNet). Features are extracted from the layer before the final classification head (512 dimensions).
*   **Backbone 2 (Iris Branch)**: ResNet18 (Pre-trained on ImageNet). Features are extracted similarly (512 dimensions).
*   **Fusion Layer**: The feature vectors from both backbones are concatenated to form a 1024-dimensional joint feature vector.
*   **Classifier (Head)**: A fully connected network processes the fused features:
    *   `Linear(1024 -> 512)` -> `ReLU` -> `Dropout(0.5)` -> `Linear(512 -> Num_Classes)`

This architecture allows the model to learn specific features for each biometric trait and then learn the correlation between them for the final decision.

## 📂 Dataset

The project is designed to work with the **MULB (Multimodal Biometric Dataset)**.
*   **Source**: The dataset is downloaded using the `kagglehub` library (or assumed to be locally available).
*   **Structure**:
    *   `hand dataset/`: Contains subfolders for each subject's hand images.
    *   `iris dataset/`: Contains subfolders for each subject's iris images.
*   **Preprocessing**:
    *   Images are resized to `224x224`.
    *   Normalized using ImageNet mean and standard deviation.

## 🛠️ Installation & Requirements

Ensure you have Python installed. Install the dependencies using pip:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow pandas kagglehub
```

## 🚀 Usage

### 1. Download Data
The data downloading logic is handled in `download_data.py` (or integrated into the main workflow). Ensure the dataset is present at the path specified in `train_evaluate.py`.

### 2. Train and Evaluate
Run the main training script to train the model and evaluate it on the test set.

```bash
python train_evaluate.py
```

**Key Parameters in `train_evaluate.py`:**
*   `BATCH_SIZE`: 16 (Adjust based on VRAM)
*   `NUM_EPOCHS`: 5 (Default)
*   `LEARNING_RATE`: 0.001

### 3. Output
*   **Training Logs**: Epoch-wise Loss and Accuracy printed to console.
*   **Evaluation Metrics**: Final Accuracy, Precision, Recall, and F1 Score on the test set.
*   **Saved Model**: The trained model weights are saved as `late_fusion_mulb_model.pth`.

## 📁 File Structure

*   `train_evaluate.py`: The main entry point. Handles data loading, model initialization, training loop, and evaluation.
*   `late_fusion_model.py`: Defines the `LateFusionModel` class with the dual-ResNet architecture.
*   `multimodal_dataset.py`: Defines the `MULBDataset` class, handling the pairing of hand and iris images for each subject.
*   `download_data.py`: Script to download the dataset from Kaggle.
*   `analyze_dataset.py`: Helper script to analyze dataset distribution (optional).

## 📊 Performance Metrics

The system evaluates performance using:
*   **Accuracy**
*   **Precision** (Weighted)
*   **Recall** (Weighted)
*   **F1 Score** (Weighted)

---
*Created by [Your Name/Username]*