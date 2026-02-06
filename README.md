# Multimodal Biometric Fusion System (Hand + Iris)


## 🌟 What is this project? (In Simple Terms)

Imagine a high-security lock that doesn't just ask for a key, but checks **two** unique parts of your body to verify your identity:
1.  **Your Hand**: The shape and vein patterns of your hand.
2.  **Your Eye (Iris)**: The unique patterns in the colored part of your eye.

This project builds an **Artificial Intelligence (AI)** system that looks at pictures of both your hand and your eye *at the same time*. It combines the information from both sources to decide exactly who you are.

By "fusing" these two pieces of evidence, the system becomes:
*   **More Accurate**: If one picture is blurry, the other can still confirm your identity.
*   **More Secure**: It is much harder to fake both a hand and an iris than just one of them.

---

## 🔧 Technical Overview

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

### Core Training Scripts
*   `train_evaluate.py`: Main entry point for Late Fusion training.
*   `train_evaluate_early_fusion.py`: Early Fusion training script.
*   `train_unimodal_optimized.py`: Optimized unimodal training with parallel GPU loading.
*   `preprocess_gpu.py`: Pre-resize images to 224x224 for faster loading.

### Models
*   `late_fusion_model.py`: Dual-ResNet Late Fusion architecture.
*   `early_fusion_model.py`: Single-ResNet Early Fusion architecture.
*   `unimodal_model.py`: Single-modality ResNet model.

### Datasets
*   `multimodal_dataset.py`: Paired Hand + Iris dataset for fusion models.
*   `unimodal_dataset_optimized.py`: Optimized single-modality dataset.

### Evaluation
*   `evaluate_confusion_matrix.py`: Late Fusion confusion matrix.
*   `evaluate_confusion_matrix_early.py`: Early Fusion confusion matrix.
*   `evaluate_confusion_matrix_unimodal.py`: Hand/Iris unimodal confusion matrices.
*   `compare_all_models.py`: Compare all model architectures.


## ⚔️ Model Comparison (Fusion vs. Unimodal)

We compared **Early Fusion**, **Late Fusion**, and **Unimodal** baselines.

| Model | Test Accuracy | Train Time | Description |
| --- | --- | --- | --- |
| **Iris Only** 🏆 | **99.73%** | 7.2s/epoch | Single ResNet18 (Iris) |
| **Late Fusion** | 98.38% | 13s/epoch | 2x ResNet18 (Feature Concat) |
| **Early Fusion** | 92.66% | 8.5s/epoch | 1x ResNet18 (Input Concat) |
| **Hand Only** | 88.59% | 7.6s/epoch | Single ResNet18 (Hand) |

### Key Insights:
- **Iris-only model achieves highest accuracy (99.73%)** - iris patterns are highly discriminative
- **Late Fusion** combines both modalities for multi-factor authentication
- **Hand biometrics** is more challenging with 188 classes

👉 **[Read the Full Comparison Report](FUSION_COMPARISON.md)**

---

## 🎯 Unimodal Training

Train individual modality models (Hand-only or Iris-only):

```bash
# Pre-resize all images first (one-time)
python preprocess_gpu.py

# Train both modalities with optimized parallel loading
python train_unimodal_optimized.py --modality both --epochs 10

# Train single modality
python train_unimodal_optimized.py --modality iris --epochs 10
python train_unimodal_optimized.py --modality hand --epochs 10
```

### Generate Confusion Matrices:
```bash
python evaluate_confusion_matrix_unimodal.py --modality both
```

Generates: `confusion_matrix_hand.png`, `confusion_matrix_iris.png`

## 📊 Performance Metrics

The system evaluates performance using:
*   **Accuracy**
*   **Precision** (Weighted)
*   **Recall** (Weighted)
*   **F1 Score** (Weighted)

---
*Created by [Your Name/Username]*