# ⚔️ Fusion Strategy Comparison: Early vs. Late Fusion

This document documents the experimental results comparing two multimodal fusion strategies—**Early Fusion** and **Late Fusion**—applied to the MULB (Hand + Iris) biometric dataset.

## 📊 Performance Summary

| Metric | Late Fusion 🔴 | Early Fusion 🟠 | Winner 🏆 |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **98.38%** | 92.66% | **Late Fusion** |
| **Training Speed** | ~13.0 sec / epoch | **~8.5 sec / epoch** | Early Fusion |
| **Model Size** | Two Backbones (2x ResNet18) | Single Backbone (1x ResNet18 modified) | Early Fusion |
| **Convergence** | Stable, monotonic improvement | Fast initially, unstable at later epochs | Late Fusion |
| **Complexity** | High (Dual stream) | Low (Single stream) | Early Fusion |

*Note: Results based on 10 epochs of training with Batch Size 64 on RTX 3060.*

---

## 🏗️ Architectural Differences

### 1. Late Fusion (Feature Level)
The **Late Fusion** model processes each modality independently through its own dedicated Convolutional Neural Network (CNN). The high-level features extracted from both modalities are merged only at the final stage.

*   **Hand Input** $\rightarrow$ `ResNet18_Hand` $\rightarrow$ Feature Vector A (512 dim)
*   **Iris Input** $\rightarrow$ `ResNet18_Iris` $\rightarrow$ Feature Vector B (512 dim)
*   **Fusion**: Concatenation $[A, B]$ $\rightarrow$ 1024 dim
*   **Classifier**: MLP $\rightarrow$ Class Prediction

**Why it works well**: It allows the network to learn modality-specific features (e.g., texture for iris vs. geometry for hand) without interference. The fusion layer then learns the *correlation* between these distinct high-level abstractions.

### 2. Early Fusion (Data Level)
The **Early Fusion** model merges the raw data immediately before processing.

*   **Inputs**: Hand Image (3 ch) + Iris Image (3 ch)
*   **Fusion**: Stacked along channel dimension $\rightarrow$ 6-channel Input Tensor
*   **Processing**: `ResNet18_Modified` (Conv1 accepts 6 channels) $\rightarrow$ Features $\rightarrow$ Class Prediction

**Why it's faster**: It uses only *one* backbone network instead of two, effectively halving the number of convolutional operations required per forward pass.
**Why it scored lower**: The network must learn to process two vastly different visual domains (macros-structure of hands vs. micro-texture of irises) simultaneously in the same convolutional filters. This early mixing adds noise and makes optimization harder.

---

## 📉 Visualizations

Confusion matrices were generated for both models:

*   **Late Fusion**: `confusion_matrix_late.png` (Shows high diagonal dominance)
*   **Early Fusion**: `confusion_matrix_early.png` (Good diagonal, but more off-diagonal noise)

## 💡 Conclusion

For this specific multimodal biometric task, **Late Fusion is superior**. 
While Early Fusion is ~35% faster to train, the **+5.7% accuracy gap** makes Late Fusion the preferred choice for a security-critical application like biometric authentication. The independence of feature extraction seems crucial when merging modalities with such different visual characteristics.
