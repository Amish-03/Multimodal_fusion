# ⚔️ Fusion Strategy Comparison: Early vs. Late vs. Unimodal

This document documents the experimental results comparing multimodal fusion strategies—**Early Fusion**, **Late Fusion**, and **Unimodal** baselines—applied to the MULB (Hand + Iris) biometric dataset.

## 📊 Performance Summary

### Complete Model Comparison

| Model | Test Accuracy | Training Speed | Model Size | Winner 🏆 |
| :--- | :--- | :--- | :--- | :--- |
| **Late Fusion** 🔴 | **98.38%** | ~13.0 sec/epoch | 2x ResNet18 | ✅ Best Accuracy |
| **Early Fusion** 🟠 | 92.66% | ~8.5 sec/epoch | 1x ResNet18 | Fastest |
| **Iris Only** 🔵 | 99.73% | ~7.2 sec/epoch | 1x ResNet18 | Single Modality Best |
| **Hand Only** 🟢 | 88.59% | ~7.6 sec/epoch | 1x ResNet18 | Baseline |

### Unimodal vs. Fusion Analysis

| Metric | Hand Only | Iris Only | Late Fusion | Early Fusion |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 88.59% | **99.73%** | 98.38% | 92.66% |
| **Precision** | 0.8844 | **0.9973** | 0.9838 | 0.9266 |
| **Recall** | 0.8175 | **0.9973** | 0.9838 | 0.9266 |
| **F1 Score** | 0.8160 | **0.9973** | 0.9838 | 0.9266 |

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

Confusion matrices were generated for all models:

### Fusion Models
*   **Late Fusion**: `confusion_matrix_late.png` (Shows high diagonal dominance)
*   **Early Fusion**: `confusion_matrix_early.png` (Good diagonal, but more off-diagonal noise)

### Unimodal Models
*   **Hand Only**: `confusion_matrix_hand.png` (88.59% accuracy, 188 classes)
*   **Iris Only**: `confusion_matrix_iris.png` (99.73% accuracy, near-perfect diagonal)

## 💡 Conclusion

### Key Findings:

1. **Iris modality alone achieves 99.73%** - outperforming all fusion methods, suggesting iris is the dominant biometric trait.
2. **Hand modality alone achieves 88.59%** - a challenging biometric with 188 classes.
3. **Late Fusion (98.38%)** provides robust multi-factor authentication but doesn't exceed unimodal iris.
4. **Early Fusion (92.66%)** is fastest but loses accuracy due to early feature mixing.

### Recommendations:
- For **maximum accuracy**: Use Iris-only model (99.73%)
- For **multi-factor security**: Use Late Fusion (combines both modalities)
- For **speed-critical applications**: Use Early Fusion with acceptable accuracy trade-off

The independence of feature extraction in Late Fusion seems crucial when merging modalities with different visual characteristics, but the exceptional performance of iris-only suggests the dataset's iris patterns are highly discriminative.
