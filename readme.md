# AST Voice Biomarker Classifier with Mixture of Experts (MoE)

This repository contains the implementation of a two-stage training pipeline that fine-tunes an Audio Spectrogram Transformer (AST) on a patient vocal dataset and subsequently sparsifies it using a custom Mixture of Experts (MoE) architecture.

## Dataset & Preprocessing

The model is trained on a custom patient-vocal-dataset containing **1,036 `.wav` audio files** classified into three voice biomarker categories:
Dataset - https://www.kaggle.com/datasets/subhajournal/patient-health-detection-using-vocal-audio/data
- **Normal:** 560 files  
- **Vox senilis:** 392 files  
- **Laryngozele:** 84 files  

### Preprocessing

- Audio loaded via **librosa** at a **16 kHz sampling rate**.
- Converted to spectrograms using the Hugging Face **ASTFeatureExtractor** (`padding="max_length"`).
- Data is split **80% for training** and **20% for testing/validation**.

---

## Implementation Details

The training pipeline consists of **two distinct stages**:

### Stage 1: Dense Model Fine-Tuning

**Base Model:** `MIT/ast-finetuned-audioset-10-10-0.4593`

- The pre-trained AST model's **classification head is re-initialized** for the **3 target classes**.
- Trained for **5 epochs** using standard **Cross-Entropy Loss** to establish a strong dense baseline.
- Model is saved to `./stage2_laryngeal_model`.

---

### Stage 2: MoE Sparsification

**Architecture Modification:**  
The dense model from Stage 1 is loaded. For the **top half of the AST encoder layers** (`layer index ≥ num_layers // 2`), the standard feed-forward networks (**intermediate** and **output**) are stripped and replaced with a custom **ASTMoEFFN** module.

#### MoE Configuration

- `NUM_EXPERTS = 4`
- `TOP_K = 2` (Tokens are routed to the top 2 experts).

Original FFN output logic is bypassed using an **FFNIdentity** module, allowing the **ASTMoEFFN** to handle the full transformation before the residual connection.

#### Freezing

All **non-MoE parameters (lower encoder layers)** are frozen. Only the **new routing mechanisms and expert weights** are trained, resulting in **113,356,800 trainable parameters**.

#### Custom Trainer (AudioMoETrainer)

Implements a **custom training loop** to inject an **auxiliary load-balancing loss**. This ensures tokens are distributed evenly across the **4 experts**, preventing routing collapse.

`AUX_LOSS_COEF = 0.01`

Final sparsified model is saved to `./ast-voice-moe-stage3`.

---

## Results

The models were evaluated on the **20% test split**, tracking **Accuracy, Weighted F1, Precision, and Recall**.

### Stage 1: Dense Model Results

The dense fine-tuning achieved highly accurate results quickly, **peaking at Epoch 4** before slightly overfitting.

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|------|---------------|----------------|----------|----------|
| 1 | 0.5026 | 0.4343 | 84.13% | 0.8095 |
| 3 | 0.1242 | 0.4957 | 85.57% | 0.8575 |
| 4 | 0.0015 | 0.3333 | 93.26% | 0.9320 |
| 5 | 0.0004 | 0.3370 | 92.78% | 0.9278 |

---

### Stage 2: MoE Model Results

After injecting the MoE layers and freezing the bottom half of the network, the model successfully **recovered the baseline's performance while benefiting from sparse top-K routing**.

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|------|---------------|----------------|----------|----------|
| 1 | 0.1275 | 0.4210 | 91.34% | 0.9131 |
| 3 | 0.1234 | 0.4553 | 93.26% | 0.9321 |
| 6 | 0.1221 | 0.5269 | 92.78% | 0.9276 |
| 10 | 0.1155 | 0.5332 | 91.82% | 0.9176 |

---

## Conclusion

The **MoE implementation successfully matches the peak accuracy of the fully dense model (93.26%)** while routing tokens through **only a subset of the parameters in the upper layers**.