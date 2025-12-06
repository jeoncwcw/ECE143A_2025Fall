This repository implements advanced neural sequence decoders for the **Brain-to-Text '24 Benchmark**. Moving beyond standard RNN baselines, we introduce **Transformer with InterCTC** and **Conformer-based architectures** enhanced with regularization techniques to decode neural activity into phonemes effectively.

## 📌 Project Abstract
We address the limitations of GRU-based decoders by implementing a **Time-Masked Transformer** with **Intermediate CTC Loss**. Our approach improves robustness against limited neural data and captures long-range dependencies better than the baseline. 
**Result:** Our best configuration achieves a **17.59% Phoneme Error Rate (PER)** on the validation set, significantly outperforming the GRU baseline (22.50%).

## 🚀 Key Contributions & Implementation
*Unlike the existing GRU implementation, the following features were newly developed by our team:*

### 1. Transformer & Conformer Architecture
**Code:** `src/transformer_decoder/bit.py`
- **BiT_Phoneme:** A custom Transformer model adapted for neural speech decoding.
- **Conformer Blocks:** Integrated **Convolutional Modules** within Transformer layers to capture both local temporal spikes and global context.

### 2. Advanced Regularization
**Code:** `src/transformer_decoder/loss.py`, `src/transformer_decoder/augmentations.py`
- **Time Masking (SpecAugment):** Applied masking in the time domain to prevent overfitting to specific neural frames.
- **Intermediate CTC (InterCTC):** Added auxiliary CTC loss at intermediate layers to accelerate convergence and improve deep network training.

## 📊 Performance Results
| Model | Configuration | PER (%) |
| :--- | :--- | :--- |
| **GRU (Baseline)** | Default | 22.50% |
| **Transformer** | + Time Masking | 19.05% |
| **Transformer (Ours)** | **+ Time Masking + Inter CTC** | **17.59%** |

## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

pip install -e .

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. Train model: `python ./scripts/train_model.py`

