# Fetal Cardiac Abnormality Detection using QViT (Ultrasound Images and NIFECG)

## Overview

This project presents a deep learning framework for detecting fetal cardiac abnormalities by integrating structural analysis from ultrasound images with functional analysis from ECG signals. The approach leverages transformer-based architectures along with a quantum-enhanced model to improve performance on limited and complex medical datasets.

The objective is to move toward a multi-modal diagnostic system that considers both anatomical structure and electrical activity of the fetal heart for more comprehensive assessment.

---

## Objectives

* Detect fetal cardiac abnormalities from ultrasound images
* Analyze cardiac function using ECG signals
* Apply transformer-based architectures such as ViT, Swin Transformer, and ConvNeXt
* Incorporate quantum-enhanced feature learning using QViT
* Provide model interpretability using Score-CAM

---

## Methodology

### Ultrasound Pipeline

1. Image preprocessing (resizing, normalization, CLAHE enhancement)
2. Data augmentation to improve generalization
3. Classification using transformer-based architectures (ViT, Swin-T, ConvNeXt, QViT)
4. Visualization of model attention using Score-CAM

### ECG Pipeline

1. Acquisition of abdominal ECG signals
2. Frequency-based filtering to isolate fetal ECG
3. Segmentation of signals into fixed-length windows
4. Conversion of signals into spectrogram representations
5. Classification using ViT and QViT

---

## Model Details

* Vision Transformer (ViT-Tiny Patch16-224) used as the base architecture
* Quantum Vision Transformer (QViT) incorporates a variational quantum circuit
* Quantum encoding using rotation gates and measurement through expectation values
* Two-phase training strategy involving partial and full model fine-tuning

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC

---

## Dataset

### Ultrasound

* Labeled fetal ultrasound images
* Binary classification: normal and abnormal

### ECG (PhysioNet)

* 26 recordings (arrhythmia and normal)
* Signals segmented and converted into spectrogram images for model input

---

## Results

The proposed approach demonstrates strong performance across evaluation metrics, with particularly high recall, which is important for medical diagnosis. The results indicate that the model is effective in capturing both structural and functional abnormalities.

---

## Extended Study

The work is extended to include ECG-based functional analysis. This allows the detection of abnormalities such as arrhythmias that may not be visible in ultrasound imaging. The framework demonstrates the feasibility of using a unified transformer-based model for both imaging and signal data.

---

## Future Work

* Integration of ultrasound and ECG into a unified multi-modal model
* Expansion of dataset size for improved generalization
* Deployment in real-time clinical settings

---

## Tech Stack

* Python
* PyTorch
* NumPy, OpenCV
* Matplotlib
